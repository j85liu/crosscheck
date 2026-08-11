"""
Generic tool-calling agent loop — the shared harness intended to eventually replace the
one-shot structured_call() pattern each agent currently hand-rolls (build a prompt, ask
for JSON, parse it) with a real tool-use loop: the model can call real tools across
multiple turns before producing its final structured answer.

Termination: rather than guessing "done" from stop_reason (which the Anthropic API sets
to "tool_use" whenever ANY tool is called, including intermediate data-gathering calls —
not just when the agent is actually finished), this appends one more tool automatically,
submit_final_answer, whose input_schema mirrors the caller's output_schema. The agent is
instructed to call it exactly when it wants to end the task, which is an explicit signal
instead of an inferred one.

Not wired into any real agent yet — see llm/agent_loop.py's __main__ block for a
standalone smoke test with a fake tool.
"""

import json
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import BaseModel

from llm.client import get_client

SUBMIT_TOOL_NAME = "submit_final_answer"


def _format_kwargs(input_dict: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in input_dict.items())


def _submit_tool_schema(output_schema: type[BaseModel]) -> dict:
    return {
        "name": SUBMIT_TOOL_NAME,
        "description": (
            "Call this when you have finished gathering information and are ready to submit "
            "your final structured result. This is the only way to end the task — plain text "
            "or calling other tools alone does not end it."
        ),
        "input_schema": output_schema.model_json_schema(),
    }


def run_tool_agent(
    system_prompt: str,
    tools: list[dict],
    tool_dispatch: dict[str, Callable],
    user_prompt: str,
    output_schema: type[BaseModel],
    model: str,
    max_turns: int = 5,
    max_tokens: int = 2048,
) -> BaseModel:
    """
    Run a tool-calling loop until the model calls submit_final_answer (validated against
    output_schema) or max_turns is exhausted.

    tools: Anthropic tool-schema dicts (name/description/input_schema), passed straight
    through to the API — submit_final_answer is appended automatically, callers shouldn't
    include it themselves.
    tool_dispatch: maps each tool name in `tools` to the Python callable that executes it;
    called as func(**block.input). A tool raising an exception doesn't crash the loop —
    the error message is fed back to the model as the tool_result so it can adapt.
    """
    client = get_client()
    all_tools = tools + [_submit_tool_schema(output_schema)]
    messages: list[dict] = [{"role": "user", "content": user_prompt}]

    for _ in range(max_turns):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            tools=all_tools,
            messages=messages,
        )

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        submit_block = next((b for b in tool_use_blocks if b.name == SUBMIT_TOOL_NAME), None)
        if submit_block is not None:
            return output_schema(**submit_block.input)

        messages.append({"role": "assistant", "content": response.content})

        if not tool_use_blocks:
            # No tool call at all — only submit_final_answer legitimately ends the loop,
            # so nudge the model back on track instead of silently stopping here.
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You must call a tool, or call submit_final_answer when you're done. "
                        "A plain text response alone doesn't end the task."
                    ),
                }
            )
            continue

        tool_result_blocks = []
        for block in tool_use_blocks:
            print(f"[tool call] {block.name}({_format_kwargs(block.input)})")
            func = tool_dispatch.get(block.name)
            if func is None:
                content = f"Error: no tool named {block.name!r} is available."
            else:
                try:
                    result = func(**block.input)
                    content = result if isinstance(result, str) else json.dumps(result, default=str)
                except Exception as e:
                    content = f"Error calling {block.name}: {e}"
            tool_result_blocks.append({"type": "tool_result", "tool_use_id": block.id, "content": content})

        messages.append({"role": "user", "content": tool_result_blocks})

    raise RuntimeError(
        f"run_tool_agent exceeded max_turns={max_turns} without a {SUBMIT_TOOL_NAME} call. "
        f"Conversation so far:\n{json.dumps(messages, indent=2, default=str)}"
    )


if __name__ == "__main__":
    from llm.client import SPECIALIST_MODEL

    class LuckyResult(BaseModel):
        seed: str
        lucky_number: int
        commentary: str

    def get_lucky_number(seed: str) -> dict:
        return {"lucky_number": len(seed) * 7}

    lucky_tool = {
        "name": "get_lucky_number",
        "description": "Compute the lucky number for a given seed string.",
        "input_schema": {
            "type": "object",
            "properties": {"seed": {"type": "string", "description": "The seed string."}},
            "required": ["seed"],
        },
    }

    result = run_tool_agent(
        system_prompt=(
            "You are a test agent. Call get_lucky_number with the seed you're given, then "
            "call submit_final_answer with the seed, the lucky_number you got back, and a "
            "one-sentence commentary about it."
        ),
        tools=[lucky_tool],
        tool_dispatch={"get_lucky_number": get_lucky_number},
        user_prompt="The seed is 'crosscheck'.",
        output_schema=LuckyResult,
        model=SPECIALIST_MODEL,
    )
    print(result.model_dump_json(indent=2))
