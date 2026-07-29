"""
Thin wrapper around the Anthropic API.

Centralizes model selection so you can tune cost/quality in one place:
- SPECIALIST_MODEL: used by individual agents (filings, risk, etc.) — these
  mostly extract/format tool results, a simpler task, so a cheaper model works.
- REASONING_MODEL: used by the orchestrator and synthesis/debate step — this
  is where cross-source contradiction detection happens and needs stronger
  reasoning.

Requires ANTHROPIC_API_KEY in your environment (see .env.example).
"""

import json
import os

from anthropic import Anthropic

_client: Anthropic | None = None

SPECIALIST_MODEL = "claude-haiku-4-5-20251001"
REASONING_MODEL = "claude-sonnet-5"


def get_client() -> Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key, "
                "or export it directly in your shell."
            )
        _client = Anthropic(api_key=api_key)
    return _client


def _call_and_parse(prompt: str, model: str, max_tokens: int) -> dict:
    client = get_client()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    text = text.strip()
    # strip accidental markdown fences if the model adds them anyway
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def structured_call(prompt: str, model: str, max_tokens: int = 1024) -> dict:
    """
    Call Claude and ask for a JSON-only response, then parse it.

    Keep the prompt itself explicit about wanting ONLY JSON — no preamble,
    no markdown fences — since we parse the raw text directly. Retries once
    on a JSON parse failure, since that's usually just a one-off malformed
    sample rather than a systematic problem with the prompt.
    """
    try:
        return _call_and_parse(prompt, model, max_tokens)
    except json.JSONDecodeError:
        return _call_and_parse(prompt, model, max_tokens)


if __name__ == "__main__":
    # quick manual test — requires ANTHROPIC_API_KEY to be set
    result = structured_call(
        prompt='Respond with ONLY this JSON, no other text: {"status": "ok", "test": true}',
        model=SPECIALIST_MODEL,
        max_tokens=100,
    )
    print(result)