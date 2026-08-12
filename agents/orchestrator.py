"""
Orchestrator — the actual "agentic" part of the system.

Uses LangGraph to define a graph where each specialist agent is a node.
For v1, all four specialist agents run in parallel (they're independent —
none needs another's output), then a synthesis node runs last and needs
all four to be done first. This is a fan-out/fan-in pattern, one of the
most common agentic graph shapes.

Run this directly to process one ticker end to end.
"""

import operator
import sys
from datetime import date
from pathlib import Path
from typing import Annotated, TypedDict

sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.graph import END, StateGraph

from agents import filings_analyst, osint_analyst, program_analyst, risk_analyst, sentiment_analyst
from schemas.models import (
    FilingsAnalysis,
    OSINTAnalysis,
    ProgramAnalysis,
    RiskAnalysis,
    SentimentAnalysis,
    SynthesisReport,
)
from synthesis import debate


class GraphState(TypedDict, total=False):
    ticker: str
    as_of_date: date | None
    filings: FilingsAnalysis
    program: ProgramAnalysis
    sentiment: SentimentAnalysis
    risk: RiskAnalysis
    osint: OSINTAnalysis
    # Annotated + operator.add: multiple specialist nodes can fail in the same fan-out
    # step, and without a reducer LangGraph can't merge concurrent writes to this key
    # (raises INVALID_CONCURRENT_GRAPH_UPDATE). operator.add concatenates the lists.
    errors: Annotated[list[str], operator.add]
    report: SynthesisReport


def _run_agent_safely(agent_module, state: GraphState, key: str) -> dict:
    """Wrap an agent call so one failing agent doesn't crash the whole graph."""
    try:
        result = agent_module.run(state["ticker"], as_of_date=state.get("as_of_date"))
        return {key: result}
    except Exception as e:
        return {"errors": [f"{agent_module.__name__} failed: {e}"]}


def filings_node(state: GraphState) -> dict:
    return _run_agent_safely(filings_analyst, state, "filings")


def program_node(state: GraphState) -> dict:
    return _run_agent_safely(program_analyst, state, "program")


def sentiment_node(state: GraphState) -> dict:
    return _run_agent_safely(sentiment_analyst, state, "sentiment")


def risk_node(state: GraphState) -> dict:
    return _run_agent_safely(risk_analyst, state, "risk")


def osint_node(state: GraphState) -> dict:
    return _run_agent_safely(osint_analyst, state, "osint")


def synthesis_node(state: GraphState) -> dict:
    missing = [k for k in ("filings", "program", "sentiment", "risk") if k not in state]
    if missing:
        return {
            "errors": [
                f"synthesis.debate skipped — missing outputs from: {missing}. "
                f"Errors so far: {state.get('errors', [])}"
            ]
        }
    try:
        report = debate.run(
            ticker=state["ticker"],
            filings=state["filings"],
            program=state["program"],
            sentiment=state["sentiment"],
            risk=state["risk"],
            osint=state.get("osint"),
            as_of_date=state.get("as_of_date"),
        )
        return {"report": report}
    except Exception as e:
        return {"errors": [f"{debate.__name__} failed: {e}"]}


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("filings", filings_node)
    graph.add_node("program", program_node)
    graph.add_node("sentiment", sentiment_node)
    graph.add_node("risk", risk_node)
    graph.add_node("osint", osint_node)
    graph.add_node("synthesis", synthesis_node)

    # fan-out: all specialist agents run in parallel from a lightweight "start" no-op node,
    # since LangGraph needs a single entry point but we want multiple parallel branches
    graph.add_node("start", lambda state: {})
    graph.set_entry_point("start")
    for node in ("filings", "program", "sentiment", "risk", "osint"):
        graph.add_edge("start", node)

    # fan-in: synthesis waits for all specialist nodes
    for node in ("filings", "program", "sentiment", "risk", "osint"):
        graph.add_edge(node, "synthesis")

    graph.add_edge("synthesis", END)

    return graph.compile()


def run_for_ticker(ticker: str, as_of_date: date | None = None) -> GraphState:
    app = build_graph()
    result = app.invoke({"ticker": ticker, "as_of_date": as_of_date, "errors": []})
    return result


if __name__ == "__main__":
    final_state = run_for_ticker("LMT")
    if final_state.get("errors"):
        print("Errors during run:", final_state["errors"])
    if "report" in final_state:
        print(final_state["report"].model_dump_json(indent=2))