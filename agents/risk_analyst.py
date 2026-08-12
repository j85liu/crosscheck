"""
Risk analyst agent.

Uses the generic tool-calling harness (llm/agent_loop.py): the model always checks the
ticker's own volatility first, then decides for itself whether an elevated reading
warrants a peer/sector comparison before flagging an anomaly — company-specific moves and
sector-wide moves should not both get flagged the same way.
"""

import functools
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.agent_loop import run_tool_agent
from llm.client import SPECIALIST_MODEL
from schemas.models import RiskAnalysis
from tools.alphavantage import compute_volatility_summary, get_peer_comparison

SYSTEM_PROMPT = """You are a risk analyst. You have two tools:

- get_volatility_summary: latest close price, 30-day and 90-day annualized realized
  volatility, and 30-day price change for one ticker. Always call this first, for the
  ticker you're analyzing.
- get_peer_comparison: the same volatility stats for a set of peer tickers (defaults to
  the other defense/aero tickers this project covers if you don't specify peer_tickers),
  plus their average 30-day volatility.

Workflow: call get_volatility_summary for the ticker first. If 30-day volatility looks
meaningfully elevated relative to that same ticker's own 90-day baseline, or there's a
large 30-day price move without an obvious multi-day trend, call get_peer_comparison
before deciding flagged_anomaly — a spike that matches the sector isn't a
company-specific anomaly, even if it's numerically elevated relative to that ticker's own
history. If the ticker's own volatility looks unremarkable relative to its own baseline,
you can decide without checking peers, though checking anyway to confirm normalcy is also
fine. Don't over-flag normal market noise as an anomaly.

When ready, call submit_final_answer with: ticker, realized_volatility_30d,
realized_volatility_90d, price_change_30d_pct (the exact figures from
get_volatility_summary — don't round beyond what you were given), flagged_anomaly,
anomaly_description (null if flagged_anomaly is false; if you checked peers, mention what
that comparison showed either way), and a 2-3 sentence plain-language summary.
"""

TOOLS = [
    {
        "name": "get_volatility_summary",
        "description": (
            "Fetch latest close price, 30-day and 90-day annualized realized volatility, "
            "and 30-day price change for one ticker."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Stock ticker, e.g. 'LMT'"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "get_peer_comparison",
        "description": (
            "Fetch the same volatility stats for a set of peer tickers plus their average "
            "30-day volatility, to judge whether an elevated reading is company-specific or "
            "sector-wide. If peer_tickers is omitted, defaults to the other defense/aero "
            "tickers covered by this project."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker being analyzed, e.g. 'LMT'"},
                "peer_tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional explicit peer list, e.g. ['RTX', 'NOC']. Omit to use the default set.",
                },
            },
            "required": ["ticker"],
        },
    },
]


def _tool_get_volatility_summary(ticker: str, as_of_date: date | None) -> dict:
    return compute_volatility_summary(ticker, as_of_date=as_of_date)


def _tool_get_peer_comparison(ticker: str, as_of_date: date | None, peer_tickers: list[str] | None = None) -> dict:
    return get_peer_comparison(ticker, peer_tickers, as_of_date=as_of_date)


def _build_tool_dispatch(as_of_date: date | None) -> dict:
    # as_of_date is baked in via functools.partial, not exposed as a tool parameter — the
    # LLM never sees or has to remember to pass it.
    return {
        "get_volatility_summary": functools.partial(_tool_get_volatility_summary, as_of_date=as_of_date),
        "get_peer_comparison": functools.partial(_tool_get_peer_comparison, as_of_date=as_of_date),
    }


def run(ticker: str, as_of_date: date | None = None) -> RiskAnalysis:
    system_prompt = SYSTEM_PROMPT
    if as_of_date is not None:
        system_prompt += (
            f"\n\nYou are analyzing data as of {as_of_date.isoformat()}. Do not reference or "
            "assume knowledge of anything after this date."
        )
    return run_tool_agent(
        system_prompt=system_prompt,
        tools=TOOLS,
        tool_dispatch=_build_tool_dispatch(as_of_date),
        user_prompt=f"Assess recent price/volatility risk for {ticker}.",
        output_schema=RiskAnalysis,
        model=SPECIALIST_MODEL,
    )


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "LMT"
    as_of = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else None
    analysis = run(ticker, as_of_date=as_of)
    print(analysis.model_dump_json(indent=2))
