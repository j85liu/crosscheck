"""
CLI entry point — runs the full CrossCheck pipeline for one or more tickers
and prints per-agent diagnostics so a failure can be localized to a specific
tool/agent instead of just seeing "it broke".

Usage:
    python demo/app.py LMT
    python demo/app.py LMT RTX NOC RKLB
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from agents import orchestrator
from tools.gdelt import SEARCH_TERMS
from tools.sec_edgar import KNOWN_CIKS
from tools.usaspending import RECIPIENT_NAMES

REQUIRED_ENV_VARS = ["ANTHROPIC_API_KEY", "ALPHAVANTAGE_API_KEY"]
AGENT_ORDER = ["filings", "program", "sentiment", "risk", "osint"]

# A ticker only actually works if every tool that needs a ticker->ID mapping has one.
SUPPORTED_TICKERS = sorted(set(KNOWN_CIKS) & set(RECIPIENT_NAMES) & set(SEARCH_TERMS))


def check_env() -> bool:
    missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        print(f"Missing required environment variables: {', '.join(missing)}")
        print("Copy .env.example to .env and fill in real keys, or export them in your shell.")
        return False
    return True


def _agent_error(errors: list[str], agent_key: str) -> str | None:
    """Find the error orchestrator._run_agent_safely logged for this agent, if any."""
    module_name = f"agents.{agent_key}_analyst"
    for e in errors:
        if e.startswith(module_name):
            return e
    return None


def summarize_agent(key: str, state: dict) -> str:
    if key in state:
        return f"  [ok]   {key:<10} {state[key].summary[:90]}"
    err = _agent_error(state.get("errors", []), key)
    return f"  [FAIL] {key:<10} {err or 'no output and no logged error (unexpected)'}"


def run_ticker(ticker: str) -> None:
    print(f"\n{'=' * 60}\n{ticker}\n{'=' * 60}")
    start = time.perf_counter()
    try:
        state = orchestrator.run_for_ticker(ticker)
    except Exception as e:
        print(f"  [FAIL] pipeline crashed after {time.perf_counter() - start:.1f}s: {e}")
        return

    for key in AGENT_ORDER:
        print(summarize_agent(key, state))

    # errors not tied to one of the known agent keys above (e.g. a bug in orchestrator itself)
    known_prefixes = tuple(f"agents.{k}_analyst" for k in AGENT_ORDER)
    for e in state.get("errors", []):
        if not e.startswith(known_prefixes):
            print(f"  [FAIL] {e}")

    print(f"  ({time.perf_counter() - start:.1f}s total)")

    report = state.get("report")
    if report is None:
        print("  No synthesis report produced.")
        return

    print(f"\n  Headline: {report.headline}")
    print(f"  Key findings ({len(report.key_findings)}):")
    for finding in report.key_findings:
        print(f"    - {finding}")
    if report.contradictions:
        print(f"  Contradictions flagged ({len(report.contradictions)}):")
        for c in report.contradictions:
            print(f"    - [{c.severity}] {', '.join(c.agents_involved)}: {c.description}")
    else:
        print("  Contradictions flagged: none")


def main() -> None:
    tickers = [t.upper() for t in sys.argv[1:]]
    if not tickers:
        print("Usage: python demo/app.py TICKER [TICKER ...]")
        print(f"Supported tickers: {', '.join(SUPPORTED_TICKERS)}")
        sys.exit(1)

    if not check_env():
        sys.exit(1)

    unsupported = [t for t in tickers if t not in SUPPORTED_TICKERS]
    if unsupported:
        print(f"Not configured (missing CIK/recipient/search-term mapping): {', '.join(unsupported)}")
        print(f"Supported tickers: {', '.join(SUPPORTED_TICKERS)}")
        sys.exit(1)

    for ticker in tickers:
        run_ticker(ticker)


if __name__ == "__main__":
    main()
