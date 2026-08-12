"""
Filings analyst agent.

Uses the generic tool-calling harness (llm/agent_loop.py) instead of a fixed
pre-fetch-everything-then-prompt-once pattern: the model decides what to call and in
what order, given three tools — filing metadata, structured XBRL financial figures, and
(expensive, used selectively) full filing text. See SYSTEM_PROMPT for the guidance on
when full text is actually warranted.
"""

import functools
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.agent_loop import run_tool_agent
from llm.client import SPECIALIST_MODEL
from schemas.models import FilingsAnalysis
from tools.sec_edgar import get_company_facts, get_company_filings, get_filing_full_text

SYSTEM_PROMPT = """You are a financial filings analyst. You have three tools available:

- get_filings_metadata: recent filing types and dates (start here).
- get_financial_facts: structured XBRL financial figures (revenue, net income, diluted
  EPS, operating income) for the most recent several reported periods — real reported
  numbers, not text.
- get_filing_full_text: full visible text (truncated) of one specific filing. Expensive
  relative to the other two tools — only call this for a specific filing (identified by
  the accession_number and primary_document you got from get_filings_metadata) when
  something looks notable enough to warrant a closer look: a large swing in a financial
  metric between periods, an unusual cluster of 8-K filings in a short window, or an 8-K
  whose purpose you can't infer from its form type and date alone. Do not fetch full text
  for every filing by default — metadata and financial facts are often sufficient on
  their own, and pulling full text of everything is wasteful.

Typical workflow: call get_filings_metadata and get_financial_facts first to get an
overview, decide from those two whether anything warrants a closer look, then optionally
call get_filing_full_text for that specific filing before submitting your answer.

Each individual highlight's "sentiment" must be exactly one of "positive", "negative", or
"neutral" — never "mixed". If a single fact has both positive and negative implications
(e.g. revenue grew but margins compressed), split it into two separate highlight entries
rather than tagging one highlight "mixed". Only the report-level "overall_sentiment" may
be "mixed".

When you have enough information, call submit_final_answer with: ticker, company_name,
overall_sentiment, highlights (each with topic, detail, sentiment, source_form, and
filed_date matching one of the filing dates you observed — never a date range or
reporting period), and a 2-3 sentence plain-language summary. Don't invent qualitative
detail (e.g. MD&A narrative, specific risk factors) beyond what the tool data — including
any full text you fetched — actually supports.
"""

TOOLS = [
    {
        "name": "get_filings_metadata",
        "description": (
            "Fetch metadata (form type, filing date, accession number, primary document "
            "filename) for the 5 most recent 10-K/10-Q/8-K filings for a company. Always "
            "start here."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Stock ticker, e.g. 'LMT'"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "get_financial_facts",
        "description": (
            "Fetch structured XBRL financial figures (revenue, net income, diluted EPS, "
            "operating income) for the most recent several reported periods."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Stock ticker, e.g. 'LMT'"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "get_filing_full_text",
        "description": (
            "Fetch the full visible text (truncated to ~3000 chars) of one specific filing's "
            "primary document. Expensive — only call for a filing that looks notable enough to "
            "warrant a closer look, using the accession_number and primary_document from "
            "get_filings_metadata. Don't call this for every filing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker, e.g. 'LMT'"},
                "accession_number": {
                    "type": "string",
                    "description": "From get_filings_metadata, e.g. '0000936468-26-000123'",
                },
                "primary_document": {
                    "type": "string",
                    "description": "From get_filings_metadata, e.g. 'lmt-20260628.htm'",
                },
            },
            "required": ["ticker", "accession_number", "primary_document"],
        },
    },
]


def _tool_get_filings_metadata(ticker: str, as_of_date: date | None) -> dict:
    return get_company_filings(ticker, form_types=["10-K", "10-Q", "8-K"], limit=5, as_of_date=as_of_date)


def _tool_get_financial_facts(ticker: str, as_of_date: date | None) -> dict:
    return get_company_facts(ticker, as_of_date=as_of_date)


def _tool_get_filing_full_text(
    ticker: str, accession_number: str, primary_document: str, as_of_date: date | None
) -> str:
    return get_filing_full_text(ticker, accession_number, primary_document, as_of_date=as_of_date)


def _build_tool_dispatch(as_of_date: date | None) -> dict:
    # as_of_date is baked in here via functools.partial rather than exposed as a tool
    # parameter — the LLM never sees or has to remember to pass it; every call it makes
    # automatically respects the cutoff.
    return {
        "get_filings_metadata": functools.partial(_tool_get_filings_metadata, as_of_date=as_of_date),
        "get_financial_facts": functools.partial(_tool_get_financial_facts, as_of_date=as_of_date),
        "get_filing_full_text": functools.partial(_tool_get_filing_full_text, as_of_date=as_of_date),
    }


def run(ticker: str, as_of_date: date | None = None) -> FilingsAnalysis:
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
        user_prompt=f"Analyze recent SEC filings for {ticker}.",
        output_schema=FilingsAnalysis,
        model=SPECIALIST_MODEL,
    )


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "LMT"
    as_of = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else None
    analysis = run(ticker, as_of_date=as_of)
    print(analysis.model_dump_json(indent=2))
