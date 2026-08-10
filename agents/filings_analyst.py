"""
Filings analyst agent.

Pulls recent SEC filings metadata plus structured XBRL financial figures (revenue,
net income, EPS, operating income) for a ticker, then uses Claude to produce a
structured FilingsAnalysis (see schemas/models.py). It still doesn't see the
qualitative sections of a filing (MD&A, risk factors, notes) — pulling and
parsing full filing text is a natural next step once this loop works.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.client import SPECIALIST_MODEL, structured_call
from schemas.models import FilingsAnalysis
from tools.sec_edgar import get_company_facts, get_company_filings

METRIC_LABELS = {
    "Revenues": "Revenue",
    "NetIncomeLoss": "Net Income",
    "EarningsPerShareDiluted": "Diluted EPS",
    "OperatingIncomeLoss": "Operating Income",
}

PROMPT_TEMPLATE = """You are a financial filings analyst. Below is a list of recent SEC filings
and recent reported financial figures for {company_name} ({ticker}).

Filings:
{filings_list}

Recent financial figures (most recent period first, from SEC XBRL structured data):
{financials}

Use the financial figures to call out real trends (e.g. revenue or margin growth/decline
period over period) as concrete highlights — you have real numbers, so don't hedge on
them. You still don't have the qualitative sections of these filings (MD&A, risk factors,
notes), so don't invent qualitative detail beyond what the numbers and filing types
support.

Each individual highlight's "sentiment" must be exactly one of "positive", "negative", or
"neutral" — never "mixed". If a single filing event or figure has both positive and
negative implications (e.g. revenue grew but margins compressed), split it into two
separate highlight entries, one per fact, each with its own clear-cut sentiment — do not
tag one highlight "mixed". Only the report-level "overall_sentiment" field below may be
"mixed".

Respond with ONLY this JSON structure, no other text:
{{
  "ticker": "{ticker}",
  "company_name": "{company_name}",
  "overall_sentiment": "positive" | "negative" | "neutral" | "mixed",
  "highlights": [
    {{"topic": "...", "detail": "...", "sentiment": "positive"|"negative"|"neutral" (never "mixed" — split into two highlights instead),
      "source_form": "...",
      "filed_date": "YYYY-MM-DD" (must exactly match one of the filing dates listed
      above — never a date range or a reporting period)}}
  ],
  "summary": "2-3 sentence plain-language summary"
}}
"""


def _format_usd(concept: str, value: float) -> str:
    if concept == "EarningsPerShareDiluted":
        return f"${value:.2f}"
    sign, abs_value = ("-", -value) if value < 0 else ("", value)
    if abs_value >= 1e9:
        return f"{sign}${abs_value / 1e9:.1f}B"
    if abs_value >= 1e6:
        return f"{sign}${abs_value / 1e6:.1f}M"
    return f"{sign}${abs_value:,.0f}"


def _format_financials(metrics: dict[str, list[dict]]) -> str:
    if not metrics:
        return "No structured financial data available."
    lines = []
    for concept, periods in metrics.items():
        label = METRIC_LABELS.get(concept, concept)
        trend = ", ".join(
            f"{_format_usd(concept, p['value'])} ({p['period_end']}, {p['form']})" for p in periods
        )
        lines.append(f"- {label}: {trend}")
    return "\n".join(lines)


VALID_HIGHLIGHT_SENTIMENTS = {"positive", "negative", "neutral"}


def _normalize_highlight_sentiments(highlights: list[dict]) -> list[dict]:
    """
    Safety net for the prompt-level instruction above: the model is told individual
    highlights must never be "mixed", but LLM output isn't guaranteed to comply, and
    FilingHighlight.sentiment doesn't accept it — remap any invalid value to "neutral"
    rather than letting one bad field crash the whole agent call.
    """
    for h in highlights:
        sentiment = h.get("sentiment")
        if sentiment not in VALID_HIGHLIGHT_SENTIMENTS:
            print(
                f"WARNING: filings_analyst got invalid highlight sentiment {sentiment!r} "
                f"for topic {h.get('topic')!r} — remapping to 'neutral'."
            )
            h["sentiment"] = "neutral"
    return highlights


def run(ticker: str) -> FilingsAnalysis:
    raw = get_company_filings(ticker, form_types=["10-K", "10-Q", "8-K"], limit=5)
    facts = get_company_facts(ticker)

    filings_list = "\n".join(
        f"- {f['form']} filed {f['filingDate']}" for f in raw["filings"]
    )
    prompt = PROMPT_TEMPLATE.format(
        company_name=raw["company_name"],
        ticker=raw["ticker"],
        filings_list=filings_list or "No recent filings found.",
        financials=_format_financials(facts["metrics"]),
    )

    result_dict = structured_call(prompt, model=SPECIALIST_MODEL, max_tokens=1024)
    result_dict["highlights"] = _normalize_highlight_sentiments(result_dict.get("highlights", []))
    return FilingsAnalysis(**result_dict)


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "LMT"
    analysis = run(ticker)
    print(analysis.model_dump_json(indent=2))
