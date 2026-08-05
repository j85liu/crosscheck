"""
Program analyst agent.

Pulls recent federal contract awards via USASpending.gov — both the biggest awards by
dollar value and the recent award *cadence* (frequency/volume, independent of size) —
then uses Claude to identify significant events and characterize contract flow into a
structured ProgramAnalysis.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.client import SPECIALIST_MODEL, structured_call
from schemas.models import ProgramAnalysis
from tools.usaspending import get_award_frequency, get_recent_awards

PROMPT_TEMPLATE = """You are a defense program analyst. Below is federal contract award
data for {company_name} ({ticker}) from USASpending.gov. "Start Date" is the award's
period-of-performance start date, not necessarily when the underlying news/announcement
happened — USASpending's award-summary API doesn't expose a true last-action/modification
date, so treat this as an approximation of recency, not an exact one.

Biggest awards (top 10 by dollar value, all-time in the queried window):
{awards_list}

Award cadence (trailing {months} months, sorted by recency — how often new awards land,
independent of size):
{frequency_summary}

Identify the most significant events (large dollar value, or notable program/agency) from
the biggest-awards list. Mark significance="major" only for awards that stand out
meaningfully from the rest. Use each award's Start Date as the "date" field in your output.
Also use the cadence data to characterize whether contract flow looks like a steady drip of
smaller awards, one-off mega-contracts, or something sparse/thin — don't just repeat the
biggest awards list.

If both the biggest-awards list and the cadence data are empty, do not treat this as a
failure — say so plainly in the summary (e.g. "no significant federal contract activity
found in the queried window — this may reflect limited government revenue exposure, or a
data/mapping gap") and set overall_sentiment to "neutral" with an empty recent_events list.

Respond with ONLY this JSON structure, no other text:
{{
  "ticker": "{ticker}",
  "company_name": "{company_name}",
  "recent_events": [
    {{"description": "...", "value_usd": <number or null>, "date": "YYYY-MM-DD" or null,
      "significance": "major"|"minor"}}
  ],
  "overall_sentiment": "positive" | "negative" | "neutral" | "mixed",
  "summary": "2-3 sentence plain-language summary"
}}
"""


def _format_frequency(freq: dict) -> str:
    if freq["award_count_trailing"] == 0:
        return f"No awards found in the trailing {freq['months']} months."
    capped_note = " (more may exist beyond this count — result was capped)" if freq["count_capped"] else ""
    return (
        f"{freq['award_count_trailing']} awards{capped_note}, "
        f"${freq['total_value_trailing']:,.0f} combined value in the trailing {freq['months']} months."
    )


def run(ticker: str) -> ProgramAnalysis:
    raw = get_recent_awards(ticker, limit=10)
    freq = get_award_frequency(ticker)

    awards_list = "\n".join(
        f"- ${a.get('Award Amount', 'N/A')}: {(a.get('Description') or 'No description')[:120]} "
        f"(Agency: {a.get('Awarding Agency', 'N/A')}, Type: {a.get('Contract Award Type', 'N/A')}, "
        f"Start Date: {a.get('Start Date', 'N/A')})"
        for a in raw["awards"]
    )

    prompt = PROMPT_TEMPLATE.format(
        company_name=raw["recipient_name"],
        ticker=raw["ticker"],
        awards_list=awards_list or "No recent awards found.",
        months=freq["months"],
        frequency_summary=_format_frequency(freq),
    )

    result_dict = structured_call(prompt, model=SPECIALIST_MODEL, max_tokens=1024)
    return ProgramAnalysis(**result_dict)


if __name__ == "__main__":
    analysis = run("LMT")
    print(analysis.model_dump_json(indent=2))