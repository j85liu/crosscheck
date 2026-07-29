"""
Program analyst agent.

Pulls recent federal contract awards via USASpending.gov, then uses Claude
to identify which ones are significant and produce a structured ProgramAnalysis.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.client import SPECIALIST_MODEL, structured_call
from schemas.models import ProgramAnalysis
from tools.usaspending import get_recent_awards

PROMPT_TEMPLATE = """You are a defense program analyst. Below are recent federal contract
awards for {company_name} ({ticker}). "Start Date" is the award's period-of-performance
start date, not necessarily when the underlying news/announcement happened — USASpending's
award-summary API doesn't expose a true last-action/modification date, so treat this as an
approximation of recency, not an exact one.

{awards_list}

Identify the most significant events (large dollar value, or notable program/agency).
Mark significance="major" only for awards that stand out meaningfully from the rest.
Use each award's Start Date as the "date" field in your output.

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


def run(ticker: str) -> ProgramAnalysis:
    raw = get_recent_awards(ticker, limit=10)

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
    )

    result_dict = structured_call(prompt, model=SPECIALIST_MODEL, max_tokens=1024)
    return ProgramAnalysis(**result_dict)


if __name__ == "__main__":
    analysis = run("LMT")
    print(analysis.model_dump_json(indent=2))