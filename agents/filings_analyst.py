"""
Filings analyst agent.

Pulls recent SEC filings metadata for a ticker, then uses Claude to produce
a structured FilingsAnalysis (see schemas/models.py). Note this v1 works off
filing *metadata* (form type, date) rather than full document text — pulling
and parsing full 10-K text is a natural next step once this loop works.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.client import SPECIALIST_MODEL, structured_call
from schemas.models import FilingsAnalysis
from tools.sec_edgar import get_company_filings

PROMPT_TEMPLATE = """You are a financial filings analyst. Below is a list of recent SEC filings
for {company_name} ({ticker}).

Filings:
{filings_list}

Based on the filing types and dates alone (you do not have the full document text yet),
produce a brief structured analysis. Since you only have metadata, keep highlights
general (e.g. "Company filed a 10-K, suggesting annual results were reported") and note
that a deeper read of the actual filing content would refine this.

Respond with ONLY this JSON structure, no other text:
{{
  "ticker": "{ticker}",
  "company_name": "{company_name}",
  "overall_sentiment": "positive" | "negative" | "neutral" | "mixed",
  "highlights": [
    {{"topic": "...", "detail": "...", "sentiment": "positive"|"negative"|"neutral",
      "source_form": "...",
      "filed_date": "YYYY-MM-DD" (must exactly match one of the filing dates listed
      above — never a date range or a reporting period)}}
  ],
  "summary": "2-3 sentence plain-language summary"
}}
"""


def run(ticker: str) -> FilingsAnalysis:
    raw = get_company_filings(ticker, form_types=["10-K", "10-Q", "8-K"], limit=5)

    filings_list = "\n".join(
        f"- {f['form']} filed {f['filingDate']}" for f in raw["filings"]
    )
    prompt = PROMPT_TEMPLATE.format(
        company_name=raw["company_name"],
        ticker=raw["ticker"],
        filings_list=filings_list or "No recent filings found.",
    )

    result_dict = structured_call(prompt, model=SPECIALIST_MODEL, max_tokens=1024)
    return FilingsAnalysis(**result_dict)


if __name__ == "__main__":
    analysis = run("LMT")
    print(analysis.model_dump_json(indent=2))
