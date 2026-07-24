"""
Sentiment analyst agent.

Pulls recent news headlines via GDELT, then uses Claude to characterize
overall sentiment and key themes into a structured SentimentAnalysis.

Note: this v1 works off headlines/titles only (GDELT's artlist mode), not
full article text — a reasonable next step is fetching full text for the
top few articles once this loop works.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.client import SPECIALIST_MODEL, structured_call
from schemas.models import SentimentAnalysis
from tools.gdelt import get_recent_articles

PROMPT_TEMPLATE = """You are a sentiment analyst. Below are recent news headlines mentioning
{query_term}:

{headlines_list}

Based on these headlines, characterize overall sentiment and the key themes being covered.

Respond with ONLY this JSON structure, no other text:
{{
  "ticker": "{ticker}",
  "overall_sentiment": "positive" | "negative" | "neutral" | "mixed",
  "key_themes": ["...", "..."],
  "sentiment_trend": "improving" | "stable" | "worsening",
  "summary": "2-3 sentence plain-language summary"
}}
"""


def run(ticker: str) -> SentimentAnalysis:
    raw = get_recent_articles(ticker, max_records=20)

    headlines_list = "\n".join(
        f"- {a.get('seendate', 'N/A')}: {a.get('title', 'No title')}"
        for a in raw["articles"]
    )

    prompt = PROMPT_TEMPLATE.format(
        query_term=raw["query_term"],
        ticker=raw["ticker"],
        headlines_list=headlines_list or "No recent articles found.",
    )

    result_dict = structured_call(prompt, model=SPECIALIST_MODEL, max_tokens=768)
    return SentimentAnalysis(**result_dict)


if __name__ == "__main__":
    analysis = run("LMT")
    print(analysis.model_dump_json(indent=2))