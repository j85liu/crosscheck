"""
Sentiment analyst agent.

Pulls recent news headlines plus GDELT's aggregate tone score via GDELT, fetches full
text for a handful of the most recent articles, then uses Claude to characterize overall
sentiment and key themes into a structured SentimentAnalysis — including whether its own
qualitative read agrees with GDELT's own numeric tone score.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.client import SPECIALIST_MODEL, structured_call
from schemas.models import SentimentAnalysis
from tools.gdelt import get_article_text, get_recent_articles

FULL_TEXT_SAMPLE_SIZE = 5

PROMPT_TEMPLATE = """You are a sentiment analyst. Below are recent news headlines mentioning
{query_term}, GDELT's own aggregate tone score for this query, and full-text excerpts from
a few of the most recent articles.

Headlines:
{headlines_list}

GDELT aggregate tone score: {avg_tone} (range roughly -10 to +10, negative=more negative
coverage, computed across {tone_sample_size} articles GDELT has indexed for this query —
a much larger, independent sample than the headlines above)

Full-text excerpts from recent articles:
{article_excerpts}

Based on all of this, characterize overall sentiment and the key themes being covered. In
your summary, explicitly note whether your own qualitative read agrees or diverges from
GDELT's numeric tone score, and briefly say why if it diverges (e.g. tone score reflects a
much larger/broader sample, or headline volume on one topic skews the numeric average).

Respond with ONLY this JSON structure, no other text:
{{
  "ticker": "{ticker}",
  "overall_sentiment": "positive" | "negative" | "neutral" | "mixed",
  "key_themes": ["...", "..."],
  "sentiment_trend": "improving" | "stable" | "worsening",
  "avg_gdelt_tone": {avg_tone_json},
  "summary": "2-3 sentence plain-language summary, including the tone-agreement note"
}}
"""


def _fetch_sample_full_text(articles: list[dict], sample_size: int) -> str:
    excerpts = []
    for a in articles:
        if len(excerpts) >= sample_size:
            break
        url = a.get("url")
        if not url:
            continue
        text = get_article_text(url)
        if text is None:
            continue
        excerpts.append(f"- \"{a.get('title', 'No title')}\" ({a.get('seendate', 'N/A')}):\n  {text}")
    return "\n".join(excerpts) if excerpts else "(no full text could be fetched for the sampled articles)"


def run(ticker: str) -> SentimentAnalysis:
    raw = get_recent_articles(ticker, max_records=20)

    headlines_list = "\n".join(
        f"- {a.get('seendate', 'N/A')}: {a.get('title', 'No title')}"
        for a in raw["articles"]
    )
    article_excerpts = _fetch_sample_full_text(raw["articles"], FULL_TEXT_SAMPLE_SIZE)

    avg_tone = raw["avg_tone"]
    prompt = PROMPT_TEMPLATE.format(
        query_term=raw["query_term"],
        ticker=raw["ticker"],
        headlines_list=headlines_list or "No recent articles found.",
        avg_tone=f"{avg_tone:.2f}" if avg_tone is not None else "N/A",
        avg_tone_json="null" if avg_tone is None else round(avg_tone, 2),
        tone_sample_size=raw["tone_sample_size"],
        article_excerpts=article_excerpts,
    )

    result_dict = structured_call(prompt, model=SPECIALIST_MODEL, max_tokens=1024)
    return SentimentAnalysis(**result_dict)


if __name__ == "__main__":
    analysis = run("LMT")
    print(analysis.model_dump_json(indent=2))