"""
Sentiment analyst agent.

Uses the generic tool-calling harness (llm/agent_loop.py): the model starts with the
default company-name search, decides for itself whether the results look too noisy to use
as-is (off-topic, wrong company/language, spam) and re-runs with a tightened query if so,
then fetches full text on a handful of the most relevant articles before characterizing
sentiment.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.agent_loop import run_tool_agent
from llm.client import SPECIALIST_MODEL
from schemas.models import SentimentAnalysis
from tools.gdelt import get_article_text, get_recent_articles, refine_search

SYSTEM_PROMPT = """You are a sentiment analyst. You have three tools:

- get_recent_articles: recent headlines (with url, seendate, domain, language, etc.) for
  this ticker's default company-name query, plus GDELT's aggregate tone score. Always
  start here.
- refine_search: re-run the article search with a custom query string instead of the
  default — use this if get_recent_articles' results look too noisy to use as-is (mostly
  off-topic articles, wrong company entirely, one language dominating unhelpfully,
  forum/spam content). Don't call this if the default results already look reasonably
  on-topic — refining unnecessarily just wastes a call.
- get_article_text: full text (truncated) of one article by URL. Use this on a handful
  (3-5) of the most relevant, on-topic recent articles so your read isn't based on
  headlines alone. May fail (paywalls, dead links) — that's expected, just skip that
  article if so.

Workflow: call get_recent_articles first. Skim what came back — if most of it looks
irrelevant or off-topic for the company you're analyzing, call refine_search with a
tightened query (e.g. adding exclusions, quoting the company name, adding a
disambiguating term) before proceeding; otherwise move straight to fetching full text.

Refine at most once or twice. Judge a refined query by relevance, not volume — a small
set of genuinely on-topic articles is a good result and worth proceeding with, even if
it's only a handful. Don't keep re-refining just because a query returned few results; a
tightly-targeted query naturally returns fewer matches than a broad noisy one, and that's
the query working as intended, not a reason to widen it again. If a tool call errors
(e.g. a rate limit), that's a transient failure of the tool, not a signal that your query
was wrong — retry the same query rather than assuming it needs to be rephrased.

Once you have a usable set of relevant articles, fetch full text for a handful of the
most relevant ones, and characterize overall sentiment and key themes using the
headlines, tone score, and full text together.

In your summary, explicitly note whether your own qualitative read agrees or diverges
from GDELT's numeric aggregate tone score, and briefly say why if it diverges (e.g. the
tone score reflects a much larger/broader sample than the articles you reviewed, or
headline volume on one topic skews the numeric average).

When ready, call submit_final_answer with: ticker, overall_sentiment, key_themes,
sentiment_trend, avg_gdelt_tone (the exact figure from the tool results, rounded to 2
decimal places, or null if unavailable), and a 2-3 sentence summary including the
tone-agreement note.
"""

TOOLS = [
    {
        "name": "get_recent_articles",
        "description": (
            "Fetch recent news headlines (url, seendate, domain, language, etc.) for this "
            "ticker's default company-name search query, plus GDELT's aggregate tone score. "
            "Always start here."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Stock ticker, e.g. 'LMT'"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "refine_search",
        "description": (
            "Re-run the article search with a custom query instead of the default "
            "company-name search — use this if get_recent_articles' results look too noisy "
            "to use as-is (off-topic articles, wrong company/language, forum/spam content), "
            "e.g. tightening 'Lockheed Martin' to '\"Lockheed Martin\" -stock -forum' or "
            "narrowing further. Don't call this if the default results already look relevant."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker, e.g. 'LMT'"},
                "custom_query": {"type": "string", "description": "Replacement GDELT query string."},
            },
            "required": ["ticker", "custom_query"],
        },
    },
    {
        "name": "get_article_text",
        "description": (
            "Fetch the full visible text (truncated) of one article by URL. Use on a handful "
            "of the most relevant recent articles to ground your read in more than headlines. "
            "May fail (paywalls, dead links, timeouts) — that's expected, just skip that "
            "article if so."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Article URL from get_recent_articles/refine_search results."}
            },
            "required": ["url"],
        },
    },
]


def _tool_get_recent_articles(ticker: str) -> dict:
    return get_recent_articles(ticker, max_records=20)


def _tool_refine_search(ticker: str, custom_query: str) -> dict:
    return refine_search(ticker, custom_query, max_records=20)


def _tool_get_article_text(url: str) -> str:
    text = get_article_text(url)
    return text if text is not None else "Error: could not fetch article text (paywall, timeout, or dead link)."


TOOL_DISPATCH = {
    "get_recent_articles": _tool_get_recent_articles,
    "refine_search": _tool_refine_search,
    "get_article_text": _tool_get_article_text,
}


def run(ticker: str) -> SentimentAnalysis:
    # Higher than the harness default (5): a realistic run here is 1 search + up to a
    # couple of refinements + several individual get_article_text calls + submission,
    # each typically a separate turn (results usually need to be seen before deciding
    # the next call) — 5 wasn't enough headroom, observed directly in testing (a noisy
    # ticker exhausted 5 turns on search/refinement alone and never reached submission).
    return run_tool_agent(
        system_prompt=SYSTEM_PROMPT,
        tools=TOOLS,
        tool_dispatch=TOOL_DISPATCH,
        user_prompt=f"Analyze recent news sentiment for {ticker}.",
        output_schema=SentimentAnalysis,
        model=SPECIALIST_MODEL,
        max_turns=9,
    )


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "LMT"
    analysis = run(ticker)
    print(analysis.model_dump_json(indent=2))
