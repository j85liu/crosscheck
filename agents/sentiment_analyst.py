"""
Sentiment analyst agent.

Uses the generic tool-calling harness (llm/agent_loop.py): the model starts with the
default company-name search, decides for itself whether the results look too noisy to use
as-is (off-topic, wrong company/language, spam) and re-runs with a tightened query if so,
then fetches full text on a few of the most relevant articles before characterizing
sentiment. GDELT's own rate limiting makes each search/refine call slow, so the prompt
caps refine_search at 1 call and get_article_text at 2-3 — a deliberate latency/
thoroughness tradeoff, not an oversight (see SYSTEM_PROMPT and run()'s max_turns).
"""

import functools
import sys
from datetime import date
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
  on-topic — refining unnecessarily just wastes a call. Call this AT MOST ONCE — GDELT is
  rate-limited and slow to query, so a second or third refinement is expensive in latency
  for little marginal benefit; if one refinement doesn't clean things up, proceed with
  what you have rather than trying again.
- get_article_text: full text (truncated) of one article by URL. Use this on AT MOST 2-3
  of the most relevant, on-topic recent articles so your read isn't based on headlines
  alone — going beyond 2-3 rarely changes the read and just adds latency. May fail
  (paywalls, dead links) — that's expected, just skip that article if so.

Workflow: call get_recent_articles first. Skim what came back — if most of it looks
irrelevant or off-topic for the company you're analyzing, call refine_search ONCE with a
tightened query (e.g. adding exclusions, quoting the company name, adding a
disambiguating term) before proceeding; otherwise move straight to fetching full text.

Refine at most once. This is a deliberate quality decision, not something to retry for
its own sake — judge the refined query by relevance, not volume; a small set of
genuinely on-topic articles is a good result and worth proceeding with, even if it's only
a handful. Don't re-refine a second time just because a query returned few results; a
tightly-targeted query naturally returns fewer matches than a broad noisy one, and that's
the query working as intended.

If a tool call errors (e.g. a rate limit), that's a transient failure of the tool, not a
signal that your query was wrong, and retrying the exact same call to recover from it
doesn't count as a second refinement — but retry AT MOST ONCE. GDELT's rate limiting can
be sustained rather than a one-off blip, and each retry is expensive in wall-clock time;
if the retry also fails, stop trying and call submit_final_answer with whatever you have
(headlines-only if full text never worked, or a neutral/no-data result with an honest
summary if the search itself never returned anything) rather than continuing to retry.

Once you have a usable set of relevant articles, fetch full text for at most 2-3 of the
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
            "narrowing further. Don't call this if the default results already look relevant. "
            "Call AT MOST ONCE — GDELT is rate-limited and slow, so repeated refinement is "
            "costly; proceed with what you get rather than refining again."
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
            "Fetch the full visible text (truncated) of one article by URL. Use on AT MOST "
            "2-3 of the most relevant recent articles to ground your read in more than "
            "headlines — going beyond that rarely changes the read. May fail (paywalls, "
            "dead links, timeouts) — that's expected, just skip that article if so."
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


def _tool_get_recent_articles(ticker: str, as_of_date: date | None) -> dict:
    return get_recent_articles(ticker, max_records=20, as_of_date=as_of_date)


def _tool_refine_search(ticker: str, custom_query: str, as_of_date: date | None) -> dict:
    return refine_search(ticker, custom_query, max_records=20, as_of_date=as_of_date)


def _tool_get_article_text(url: str) -> str:
    text = get_article_text(url)
    return text if text is not None else "Error: could not fetch article text (paywall, timeout, or dead link)."


def _build_tool_dispatch(as_of_date: date | None) -> dict:
    # as_of_date is baked in via functools.partial for the two search tools — the LLM
    # never sees or has to remember to pass it. get_article_text has no date concept of
    # its own (it fetches whatever URL it's given, already sourced from an as_of_date
    # -filtered search), so it needs no binding.
    return {
        "get_recent_articles": functools.partial(_tool_get_recent_articles, as_of_date=as_of_date),
        "refine_search": functools.partial(_tool_refine_search, as_of_date=as_of_date),
        "get_article_text": _tool_get_article_text,
    }


def run(ticker: str, as_of_date: date | None = None) -> SentimentAnalysis:
    system_prompt = SYSTEM_PROMPT
    if as_of_date is not None:
        system_prompt += (
            f"\n\nYou are analyzing data as of {as_of_date.isoformat()}. Do not reference or "
            "assume knowledge of anything after this date."
        )
    # The system prompt caps refine_search at 1 call and get_article_text at 2-3 (GDELT's
    # rate limiting makes each search/refine call expensive in wall-clock time, so this
    # trades a little thoroughness for a big latency win — going from 3 to 5 article
    # fetches rarely changes the read). max_turns=7 matches that tightened budget with a
    # little slack for an error-recovery retry (1 search + 1 refine + up to 3 fetches + 1
    # submit = 6, +1 buffer), down from 9 when the budget was looser — a model that
    # ignores the prompt's caps now fails fast instead of grinding through rate-limited
    # retries for several more turns.
    return run_tool_agent(
        system_prompt=system_prompt,
        tools=TOOLS,
        tool_dispatch=_build_tool_dispatch(as_of_date),
        user_prompt=f"Analyze recent news sentiment for {ticker}.",
        output_schema=SentimentAnalysis,
        model=SPECIALIST_MODEL,
        max_turns=7,
    )


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "LMT"
    as_of = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else None
    analysis = run(ticker, as_of_date=as_of)
    print(analysis.model_dump_json(indent=2))
