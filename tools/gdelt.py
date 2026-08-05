"""
Thin wrapper around GDELT's free DOC 2.0 API. No key required.
Docs: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/

GDELT's free tier rate-limits fairly aggressively (429s if you hit it too
often in a short window), so this caches responses locally and retries with
backoff on 429s.
"""

import hashlib
import json
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Company name to search for — GDELT searches article text, so use recognizable names.
SEARCH_TERMS = {
    "LMT": "Lockheed Martin",
    "RTX": "RTX Corporation OR Raytheon",
    "NOC": "Northrop Grumman",
    "RKLB": "Rocket Lab",
    "BA": "Boeing",
    "GD": "General Dynamics",
    "SPCX": "SpaceX",
}


def _cache_path(params: dict) -> Path:
    key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
    return CACHE_DIR / f"gdelt_{key}.json"


def _get_with_retry(params: dict, max_retries: int = 4) -> dict:
    cache_file = _cache_path(params)
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    delay = 5  # seconds, doubles each retry
    for attempt in range(max_retries):
        resp = requests.get(BASE_URL, params=params, timeout=20)
        if resp.status_code == 429:
            if attempt == max_retries - 1:
                resp.raise_for_status()
            print(f"GDELT rate-limited (429), waiting {delay}s before retry...")
            time.sleep(delay)
            delay *= 2
            continue
        resp.raise_for_status()
        try:
            data = resp.json()
        except json.JSONDecodeError:
            # GDELT occasionally returns HTTP 200 with an empty body (seen right after
            # a 429), instead of a clean error — treat it as transient and retry.
            if attempt == max_retries - 1:
                raise RuntimeError("GDELT returned an empty/invalid response after retries.")
            print(f"GDELT returned an empty response, waiting {delay}s before retry...")
            time.sleep(delay)
            delay *= 2
            continue
        cache_file.write_text(json.dumps(data))
        return data

    raise RuntimeError("GDELT request failed after retries.")


def _avg_tone(query_term: str) -> tuple[float | None, int]:
    """
    GDELT's artlist mode (used by get_recent_articles) doesn't include a per-article
    tone score — tone is only exposed in aggregate via mode=tonechart, a histogram of
    {bin: tone value, count: articles at that tone} over the full matching set for the
    query (not capped by maxrecords). Compute a count-weighted average from that.
    """
    params = {"query": query_term, "mode": "tonechart", "format": "json"}
    data = _get_with_retry(params)
    bins = data.get("tonechart", [])
    total_count = sum(b.get("count", 0) for b in bins)
    if total_count == 0:
        return None, 0
    weighted_sum = sum(b.get("bin", 0) * b.get("count", 0) for b in bins)
    return weighted_sum / total_count, total_count


def get_recent_articles(ticker: str, max_records: int = 20) -> dict:
    """
    Fetch recent news articles mentioning the company (headline, url, domain,
    sourcecountry, etc. — whatever GDELT's artlist response includes), plus an
    aggregate tone score computed separately (see _avg_tone).
    """
    query_term = SEARCH_TERMS.get(ticker.upper())
    if not query_term:
        raise ValueError(f"No search term mapped for {ticker}. Add it to SEARCH_TERMS in tools/gdelt.py")

    params = {
        "query": query_term,
        "mode": "artlist",
        "maxrecords": max_records,
        "format": "json",
        "sort": "datedesc",
    }

    data = _get_with_retry(params)
    avg_tone, tone_sample_size = _avg_tone(query_term)

    return {
        "ticker": ticker.upper(),
        "query_term": query_term,
        "articles": data.get("articles", []),
        "avg_tone": avg_tone,
        "tone_sample_size": tone_sample_size,
    }


def get_article_text(url: str, max_chars: int = 1500) -> str | None:
    """
    Best-effort fetch of an article's main visible text. Paywalls, JS-rendered pages,
    timeouts, and dead links are all expected here and should not crash the pipeline —
    any failure just returns None so the caller can skip that article.
    """
    try:
        resp = requests.get(
            url, timeout=10, headers={"User-Agent": "Mozilla/5.0 (CrossCheck research bot)"}
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        return text[:max_chars] if text else None
    except Exception:
        return None


if __name__ == "__main__":
    result = get_recent_articles("LMT", max_records=10)
    print(f"Found {len(result['articles'])} recent articles for {result['query_term']}:")
    for a in result["articles"][:5]:
        print(f"  {a.get('seendate')}  {a.get('title', '')[:70]}")
    print(f"\nAvg tone: {result['avg_tone']} (over {result['tone_sample_size']} articles)")

    if result["articles"]:
        first_url = result["articles"][0]["url"]
        text = get_article_text(first_url)
        print(f"\nFetched text from {first_url}:")
        print(text[:300] if text else "(fetch failed — None)")