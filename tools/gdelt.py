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

BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Company name to search for — GDELT searches article text, so use recognizable names.
SEARCH_TERMS = {
    "LMT": "Lockheed Martin",
    "RTX": "RTX Corporation OR Raytheon",
    "NOC": "Northrop Grumman",
    "RKLB": "Rocket Lab",
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
        data = resp.json()
        cache_file.write_text(json.dumps(data))
        return data

    raise RuntimeError("GDELT request failed after retries.")


def get_recent_articles(ticker: str, max_records: int = 20) -> dict:
    """Fetch recent news articles mentioning the company, with GDELT's tone score."""
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

    return {
        "ticker": ticker.upper(),
        "query_term": query_term,
        "articles": data.get("articles", []),
    }


if __name__ == "__main__":
    result = get_recent_articles("LMT", max_records=10)
    print(f"Found {len(result['articles'])} recent articles for {result['query_term']}:")
    for a in result["articles"][:5]:
        print(f"  {a.get('seendate')}  {a.get('title', '')[:70]}")