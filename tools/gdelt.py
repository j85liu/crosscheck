"""
Thin wrapper around GDELT's free DOC 2.0 API. No key required.
Docs: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
"""

import requests

BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Company name to search for — GDELT searches article text, so use recognizable names.
SEARCH_TERMS = {
    "LMT": "Lockheed Martin",
    "RTX": "RTX Corporation OR Raytheon",
    "NOC": "Northrop Grumman",
    "RKLB": "Rocket Lab",
}


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

    resp = requests.get(BASE_URL, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()

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