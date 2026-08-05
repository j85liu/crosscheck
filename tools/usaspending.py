"""
Thin wrapper around USASpending.gov's public API. No API key required.
Docs: https://api.usaspending.gov/docs/endpoints
"""

from datetime import date, timedelta

import requests

BASE_URL = "https://api.usaspending.gov/api/v2"

# USASpending searches by recipient name, not ticker, so we map explicitly.
RECIPIENT_NAMES = {
    "LMT": "LOCKHEED MARTIN CORPORATION",
    "RTX": "RTX CORPORATION",
    "NOC": "NORTHROP GRUMMAN CORPORATION",
    "RKLB": "ROCKET LAB USA, INC.",
    "BA": "THE BOEING COMPANY",
    "GD": "GENERAL DYNAMICS CORPORATION",
    "SPCX": "SPACE EXPLORATION TECHNOLOGIES CORP.",
}


def get_recent_awards(ticker: str, limit: int = 10) -> dict:
    """Fetch recent federal contract awards for a company by ticker."""
    recipient_name = RECIPIENT_NAMES.get(ticker.upper())
    if not recipient_name:
        raise ValueError(f"No recipient name mapped for {ticker}. Add it to RECIPIENT_NAMES in tools/usaspending.py")

    payload = {
        "filters": {
            "recipient_search_text": [recipient_name],
            "award_type_codes": ["A", "B", "C", "D"],  # contract types
            "time_period": [{"start_date": "2024-01-01", "end_date": "2026-12-31"}],
        },
        "fields": [
            "Award ID",
            "Recipient Name",
            "Award Amount",
            "Contract Award Type",
            "Start Date",
            "Description",
            "Awarding Agency",
        ],
        "sort": "Award Amount",
        "order": "desc",
        "limit": limit,
    }

    resp = requests.post(f"{BASE_URL}/search/spending_by_award/", json=payload, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    return {
        "ticker": ticker.upper(),
        "recipient_name": recipient_name,
        "awards": data.get("results", []),
    }


def get_award_frequency(ticker: str, months: int = 12, limit: int = 100) -> dict:
    """
    Fetch awards sorted by recency (Start Date) rather than dollar value, as a cadence
    signal — is this a steady drip of smaller awards or one/few mega-contracts? — distinct
    from get_recent_awards' "biggest wins" view.

    USASpending's spending_by_award endpoint doesn't return a total match count, only
    the page of results requested, so award_count_trailing reflects up to `limit` awards
    in the window; "count_capped" is True if there may be more than that (check
    page_metadata.hasNext).
    """
    recipient_name = RECIPIENT_NAMES.get(ticker.upper())
    if not recipient_name:
        raise ValueError(f"No recipient name mapped for {ticker}. Add it to RECIPIENT_NAMES in tools/usaspending.py")

    start_date = (date.today() - timedelta(days=months * 30)).isoformat()
    end_date = date.today().isoformat()

    payload = {
        "filters": {
            "recipient_search_text": [recipient_name],
            "award_type_codes": ["A", "B", "C", "D"],
            "time_period": [{"start_date": start_date, "end_date": end_date}],
        },
        "fields": ["Award ID", "Recipient Name", "Award Amount", "Start Date"],
        "sort": "Start Date",
        "order": "desc",
        "limit": limit,
    }

    resp = requests.post(f"{BASE_URL}/search/spending_by_award/", json=payload, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])

    return {
        "ticker": ticker.upper(),
        "recipient_name": recipient_name,
        "months": months,
        "award_count_trailing": len(results),
        "total_value_trailing": sum(a.get("Award Amount") or 0 for a in results),
        "count_capped": data.get("page_metadata", {}).get("hasNext", False),
    }


if __name__ == "__main__":
    result = get_recent_awards("LMT", limit=5)
    print(f"{result['recipient_name']} — {len(result['awards'])} awards:")
    for a in result["awards"]:
        print(f"  {a.get('Award Amount')}  {a.get('Description', '')[:60]}")

    freq = get_award_frequency("LMT")
    print(
        f"\n{freq['recipient_name']} — {freq['award_count_trailing']} awards in trailing "
        f"{freq['months']}mo, ${freq['total_value_trailing']:,.0f} total"
        f"{' (capped, more exist)' if freq['count_capped'] else ''}"
    )