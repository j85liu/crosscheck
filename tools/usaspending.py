"""
Thin wrapper around USASpending.gov's public API. No API key required.
Docs: https://api.usaspending.gov/docs/endpoints
"""

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


if __name__ == "__main__":
    result = get_recent_awards("LMT", limit=5)
    print(f"{result['recipient_name']} — {len(result['awards'])} awards:")
    for a in result["awards"]:
        print(f"  {a.get('Award Amount')}  {a.get('Description', '')[:60]}")