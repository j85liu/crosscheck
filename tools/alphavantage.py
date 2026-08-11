"""
Thin wrapper around Alpha Vantage's free API.

Requires ALPHAVANTAGE_API_KEY (free tier signup: https://www.alphavantage.co/support/#api-key).
Free tier is rate-limited (check current limits on their site — they change over time),
so this module caches raw responses to data/cache/ to avoid re-hitting the API while iterating.
"""

import hashlib
import json
import os
import statistics
from pathlib import Path

import requests

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.alphavantage.co/query"

# Defense/aero tickers this project covers — used as the default peer set for
# get_peer_comparison() when the caller doesn't specify one.
DEFENSE_AERO_TICKERS = ["LMT", "RTX", "NOC", "GD", "BA", "RKLB", "SPCX"]


def _cache_path(params: dict) -> Path:
    key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
    return CACHE_DIR / f"{key}.json"


def _get(params: dict) -> dict:
    """Fetch from Alpha Vantage, using a local cache to avoid burning rate limits during dev."""
    cache_file = _cache_path(params)
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY not set. Add it to your .env file.")

    resp = requests.get(BASE_URL, params={**params, "apikey": api_key}, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if "Error Message" in data or "Note" in data:
        # "Note" usually means you hit the rate limit
        raise RuntimeError(f"Alpha Vantage error/rate-limit: {data.get('Note') or data.get('Error Message')}")

    cache_file.write_text(json.dumps(data))
    return data


def get_daily_prices(ticker: str, outputsize: str = "compact") -> dict:
    """
    Fetch daily OHLCV data. outputsize='compact' gives ~100 most recent trading days,
    which is enough for 30/90-day volatility calcs.
    """
    return _get(
        {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker.upper(),
            "outputsize": outputsize,
        }
    )


def compute_volatility_summary(ticker: str) -> dict:
    """
    Pull daily prices and compute basic realized volatility stats.
    Returns raw numbers — the risk_analyst agent turns these into a structured
    write-up via the LLM.
    """
    raw = get_daily_prices(ticker)
    series = raw.get("Time Series (Daily)")
    if not series:
        raise RuntimeError(f"No price series returned for {ticker}: {raw}")

    dates_sorted = sorted(series.keys(), reverse=True)  # most recent first
    closes = [float(series[d]["4. close"]) for d in dates_sorted]

    def realized_vol(window: int) -> float | None:
        if len(closes) < window + 1:
            return None
        prices = closes[: window + 1]
        returns = [
            (prices[i] - prices[i + 1]) / prices[i + 1] for i in range(len(prices) - 1)
        ]
        return statistics.stdev(returns) * (252 ** 0.5)  # annualized

    price_change_30d = None
    if len(closes) > 30:
        price_change_30d = (closes[0] - closes[30]) / closes[30] * 100

    return {
        "ticker": ticker.upper(),
        "latest_close": closes[0],
        "realized_volatility_30d": realized_vol(30),
        "realized_volatility_90d": realized_vol(90) if len(closes) > 90 else None,
        "price_change_30d_pct": price_change_30d,
    }


def get_peer_comparison(ticker: str, peer_tickers: list[str] | None = None) -> dict:
    """
    Compute volatility summaries for a set of peer tickers alongside the target, so the
    risk agent can judge whether an elevated volatility reading is company-specific or
    sector-wide. Defaults to the other defense/aero tickers this project covers if
    peer_tickers isn't given.

    Each peer goes through compute_volatility_summary() -> get_daily_prices() -> _get(),
    which already caches per-ticker to data/cache/ — no separate caching layer needed
    here. A peer that fails (e.g. rate limit) is recorded with an "error" key rather than
    aborting the whole comparison.
    """
    ticker = ticker.upper()
    if peer_tickers is None:
        peer_tickers = [t for t in DEFENSE_AERO_TICKERS if t != ticker]

    peers = {}
    for peer in peer_tickers:
        peer = peer.upper()
        if peer == ticker:
            continue
        try:
            peers[peer] = compute_volatility_summary(peer)
        except Exception as e:
            peers[peer] = {"error": str(e)}

    vols_30d = [
        p["realized_volatility_30d"]
        for p in peers.values()
        if isinstance(p, dict) and p.get("realized_volatility_30d") is not None
    ]

    return {
        "ticker": ticker,
        "peers": peers,
        "sector_avg_vol_30d": statistics.mean(vols_30d) if vols_30d else None,
    }


if __name__ == "__main__":
    print(compute_volatility_summary("LMT"))
    print(get_peer_comparison("LMT"))