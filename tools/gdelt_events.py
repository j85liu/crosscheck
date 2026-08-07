"""
Thin wrapper around GDELT's Events database — structured, CAMEO-coded geopolitical
events ("who did what to whom, where"), distinct from tools/gdelt.py's news-article
DOC 2.0 API (which has no concept of event codes or the Goldstein conflict/cooperation
scale at all).

APPROACH: GDELT's full Events dataset is only queryable via Google BigQuery, which
needs a paid/authenticated GCP project — not appropriate for this project. Instead this
pulls directly from GDELT's free, no-auth raw CSV export files published every 15
minutes at http://data.gdeltproject.org/gdeltv2/ (the same underlying data BigQuery
serves, just unaggregated). Each file is the full global event stream for that
15-minute window (~800-1000 events, 61-column TSV, no header row — the fixed GDELT 2.0
Event Codebook: https://www.gdeltproject.org/data/documentation/GDELT-Event_Codebook-V2.0.pdf).
Verified live: this file host has no rate limiting (unlike api.gdeltproject.org's DOC
API), so no 429/backoff handling is needed here — just a plain retry on transient
network errors.

This is real, unfiltered GDELT event data, not a proxy through the article-search API.
The tradeoff: any single country shows up in only a handful of events per 15-minute
window, so getting a useful sample means scanning backward through several hours of
files rather than one snapshot (see SCAN_HOURS below) — still well under a second per
file, so this stays cheap even scanning hours of history.

GOTCHA (verified against live data, not assumed): GDELT uses TWO DIFFERENT country code
standards in the same row, and mixing them up silently corrupts matching:
  - Actor1CountryCode / Actor2CountryCode (fields 8, 18) use ISO 3166-1 ALPHA-3 (e.g.
    "AUS", "ISR", "GBR").
  - Actor1Geo/Actor2Geo/ActionGeo CountryCode (fields 39, 47, 55) use FIPS 10-4, a
    different, older 2-letter standard where e.g. Australia is "AS" (not ISO's "AU" —
    "AU" is Austria in FIPS), Israel is "IS" (not "IL"), Japan is "JA" (not "JP"),
    South Korea is "KS" (not "KR"), the UK is "UK" (not "GB" — "GB" is Gabon in FIPS),
    and Ukraine is "UP" (not "UA"). Getting this wrong doesn't error, it just silently
    returns events about the wrong country. COUNTRY_CONTEXT below stores both code sets
    per country explicitly so this can't be quietly reintroduced.

KNOWN LIMITATION: GDELT's event coding is fully automated (no human review) and its
geocoding sometimes misattributes a story to a country mentioned only in passing (e.g. a
Catholic newsletter story picked up as a Poland-tagged event). This wrapper does not
attempt to filter that out beyond basic mention-count sanity — treat results as raw
signal to be judged, not ground truth. This is a well-documented, inherent property of
GDELT's free data, not a bug in this integration.
"""

import csv
import hashlib
import io
import json
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

BASE_URL = "http://data.gdeltproject.org/gdeltv2"

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 0-indexed column positions in the 61-column GDELT 2.0 event export, per the official codebook.
COL = {
    "global_event_id": 0,
    "day": 1,
    "actor1_country": 7,     # ISO alpha-3
    "actor2_country": 17,    # ISO alpha-3
    "event_code": 26,
    "event_root_code": 28,
    "goldstein_scale": 30,
    "num_mentions": 31,
    "avg_tone": 34,
    "action_geo_fullname": 52,
    "action_geo_country": 53,  # FIPS 10-4
    "source_url": 60,
}

# CAMEO event root codes (01-20), the standard published taxonomy.
EVENT_ROOT_LABELS = {
    "01": "Make public statement", "02": "Appeal", "03": "Express intent to cooperate",
    "04": "Consult", "05": "Engage in diplomatic cooperation", "06": "Engage in material cooperation",
    "07": "Provide aid", "08": "Yield", "09": "Investigate", "10": "Demand", "11": "Disapprove",
    "12": "Reject", "13": "Threaten", "14": "Protest", "15": "Exhibit force posture",
    "16": "Reduce relations", "17": "Coerce", "18": "Assault", "19": "Fight",
    "20": "Use unconventional mass violence",
}

# Countries tied to each ticker's major international program exposure (export customers,
# JV/co-production partners, key facilities) — a judgment call based on known public
# program history, not an exhaustive list. Deliberately excludes the US for the primes with
# a genuine international footprint (domestic US events would dominate/dilute the signal);
# kept for RKLB, whose international footprint is thin enough that dropping it would leave
# almost no signal. Each country lists both code standards GDELT actually uses (see the
# module docstring) — "geo" (FIPS 10-4) matches Actor/ActionGeo_CountryCode, "actor" (ISO
# alpha-3) matches Actor1/2CountryCode.
COUNTRY_CONTEXT = {
    "LMT": {
        "countries": [
            {"name": "Poland", "geo": "PL", "actor": "POL"},
            {"name": "Israel", "geo": "IS", "actor": "ISR"},
            {"name": "Japan", "geo": "JA", "actor": "JPN"},
            {"name": "South Korea", "geo": "KS", "actor": "KOR"},
            {"name": "Finland", "geo": "FI", "actor": "FIN"},
            {"name": "Australia", "geo": "AS", "actor": "AUS"},
        ],
        "rationale": "F-35, THAAD, and PAC-3 partner/customer nations",
    },
    "RTX": {
        "countries": [
            {"name": "Poland", "geo": "PL", "actor": "POL"},
            {"name": "Germany", "geo": "GM", "actor": "DEU"},
            {"name": "Saudi Arabia", "geo": "SA", "actor": "SAU"},
            {"name": "Qatar", "geo": "QA", "actor": "QAT"},
            {"name": "United Arab Emirates", "geo": "AE", "actor": "ARE"},
        ],
        "rationale": "Patriot and missile-defense export customers",
    },
    "NOC": {
        "countries": [
            {"name": "Australia", "geo": "AS", "actor": "AUS"},
            {"name": "Japan", "geo": "JA", "actor": "JPN"},
            {"name": "Norway", "geo": "NO", "actor": "NOR"},
        ],
        "rationale": "Triton, Global Hawk, and AUKUS-adjacent program partners",
    },
    "RKLB": {
        "countries": [
            {"name": "New Zealand", "geo": "NZ", "actor": "NZL"},
            {"name": "United States", "geo": "US", "actor": "USA"},
        ],
        "rationale": "Launch Complex 1 (New Zealand) plus US government launch customers",
    },
    "BA": {
        "countries": [
            {"name": "Japan", "geo": "JA", "actor": "JPN"},
            {"name": "India", "geo": "IN", "actor": "IND"},
            {"name": "Qatar", "geo": "QA", "actor": "QAT"},
            {"name": "United Kingdom", "geo": "UK", "actor": "GBR"},
        ],
        "rationale": "major commercial and defense export markets",
    },
    "GD": {
        "countries": [
            {"name": "Taiwan", "geo": "TW", "actor": "TWN"},
            {"name": "Australia", "geo": "AS", "actor": "AUS"},
            {"name": "Canada", "geo": "CA", "actor": "CAN"},
        ],
        "rationale": "Abrams (Taiwan), combat vehicles (Australia), LAV (Canada)",
    },
    "SPCX": {
        "countries": [
            {"name": "Ukraine", "geo": "UP", "actor": "UKR"},
            {"name": "Taiwan", "geo": "TW", "actor": "TWN"},
        ],
        "rationale": "Starlink/launch-related geopolitical flashpoints",
    },
}

SCAN_HOURS = 8              # how far back to scan 15-minute files looking for matches
CANDIDATE_POOL_TARGET = 60  # stop scanning early once we have this many candidates to sort from


def _file_timestamps(hours: int) -> list[datetime]:
    now = datetime.now(timezone.utc)
    floor_minute = (now.minute // 15) * 15
    # start one slot before the current 15-min mark — GDELT publishes on a short lag, so
    # the very latest slot may 404 if requested too early
    latest = now.replace(minute=floor_minute, second=0, microsecond=0) - timedelta(minutes=15)
    return [latest - timedelta(minutes=15 * i) for i in range(hours * 4)]


def _cache_path(fname: str, ticker: str) -> Path:
    key = hashlib.sha256(f"{fname}:{ticker}".encode()).hexdigest()[:16]
    return CACHE_DIR / f"gdeltevents_{key}.json"


def _fetch_and_filter(ts: datetime, ticker: str, geo_codes: set[str], actor_codes: set[str], max_retries: int = 2) -> list[dict] | None:
    """
    Download one 15-minute export file and return only rows touching this ticker's
    country context. Returns None (distinct from []) if the file couldn't be fetched at
    all (doesn't exist yet, or a persistent network error) vs. [] meaning it fetched
    fine but had zero matching events.
    """
    fname = ts.strftime("%Y%m%d%H%M%S")
    cache_file = _cache_path(fname, ticker)
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    url = f"{BASE_URL}/{fname}.export.CSV.zip"
    delay = 2
    resp = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=20)
            break
        except requests.RequestException:
            if attempt == max_retries - 1:
                return None
            time.sleep(delay)
            delay *= 2

    if resp is None or resp.status_code != 200:
        return None  # most commonly a 404 — file not published yet for this slot

    try:
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        raw = zf.read(zf.namelist()[0]).decode("utf-8", errors="replace")
    except (zipfile.BadZipFile, IndexError):
        return None

    matches = []
    for row in csv.reader(io.StringIO(raw), delimiter="\t"):
        if len(row) < 61:
            continue
        if not (
            row[COL["actor1_country"]] in actor_codes
            or row[COL["actor2_country"]] in actor_codes
            or row[COL["action_geo_country"]] in geo_codes
        ):
            continue
        try:
            goldstein = float(row[COL["goldstein_scale"]]) if row[COL["goldstein_scale"]] else None
            avg_tone = float(row[COL["avg_tone"]]) if row[COL["avg_tone"]] else None
            num_mentions = int(row[COL["num_mentions"]]) if row[COL["num_mentions"]] else 0
        except ValueError:
            continue
        matches.append(
            {
                "event_id": row[COL["global_event_id"]],
                "date": row[COL["day"]],
                "event_code": row[COL["event_code"]],
                "event_root_code": row[COL["event_root_code"]],
                "goldstein_scale": goldstein,
                "avg_tone": avg_tone,
                "num_mentions": num_mentions,
                "geo_country_code": row[COL["action_geo_country"]],
                "location": row[COL["action_geo_fullname"]],
                "source_url": row[COL["source_url"]],
            }
        )

    cache_file.write_text(json.dumps(matches))
    return matches


def get_geopolitical_events(ticker: str, max_records: int = 20) -> dict:
    """
    Fetch recent geopolitical events (CAMEO-coded, with Goldstein conflict/cooperation
    scale) tied to countries relevant to this ticker's international program exposure.

    Scans up to SCAN_HOURS of GDELT's 15-minute event export files for events touching
    the ticker's COUNTRY_CONTEXT, stopping early once CANDIDATE_POOL_TARGET candidates
    are found. Returns up to max_records events, deduped by source URL and sorted by
    |Goldstein scale| (most conflictual/cooperative first) as a proxy for "most notable."
    """
    ctx = COUNTRY_CONTEXT.get(ticker.upper())
    if not ctx:
        raise ValueError(f"No country context mapped for {ticker}. Add it to COUNTRY_CONTEXT in tools/gdelt_events.py")
    geo_codes = {c["geo"] for c in ctx["countries"]}
    actor_codes = {c["actor"] for c in ctx["countries"]}
    geo_to_name = {c["geo"]: c["name"] for c in ctx["countries"]}

    seen_urls = set()
    candidates = []
    files_scanned = 0
    for ts in _file_timestamps(SCAN_HOURS):
        rows = _fetch_and_filter(ts, ticker.upper(), geo_codes, actor_codes)
        if rows is None:
            continue
        files_scanned += 1
        for r in rows:
            if r["source_url"] in seen_urls:
                continue
            seen_urls.add(r["source_url"])
            r["country"] = geo_to_name.get(r["geo_country_code"], r["geo_country_code"])
            candidates.append(r)
        if len(candidates) >= CANDIDATE_POOL_TARGET:
            break

    candidates.sort(
        key=lambda r: abs(r["goldstein_scale"]) if r["goldstein_scale"] is not None else -1, reverse=True
    )

    return {
        "ticker": ticker.upper(),
        "countries": [c["name"] for c in ctx["countries"]],
        "rationale": ctx["rationale"],
        "files_scanned": files_scanned,
        "events": candidates[:max_records],
    }


if __name__ == "__main__":
    result = get_geopolitical_events("LMT")
    print(
        f"{result['ticker']} — scanned {result['files_scanned']} files for "
        f"{', '.join(result['countries'])} ({result['rationale']})"
    )
    print(f"Found {len(result['events'])} events:")
    for e in result["events"][:10]:
        label = EVENT_ROOT_LABELS.get(e["event_root_code"], e["event_root_code"])
        goldstein = e["goldstein_scale"] if e["goldstein_scale"] is not None else 0.0
        print(
            f"  {e['date']}  {label:35s} goldstein={goldstein:+.1f}  "
            f"{e['country']:15s}  {e['location'][:35]:35s}  {e['source_url'][:50]}"
        )
