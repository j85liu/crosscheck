"""
Thin wrapper around SEC EDGAR's public JSON API.

No API key needed, but SEC requires a descriptive User-Agent header
identifying your app (they'll rate-limit or block generic ones).
Docs: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
"""

from datetime import date

import requests
from bs4 import BeautifulSoup

USER_AGENT = "CrossCheck research project james.liu@columbia.edu"  # update with your info
BASE_HEADERS = {"User-Agent": USER_AGENT}

# CIK lookup requires the zero-padded 10-digit SEC company identifier.
# You look these up once at https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany
# Defense/aero tickers we're targeting:
KNOWN_CIKS = {
    "LMT": "0000936468",    # Lockheed Martin
    "RTX": "0000101829",    # RTX Corporation
    "NOC": "0001133421",    # Northrop Grumman
    "RKLB": "0001819994",   # Rocket Lab
    "BA": "0000012927",     # Boeing
    "GD": "0000040533",     # General Dynamics
    "SPCX": "0001181412",   # Space Exploration Technologies (SpaceX), IPO'd June 2026
}


def get_company_filings(
    ticker: str, form_types: list[str] | None = None, limit: int = 5, as_of_date: date | None = None
) -> dict:
    """
    Fetch recent filings metadata for a company.

    Returns a dict with keys like 'form', 'filingDate', 'accessionNumber', 'primaryDocument'
    for each recent filing, filtered to form_types if given (e.g. ["10-K", "10-Q", "8-K"]).

    as_of_date, if given, excludes any filing dated after it — SEC's submissions feed is
    already sorted most-recent-first, so this just skips entries newer than the cutoff
    until it reaches ones at or before it, then takes `limit` from there.
    """
    cik = KNOWN_CIKS.get(ticker.upper())
    if not cik:
        raise ValueError(f"No CIK mapped for {ticker}. Add it to KNOWN_CIKS in tools/sec_edgar.py")

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=BASE_HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    recent = data["filings"]["recent"]
    filings = []
    for i in range(len(recent["form"])):
        if form_types and recent["form"][i] not in form_types:
            continue
        filing_date = recent["filingDate"][i]
        if as_of_date is not None and date.fromisoformat(filing_date) > as_of_date:
            continue
        filings.append(
            {
                "form": recent["form"][i],
                "filingDate": filing_date,
                "accessionNumber": recent["accessionNumber"][i],
                "primaryDocument": recent["primaryDocument"][i],
            }
        )
        if len(filings) >= limit:
            break

    return {
        "ticker": ticker.upper(),
        "company_name": data.get("name", ticker),
        "cik": cik,
        "filings": filings,
        "as_of_date": as_of_date.isoformat() if as_of_date else None,
    }


# XBRL tags companies use for the same real-world figure vary (e.g. some tag revenue
# as "Revenues", others as "RevenueFromContractWithCustomerExcludingAssessedTax" post
# ASC 606 adoption) — try each in order and use whichever the company actually reports.
CONCEPT_ALTERNATES = {
    "Revenues": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "NetIncomeLoss": ["NetIncomeLoss"],
    "EarningsPerShareDiluted": ["EarningsPerShareDiluted"],
    "OperatingIncomeLoss": ["OperatingIncomeLoss"],
}

DEFAULT_CONCEPTS = list(CONCEPT_ALTERNATES)

PREFERRED_UNITS = {
    "EarningsPerShareDiluted": "USD/shares",
}


def _is_single_period(entry: dict) -> bool:
    """
    True for a standalone quarter (~90 days) or fiscal year (~365 days).

    10-Q filings report duration facts for both the single quarter AND the
    cumulative year-to-date (e.g. Q2's XBRL data includes both the Q2-only figure
    and the H1 YTD figure, both dated 'end'=quarter end) — without this filter
    we'd double-count the same period with two very different values.
    """
    start, end = entry.get("start"), entry.get("end")
    if not start or not end:
        return False
    days = (date.fromisoformat(end) - date.fromisoformat(start)).days
    return 80 <= days <= 100 or 350 <= days <= 380


def _recent_periods(entries: list[dict], max_periods: int = 6, as_of_date: date | None = None) -> list[dict]:
    """
    Narrow raw XBRL fact entries down to a short, clean recent trend: annual/quarterly
    filings only (skip amended 10-K/A, 10-Q/A, and non-filing forms), single-quarter or
    single-year periods only (not YTD cumulative), one value per distinct reporting
    period (the most recently filed value, in case of restatement), most recent first.

    as_of_date, if given, filters on each fact's 'filed' date — NOT its 'end' (fiscal
    period end) date. A quarter that ENDED before as_of_date may not have been FILED yet
    as of that date (e.g. Q2 ends in late June but isn't filed until late July) — filtering
    on 'end' instead of 'filed' would leak look-ahead information nobody actually had on
    that date, which defeats the point of point-in-time analysis.
    """
    filtered = [e for e in entries if e.get("form") in ("10-K", "10-Q") and _is_single_period(e)]
    if as_of_date is not None:
        filtered = [e for e in filtered if e.get("filed") and date.fromisoformat(e["filed"]) <= as_of_date]

    latest_by_period: dict[tuple, dict] = {}
    for e in filtered:
        key = (e.get("start"), e.get("end"))
        existing = latest_by_period.get(key)
        if existing is None or e.get("filed", "") > existing.get("filed", ""):
            latest_by_period[key] = e

    periods = sorted(latest_by_period.values(), key=lambda e: e["end"], reverse=True)
    return periods[:max_periods]


def get_company_facts(ticker: str, concepts: list[str] | None = None, as_of_date: date | None = None) -> dict:
    """
    Fetch recent structured financial figures ("company facts" XBRL data) for a company —
    actual reported numbers (revenue, net income, diluted EPS, operating income by default),
    not just filing metadata. Standardized by SEC, so no HTML/text parsing needed.

    For each concept, returns up to the 6 most recent reported periods so callers get a
    short trend rather than the entire multi-year history. Concepts a company doesn't
    report under any known tag are silently omitted (see "missing_concepts" in the result).

    as_of_date, if given, excludes facts not yet filed as of that date — see
    _recent_periods for why this filters on 'filed' rather than the fiscal period's 'end'.
    """
    cik = KNOWN_CIKS.get(ticker.upper())
    if not cik:
        raise ValueError(f"No CIK mapped for {ticker}. Add it to KNOWN_CIKS in tools/sec_edgar.py")

    concepts = concepts or DEFAULT_CONCEPTS

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    resp = requests.get(url, headers=BASE_HEADERS, timeout=15)
    if resp.status_code == 404:
        return {
            "ticker": ticker.upper(),
            "cik": cik,
            "metrics": {},
            "missing_concepts": concepts,
            "as_of_date": as_of_date.isoformat() if as_of_date else None,
        }
    resp.raise_for_status()
    data = resp.json()

    us_gaap = data.get("facts", {}).get("us-gaap", {})
    metrics: dict[str, list[dict]] = {}
    missing: list[str] = []

    for concept in concepts:
        concept_data = None
        for name in CONCEPT_ALTERNATES.get(concept, [concept]):
            if name in us_gaap:
                concept_data = us_gaap[name]
                break
        if concept_data is None:
            missing.append(concept)
            continue

        units = concept_data.get("units", {})
        unit_key = PREFERRED_UNITS.get(concept)
        if unit_key not in units:
            unit_key = "USD" if "USD" in units else next(iter(units), None)
        if unit_key is None:
            missing.append(concept)
            continue

        periods = _recent_periods(units[unit_key], as_of_date=as_of_date)
        if not periods:
            missing.append(concept)
            continue

        metrics[concept] = [
            {"period_end": p["end"], "value": p["val"], "form": p["form"], "filed": p["filed"]} for p in periods
        ]

    return {
        "ticker": ticker.upper(),
        "cik": cik,
        "metrics": metrics,
        "missing_concepts": missing,
        "as_of_date": as_of_date.isoformat() if as_of_date else None,
    }


def get_filing_document_url(ticker: str, accession_number: str, primary_document: str) -> str:
    """Build the direct URL to a filing's primary document (e.g. the actual 10-K HTML)."""
    cik = KNOWN_CIKS.get(ticker.upper())
    accession_nodash = accession_number.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodash}/{primary_document}"


def get_filing_full_text(
    ticker: str, accession_number: str, primary_document: str, max_chars: int = 3000, as_of_date: date | None = None
) -> str:
    """
    Fetch and extract the visible text of one filing's primary document, truncated to
    max_chars. This is meant to be called mid agent-loop (an agent deciding a specific
    filing warrants a closer look), so failures return a clear error string instead of
    raising — the calling agent can see the failure and adapt (e.g. proceed without full
    text) rather than the whole run crashing over one slow/unavailable document.

    Accepts as_of_date for interface consistency with the other tools in this module, but
    doesn't filter on it here: this function fetches one specific already-identified
    filing (by accession_number/primary_document), and point-in-time correctness for
    which filings are reachable at all is already enforced upstream, by get_company_filings
    only returning accession numbers for filings at or before as_of_date in the first place.
    """
    try:
        url = get_filing_document_url(ticker, accession_number, primary_document)
        resp = requests.get(url, headers=BASE_HEADERS, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        return text[:max_chars] if text else "(document fetched but contained no visible text)"
    except Exception as e:
        return f"Error fetching filing full text: {e}"


if __name__ == "__main__":
    # quick manual test
    result = get_company_filings("LMT", form_types=["10-K", "10-Q", "8-K"], limit=5)
    print(f"{result['company_name']} ({result['ticker']}) — {len(result['filings'])} recent filings:")
    for f in result["filings"]:
        print(f"  {f['form']:6s} filed {f['filingDate']}  {f['primaryDocument']}")

    facts = get_company_facts("LMT")
    print(f"\nCompany facts for {facts['ticker']} (CIK {facts['cik']}):")
    for concept, periods in facts["metrics"].items():
        print(f"  {concept}:")
        for p in periods:
            print(f"    {p['period_end']}  {p['value']:>15,}  ({p['form']})")
    if facts["missing_concepts"]:
        print(f"  Missing/unavailable concepts: {', '.join(facts['missing_concepts'])}")
