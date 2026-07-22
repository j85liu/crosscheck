"""
Thin wrapper around SEC EDGAR's public JSON API.

No API key needed, but SEC requires a descriptive User-Agent header
identifying your app (they'll rate-limit or block generic ones).
Docs: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
"""

import requests

USER_AGENT = "CrossCheck research project james.liu@columbia.edu"  # update with your info
BASE_HEADERS = {"User-Agent": USER_AGENT}

# CIK lookup requires the zero-padded 10-digit SEC company identifier.
# You look these up once at https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany
# Defense/aero tickers we're targeting:
KNOWN_CIKS = {
    "LMT": "0000936468",   # Lockheed Martin
    "RTX": "0000101829",   # RTX Corporation
    "NOC": "0001133421",   # Northrop Grumman
    "RKLB": "0001819994",  # Rocket Lab
}


def get_company_filings(ticker: str, form_types: list[str] | None = None, limit: int = 5) -> dict:
    """
    Fetch recent filings metadata for a company.

    Returns a dict with keys like 'form', 'filingDate', 'accessionNumber', 'primaryDocument'
    for each recent filing, filtered to form_types if given (e.g. ["10-K", "10-Q", "8-K"]).
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
        filings.append(
            {
                "form": recent["form"][i],
                "filingDate": recent["filingDate"][i],
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
    }


def get_filing_document_url(ticker: str, accession_number: str, primary_document: str) -> str:
    """Build the direct URL to a filing's primary document (e.g. the actual 10-K HTML)."""
    cik = KNOWN_CIKS.get(ticker.upper())
    accession_nodash = accession_number.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodash}/{primary_document}"


if __name__ == "__main__":
    # quick manual test
    result = get_company_filings("LMT", form_types=["10-K", "10-Q", "8-K"], limit=5)
    print(f"{result['company_name']} ({result['ticker']}) — {len(result['filings'])} recent filings:")
    for f in result["filings"]:
        print(f"  {f['form']:6s} filed {f['filingDate']}  {f['primaryDocument']}")
