"""
Sentinel Hub / Copernicus satellite imagery — STUB, not implemented.

Once wired up, this would:
1. Look up known facility coordinates for the ticker (e.g. Lockheed's Fort Worth F-35
   final assembly plant, Boeing's Everett/Renton plants, SpaceX's Starbase/Hawthorne) in
   a FACILITY_COORDS mapping analogous to KNOWN_CIKS/RECIPIENT_NAMES elsewhere in this repo.
2. Authenticate against Sentinel Hub's OAuth2 API using a free Copernicus Data Space
   Ecosystem account (client_id/client_secret — a new .env var, not yet added).
3. Pull recent Sentinel-2 L2A imagery for each facility's bounding box (10m resolution,
   ~5-day revisit) via the Process API or Statistical API.
4. Compare against a historical baseline (e.g. a simple parking-lot/rooftop pixel-count
   proxy) to flag meaningful changes — busier lots, new construction — as a rough proxy
   for production/facility activity.

This is a meaningfully bigger integration (OAuth setup, choosing and maintaining facility
coordinates, picking a real activity-detection method) than the rest of this project's
tools, so it's scoped out for now. Callers (osint_analyst.py) must handle a None return.
"""


def get_facility_imagery_summary(ticker: str) -> dict | None:
    """Not implemented — always returns None. See module docstring for the planned approach."""
    return None
