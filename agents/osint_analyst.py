"""
OSINT / geopolitical analyst agent.

Pulls structured geopolitical events from GDELT's Events database (CAMEO-coded, with a
Goldstein conflict/cooperation score) for countries tied to a ticker's international
program exposure, then uses Claude to characterize how — if at all — those events relate
to the company's business. Satellite imagery (tools/sentinel_hub.py) is stubbed out for
now and always returns None; this agent handles that gracefully.
"""

import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.client import SPECIALIST_MODEL, structured_call
from schemas.models import OSINTAnalysis
from tools.gdelt_events import EVENT_ROOT_LABELS, get_geopolitical_events
from tools.sentinel_hub import get_facility_imagery_summary

ENABLED = True

PROMPT_TEMPLATE = """You are a geopolitical/OSINT analyst for a defense/aerospace equity
research team. Below are recent geopolitical events (from GDELT's structured event
database, CAMEO-coded) tied to countries relevant to {ticker}'s international program
exposure: {countries_desc}.

Each event includes a Goldstein score (-10 to +10: how conflictual vs. cooperative it is)
and GDELT's tone score for coverage of it (-10 to +10, negative = more negative coverage).
This is raw, automatically-coded event data — some entries may be geocoding noise or
unrelated stories that happen to touch a relevant country/actor; use judgment and don't
force a connection to {ticker} that isn't genuinely there.

Events (most notable by |Goldstein score| first):
{events_list}
{imagery_section}
Characterize how these events, if at all, relate to {ticker}'s business — e.g. escalating
tension in a country with active program exposure could signal opportunity (increased
defense spending/exports) or risk (program disruption, export control changes), depending
on context. If nothing in the event list is genuinely relevant to {ticker}, say so plainly
in the summary rather than forcing a connection, and leave geopolitical_factors empty.

Respond with ONLY this JSON structure, no other text:
{{
  "ticker": "{ticker}",
  "geopolitical_factors": ["short bullet-style string tying a specific event to {ticker}'s business, e.g. 'Escalating tension in X (Goldstein: -6.2) coinciding with active program presence there'", "..."],
  "summary": "2-3 sentence plain-language summary"
}}
"""


def _format_events(events: list[dict]) -> str:
    if not events:
        return "No events found in the queried window."
    lines = []
    for e in events:
        label = EVENT_ROOT_LABELS.get(e["event_root_code"], e["event_root_code"])
        goldstein = e["goldstein_scale"] if e["goldstein_scale"] is not None else 0.0
        tone = e["avg_tone"] if e["avg_tone"] is not None else 0.0
        domain = urlparse(e["source_url"]).netloc
        location = e["location"] or e["country"]
        lines.append(
            f"- {e['date']}: [{label}] {location} "
            f"(Goldstein: {goldstein:+.1f}, tone: {tone:+.1f}, {e['num_mentions']} mentions) — {domain}"
        )
    return "\n".join(lines)


def run(ticker: str) -> OSINTAnalysis:
    if not ENABLED:
        return OSINTAnalysis(
            ticker=ticker.upper(),
            geopolitical_factors=[],
            summary="OSINT agent not yet implemented — placeholder output.",
        )

    events_raw = get_geopolitical_events(ticker)
    imagery = get_facility_imagery_summary(ticker)  # always None for now — stubbed

    if not events_raw["events"]:
        return OSINTAnalysis(
            ticker=events_raw["ticker"],
            geopolitical_factors=[],
            summary=(
                f"No notable geopolitical signal in the queried window for countries tied to "
                f"{events_raw['ticker']}'s program exposure ({', '.join(events_raw['countries'])})."
            ),
        )

    imagery_section = f"\nSatellite imagery signal:\n{imagery}\n" if imagery else ""
    countries_desc = f"{', '.join(events_raw['countries'])} ({events_raw['rationale']})"

    prompt = PROMPT_TEMPLATE.format(
        ticker=events_raw["ticker"],
        countries_desc=countries_desc,
        events_list=_format_events(events_raw["events"]),
        imagery_section=imagery_section,
    )

    result_dict = structured_call(prompt, model=SPECIALIST_MODEL, max_tokens=1024)
    return OSINTAnalysis(**result_dict)


if __name__ == "__main__":
    print(run("LMT").model_dump_json(indent=2))

