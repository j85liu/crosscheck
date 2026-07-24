"""
OSINT / geopolitical analyst agent — STUB.

Not implemented yet. Planned data sources: Sentinel Hub / Copernicus (free
satellite imagery) and GDELT's event database (geopolitical events, distinct
from the news-sentiment use of GDELT in sentiment_analyst.py).

The orchestrator can route to this agent once ENABLED is set True and run()
is implemented — until then it returns a placeholder so the graph doesn't break.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas.models import OSINTAnalysis

ENABLED = False


def run(ticker: str) -> OSINTAnalysis:
    if not ENABLED:
        return OSINTAnalysis(
            ticker=ticker.upper(),
            geopolitical_factors=[],
            summary="OSINT agent not yet implemented — placeholder output.",
        )
    raise NotImplementedError("Wire up Sentinel Hub / GDELT events here.")


if __name__ == "__main__":
    print(run("LMT").model_dump_json(indent=2))