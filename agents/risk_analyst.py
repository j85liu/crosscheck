"""
Risk analyst agent.

Pulls price/volatility data via Alpha Vantage, then uses Claude to interpret
the numbers into a structured RiskAnalysis — including deciding whether
anything looks like an anomaly worth flagging to the synthesis step.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.client import SPECIALIST_MODEL, structured_call
from schemas.models import RiskAnalysis
from tools.alphavantage import compute_volatility_summary

PROMPT_TEMPLATE = """You are a risk analyst. Below are computed volatility statistics for
{ticker}:

- Latest close: {latest_close}
- 30-day realized volatility (annualized): {vol_30d}
- 90-day realized volatility (annualized): {vol_90d}
- 30-day price change: {price_change_30d}%

Interpret these numbers. Flag flagged_anomaly=true only if something looks genuinely
notable (e.g. 30-day vol meaningfully elevated vs 90-day vol, or a large price move
without an obvious multi-day trend). Don't over-flag normal market noise.

Respond with ONLY this JSON structure, no other text:
{{
  "ticker": "{ticker}",
  "realized_volatility_30d": {vol_30d_num},
  "realized_volatility_90d": {vol_90d_num},
  "price_change_30d_pct": {price_change_num},
  "flagged_anomaly": true | false,
  "anomaly_description": "..." or null,
  "summary": "2-3 sentence plain-language summary"
}}
"""


def run(ticker: str) -> RiskAnalysis:
    stats = compute_volatility_summary(ticker)

    def fmt(v):
        return f"{v:.4f}" if v is not None else "N/A"

    def num(v):
        return "null" if v is None else v

    prompt = PROMPT_TEMPLATE.format(
        ticker=stats["ticker"],
        latest_close=stats["latest_close"],
        vol_30d=fmt(stats["realized_volatility_30d"]),
        vol_90d=fmt(stats["realized_volatility_90d"]),
        price_change_30d=fmt(stats["price_change_30d_pct"]),
        vol_30d_num=num(stats["realized_volatility_30d"]),
        vol_90d_num=num(stats["realized_volatility_90d"]),
        price_change_num=num(stats["price_change_30d_pct"]),
    )

    result_dict = structured_call(prompt, model=SPECIALIST_MODEL, max_tokens=512)
    return RiskAnalysis(**result_dict)


if __name__ == "__main__":
    analysis = run("LMT")
    print(analysis.model_dump_json(indent=2))