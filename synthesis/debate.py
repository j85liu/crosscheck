"""
Synthesis and debate step.

Takes the structured outputs from all specialist agents and asks Claude to:
1. Merge non-conflicting findings into a coherent narrative
2. Explicitly flag contradictions between agents (this is the actual
   research contribution of the project — most pipelines skip this step
   and just concatenate agent outputs)

Uses REASONING_MODEL (not the cheaper specialist model) since this step
requires holding multiple sources in mind and comparing them, which is a
harder task than any single agent's extraction work.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.client import REASONING_MODEL, structured_call
from schemas.models import (
    FilingsAnalysis,
    OSINTAnalysis,
    ProgramAnalysis,
    RiskAnalysis,
    SentimentAnalysis,
    SynthesisReport,
)

PROMPT_TEMPLATE = """You are a senior research analyst reviewing findings from four specialist
analysts on {ticker}. Your job is to synthesize their findings into one coherent view AND
explicitly flag any contradictions between what they found — this is the most important
part of your job. Don't smooth over disagreements; surface them.

=== FILINGS ANALYST ===
{filings_summary}

=== PROGRAM/CONTRACT ANALYST ===
{program_summary}

=== SENTIMENT ANALYST ===
{sentiment_summary}

=== RISK ANALYST ===
{risk_summary}

{osint_section}

Look specifically for cases like: positive filings/contract news but negative or unchanged
market sentiment/volatility (possible market skepticism or lag), or negative fundamentals
with unexplained bullish sentiment (possible overreaction). A contradiction is only worth
flagging if it's a genuine, specific disagreement — not just "different agents mentioned
different things."

Respond with ONLY this JSON structure, no other text:
{{
  "ticker": "{ticker}",
  "company_name": "{company_name}",
  "headline": "1-2 sentence overall takeaway",
  "key_findings": ["...", "...", "..."],
  "contradictions": [
    {{"agents_involved": ["...", "..."], "description": "...", "severity": "high"|"medium"|"low"}}
  ],
  "sources": ["filings_analyst", "program_analyst", "sentiment_analyst", "risk_analyst"]
}}
"""


def run(
    ticker: str,
    filings: FilingsAnalysis,
    program: ProgramAnalysis,
    sentiment: SentimentAnalysis,
    risk: RiskAnalysis,
    osint: OSINTAnalysis | None = None,
) -> SynthesisReport:
    osint_section = ""
    if osint and osint.summary != "OSINT agent not yet implemented — placeholder output.":
        osint_section = f"=== OSINT/GEOPOLITICAL ANALYST ===\n{osint.summary}\n"

    prompt = PROMPT_TEMPLATE.format(
        ticker=ticker,
        company_name=filings.company_name,
        filings_summary=filings.summary,
        program_summary=program.summary,
        sentiment_summary=sentiment.summary,
        risk_summary=risk.summary,
        osint_section=osint_section,
    )

    result_dict = structured_call(prompt, model=REASONING_MODEL, max_tokens=1536)
    return SynthesisReport(**result_dict)


if __name__ == "__main__":
    # quick manual test with dummy data — real usage goes through the orchestrator
    dummy_filings = FilingsAnalysis(
        ticker="LMT", company_name="Lockheed Martin", overall_sentiment="positive",
        highlights=[], summary="Filed 10-Q showing revenue growth and raised guidance."
    )
    dummy_program = ProgramAnalysis(
        ticker="LMT", company_name="Lockheed Martin", recent_events=[],
        overall_sentiment="positive", summary="Awarded major new contract for F-35 sustainment."
    )
    dummy_sentiment = SentimentAnalysis(
        ticker="LMT", overall_sentiment="neutral", key_themes=["earnings", "defense budget"],
        sentiment_trend="stable", summary="Coverage neutral, focused on broader defense budget debates."
    )
    dummy_risk = RiskAnalysis(
        ticker="LMT", realized_volatility_30d=0.18, realized_volatility_90d=0.15,
        price_change_30d_pct=-2.1, flagged_anomaly=True,
        anomaly_description="Price declined despite positive fundamentals.",
        summary="Elevated 30-day vol and a price decline, despite no negative fundamental news."
    )
    report = run("LMT", dummy_filings, dummy_program, dummy_sentiment, dummy_risk)
    print(report.model_dump_json(indent=2))