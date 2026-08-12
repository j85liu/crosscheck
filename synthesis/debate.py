"""
Synthesis and debate step.

Takes the structured outputs from all specialist agents and asks Claude to:
1. Merge non-conflicting findings into a coherent narrative
2. Explicitly flag contradictions between agents (this is the actual
   research contribution of the project — most pipelines skip this step
   and just concatenate agent outputs)
3. Self-check (verify()) the draft from steps 1-2 against the actual source
   agent outputs, catching cases where the draft overstates, invents
   specifics, or misattributes something to a source that doesn't support
   it — unsupported claims are dropped, overstated ones are softened, and
   every check is recorded in the final report's verification_flags so
   nothing disappears silently.

Uses REASONING_MODEL (not the cheaper specialist model) for both the draft
and verification passes, since each requires holding multiple sources in
mind and comparing them, which is a harder task than any single agent's
extraction work.
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.client import REASONING_MODEL, structured_call
from schemas.models import (
    Contradiction,
    FilingsAnalysis,
    OSINTAnalysis,
    ProgramAnalysis,
    RiskAnalysis,
    SentimentAnalysis,
    SynthesisReport,
    VerificationFlag,
)

PROMPT_TEMPLATE = """You are a senior research analyst reviewing findings from several specialist
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
with unexplained bullish sentiment (possible overreaction). If OSINT/geopolitical findings
are present above, cross-check them against the other four too — e.g. does a geopolitical
risk or opportunity story align with what filings/program/sentiment/risk are already
showing, or does it suggest something the market hasn't priced in yet? A contradiction is
only worth flagging if it's a genuine, specific disagreement — not just "different agents
mentioned different things."

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


VERIFY_PROMPT_TEMPLATE = """You are a fact-checker reviewing a draft research synthesis on {ticker}
before it ships. Below are the draft's claims (key findings and contradictions), followed by the
actual source data from each specialist agent the draft is supposed to be based on.

For each claim, check whether it is directly supported by the source data below — don't accept a
claim just because it sounds plausible or internally consistent. Only mark "supported" if you can
point to specific text in the source that backs it up. Be skeptical of specific-sounding details
(numbers, named events, causal explanations) that don't clearly trace back to a source. Mark
"overstated" if the source has related information but the claim goes further than what the source
actually says (e.g. asserting a cause the source didn't state, or a figure the source didn't give)
— for these, also provide a corrected_claim rewritten to only what the source actually supports.
Mark "unsupported" if there's no evidence for the claim anywhere in the source data.

=== DRAFT KEY FINDINGS ===
{key_findings_list}

=== DRAFT CONTRADICTIONS ===
{contradictions_list}

=== SOURCE DATA (ground truth) ===

--- FILINGS ANALYST ---
Summary: {filings_summary}
Highlights:
{filings_highlights}

--- PROGRAM/CONTRACT ANALYST ---
Summary: {program_summary}

--- SENTIMENT ANALYST ---
Summary: {sentiment_summary}
Key themes: {sentiment_key_themes}

--- RISK ANALYST ---
Summary: {risk_summary}
Numeric data: 30d realized vol={risk_vol_30d}, 90d realized vol={risk_vol_90d}, 30d price
change={risk_price_change}%, flagged_anomaly={risk_flagged_anomaly}, anomaly_description={risk_anomaly_description}

{osint_section}
Produce one verification flag per item listed under DRAFT KEY FINDINGS and DRAFT CONTRADICTIONS
above — every single one, none skipped. Each contradiction below is shown as its description text
on its own line, prefixed with metadata (severity, agents involved) on a separate line above it —
that metadata is context only. The "claim" field MUST be the exact verbatim description text alone
(copied character for character, WITHOUT the "[severity] agents:" metadata prefix) for
contradictions, or the exact verbatim key finding text for key findings, so it can be matched back
programmatically.

Respond with ONLY this JSON structure, no other text:
{{
  "verification_flags": [
    {{"claim": "...", "status": "supported"|"unsupported"|"overstated", "note": "...",
      "corrected_claim": "..." or null}}
  ]
}}
"""


def _format_key_findings(findings: list[str]) -> str:
    return "\n".join(f"- {f}" for f in findings) if findings else "(none)"


def _format_contradictions(contradictions: list[Contradiction]) -> str:
    if not contradictions:
        return "(none)"
    return "\n".join(
        f"[severity: {c.severity}] [agents: {', '.join(c.agents_involved)}]\ndescription: {c.description}"
        for c in contradictions
    )


def _format_highlights(highlights: list) -> str:
    if not highlights:
        return "  (none)"
    return "\n".join(
        f"  - [{h.sentiment}] {h.topic}: {h.detail} (source: {h.source_form}, filed {h.filed_date})"
        for h in highlights
    )


def _format_osint_section(osint: OSINTAnalysis | None) -> str:
    if not osint or osint.summary == "OSINT agent not yet implemented — placeholder output.":
        return ""
    factors = "\n".join(f"  - {f}" for f in osint.geopolitical_factors) or "  (none)"
    return f"--- OSINT/GEOPOLITICAL ANALYST ---\nSummary: {osint.summary}\nGeopolitical factors:\n{factors}\n"


def verify(
    draft_report: SynthesisReport,
    filings: FilingsAnalysis,
    program: ProgramAnalysis,
    sentiment: SentimentAnalysis,
    risk: RiskAnalysis,
    osint: OSINTAnalysis | None = None,
    as_of_date: date | None = None,
) -> SynthesisReport:
    """
    Check every key finding and contradiction in draft_report against the actual source
    agent outputs. Unsupported claims are dropped from the returned report; overstated
    ones are replaced with their corrected_claim (if the model provided one); supported
    ones pass through unchanged. Every check — including drops and corrections — is kept
    in the returned report's verification_flags so nothing disappears silently.

    Claims are matched back to flags primarily by exact string equality against the
    "claim" field, which the prompt instructs the model to copy verbatim; as a fallback
    for near-verbatim echoes (trailing punctuation, minor rewording), substring
    containment in either direction is also accepted (see _find_flag). If no flag matches
    a claim at all, that claim is left as-is rather than dropped — a missing/unmatched
    flag is treated as "couldn't verify," not "unsupported."
    """
    prompt = VERIFY_PROMPT_TEMPLATE.format(
        ticker=draft_report.ticker,
        key_findings_list=_format_key_findings(draft_report.key_findings),
        contradictions_list=_format_contradictions(draft_report.contradictions),
        filings_summary=filings.summary,
        filings_highlights=_format_highlights(filings.highlights),
        program_summary=program.summary,
        sentiment_summary=sentiment.summary,
        sentiment_key_themes=", ".join(sentiment.key_themes) if sentiment.key_themes else "(none)",
        risk_summary=risk.summary,
        risk_vol_30d=risk.realized_volatility_30d,
        risk_vol_90d=risk.realized_volatility_90d,
        risk_price_change=risk.price_change_30d_pct,
        risk_flagged_anomaly=risk.flagged_anomaly,
        risk_anomaly_description=risk.anomaly_description,
        osint_section=_format_osint_section(osint),
    )
    if as_of_date is not None:
        prompt = (
            f"You are fact-checking a report analyzing data as of {as_of_date.isoformat()}. Do not "
            f"reference or assume knowledge of anything after this date.\n\n{prompt}"
        )

    # Much bigger budget than the draft call: verify's output echoes both the original
    # claim and (for overstated items) a corrected_claim, plus a note, for every finding
    # and contradiction. Measured directly against a real 5-agent LMT run: this genuinely
    # needed ~5140 output tokens (8 flags) and silently truncated mid-JSON at 4096 — not
    # a rare edge case, just the actual size of this task once OSINT is in the mix.
    result_dict = structured_call(prompt, model=REASONING_MODEL, max_tokens=8192)
    flags = [VerificationFlag(**f) for f in result_dict.get("verification_flags", [])]

    def _find_flag(text: str) -> VerificationFlag | None:
        for f in flags:
            if f.claim == text:
                return f
        for f in flags:
            if text in f.claim or f.claim in text:
                return f
        return None

    final_key_findings = []
    for finding in draft_report.key_findings:
        flag = _find_flag(finding)
        if flag is None or flag.status == "supported":
            final_key_findings.append(finding)
        elif flag.status == "overstated":
            final_key_findings.append(flag.corrected_claim or finding)
        # status == "unsupported" -> dropped, but stays visible in verification_flags

    final_contradictions = []
    for c in draft_report.contradictions:
        flag = _find_flag(c.description)
        if flag is None or flag.status == "supported":
            final_contradictions.append(c)
        elif flag.status == "overstated":
            final_contradictions.append(c.model_copy(update={"description": flag.corrected_claim or c.description}))
        # status == "unsupported" -> dropped, but stays visible in verification_flags

    return draft_report.model_copy(
        update={
            "key_findings": final_key_findings,
            "contradictions": final_contradictions,
            "verification_flags": flags,
        }
    )


def run(
    ticker: str,
    filings: FilingsAnalysis,
    program: ProgramAnalysis,
    sentiment: SentimentAnalysis,
    risk: RiskAnalysis,
    osint: OSINTAnalysis | None = None,
    as_of_date: date | None = None,
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
    if as_of_date is not None:
        prompt = (
            f"You are synthesizing a report analyzing data as of {as_of_date.isoformat()}. Do not "
            f"reference or assume knowledge of anything after this date.\n\n{prompt}"
        )

    result_dict = structured_call(prompt, model=REASONING_MODEL, max_tokens=4096)
    draft_report = SynthesisReport(**result_dict)
    return verify(draft_report, filings, program, sentiment, risk, osint, as_of_date=as_of_date)


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