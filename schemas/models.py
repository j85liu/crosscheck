"""
Structured output schemas for all agents.

Every specialist agent returns one of these instead of free text. This is
what lets the synthesis/debate step compare findings across agents
programmatically (e.g. "does risk_analyst.flagged_anomaly conflict with
filings_analyst.sentiment?") rather than trying to parse prose.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class FilingHighlight(BaseModel):
    """A single notable fact pulled from a filing."""

    topic: str = Field(..., description="Short label, e.g. 'Revenue guidance', 'New contract'")
    detail: str = Field(..., description="1-2 sentence summary of the fact")
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        ..., description="Directional read on this specific fact"
    )
    source_form: str = Field(..., description="e.g. '10-K', '10-Q', '8-K'")
    filed_date: date | None = None


class FilingsAnalysis(BaseModel):
    """Output of the filings_analyst agent for one company."""

    ticker: str
    company_name: str
    overall_sentiment: Literal["positive", "negative", "neutral", "mixed"]
    highlights: list[FilingHighlight]
    summary: str = Field(..., description="2-3 sentence plain-language summary")


class RiskAnalysis(BaseModel):
    """Output of the risk_analyst agent for one company. Filled in later."""

    ticker: str
    realized_volatility_30d: float | None = None
    realized_volatility_90d: float | None = None
    price_change_30d_pct: float | None = None
    flagged_anomaly: bool = False
    anomaly_description: str | None = None
    summary: str


class ContractEvent(BaseModel):
    """A single notable contract/program award or milestone."""

    description: str = Field(..., description="1-2 sentence summary, e.g. '$2B contract awarded for F-35 sustainment'")
    value_usd: float | None = Field(None, description="Contract value in USD if known")
    date: str | None = Field(None, description="Award/event date, YYYY-MM-DD if known")
    significance: Literal["major", "minor"] = "minor"


class ProgramAnalysis(BaseModel):
    """Output of the program_analyst agent for one company."""

    ticker: str
    company_name: str
    recent_events: list[ContractEvent]
    overall_sentiment: Literal["positive", "negative", "neutral", "mixed"]
    summary: str


class SentimentAnalysis(BaseModel):
    """Output of the sentiment_analyst agent for one company."""

    ticker: str
    overall_sentiment: Literal["positive", "negative", "neutral", "mixed"]
    key_themes: list[str] = Field(..., description="Short phrases capturing what news/commentary is focused on")
    sentiment_trend: Literal["improving", "stable", "worsening"] = "stable"
    avg_gdelt_tone: float | None = Field(
        None, description="Count-weighted average GDELT tone score for the query, if available"
    )
    summary: str


class OSINTAnalysis(BaseModel):
    """Output of the osint_analyst agent for one company. Stubbed until wired up."""

    ticker: str
    geopolitical_factors: list[str] = Field(default_factory=list)
    summary: str = "Not yet implemented."


class Contradiction(BaseModel):
    """A flagged disagreement between two or more agents' findings."""

    agents_involved: list[str] = Field(..., description="e.g. ['filings_analyst', 'risk_analyst']")
    description: str = Field(..., description="What the contradiction actually is")
    severity: Literal["high", "medium", "low"] = "medium"


class VerificationFlag(BaseModel):
    """A claim from the synthesis report checked against its cited source."""

    claim: str = Field(..., description="The specific claim being checked, verbatim from the draft report")
    status: Literal["supported", "unsupported", "overstated"] = Field(
        ...,
        description="supported = source backs this up; unsupported = no evidence in source; "
        "overstated = source has related info but this claim goes beyond what it actually says",
    )
    note: str = Field(..., description="Brief explanation of the verification finding")
    corrected_claim: str | None = Field(
        None,
        description="For status=overstated: a corrected version of the claim scoped to what the source actually supports",
    )


class SynthesisReport(BaseModel):
    """Final output of the synthesis/debate step — this is the actual deliverable."""

    ticker: str
    company_name: str
    headline: str = Field(..., description="1-2 sentence overall takeaway")
    key_findings: list[str] = Field(..., description="Bulleted findings merged across agents, in plain language")
    contradictions: list[Contradiction] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list, description="Which agents contributed data")
    verification_flags: list[VerificationFlag] = Field(
        default_factory=list, description="Self-check results from the critic pass; empty if verification hasn't run"
    )


class AgentRun(BaseModel):
    """Generic wrapper so the orchestrator can log/route any agent's output uniformly."""

    agent_name: str
    ticker: str
    success: bool
    error: str | None = None
    data: dict