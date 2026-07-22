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


class AgentRun(BaseModel):
    """Generic wrapper so the orchestrator can log/route any agent's output uniformly."""

    agent_name: str
    ticker: str
    success: bool
    error: str | None = None
    data: dict
