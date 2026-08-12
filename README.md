# CrossCheck

A multi-agent equity research system built with LangGraph and the Anthropic API.
Five specialist agents independently analyze a company from different angles, then
a synthesis step cross-checks their findings for contradictions — and fact-checks
its own draft against the source data — before generating a concise research memo.

Currently scoped to defense/aerospace tickers: LMT, RTX, NOC, RKLB, BA, GD, SPCX.

## Architecture

```
                    ┌──────────────┐
                    │ Research     │
                    │ request      │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Orchestrator │  (LangGraph)
                    └──────┬───────┘
        ┌──────────┬───────┼───────┬──────────┐
        ▼          ▼       ▼       ▼          ▼
   ┌────────┐ ┌────────┐┌──────┐┌──────┐ ┌─────────┐
   │Filings │ │Program ││Sentmt││Risk  │ │OSINT    │
   │analyst │ │analyst ││analyst│analyst│ │analyst  │
   └────┬───┘ └────┬───┘└───┬──┘└──┬───┘ └────┬────┘
        └──────────┴────────┴──────┴──────────┘
                           │
                    ┌──────▼───────┐
                    │  Synthesis / │
                    │    Debate    │
                    │  + self-check│
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Research memo│
                    └──────────────┘
```

Each specialist agent pulls data from a free public API and uses Claude to turn it
into a structured output (see `schemas/models.py`). Three of them — filings, risk,
and sentiment — are real tool-calling agents built on the generic harness in
`llm/agent_loop.py`: the model decides for itself which tools to call, in what
order, and when it has enough to answer (e.g. the filings agent only pulls a
filing's full text when something in the metadata or financials looks worth a
closer look; the risk agent only checks sector peers when its own volatility
looks anomalous). Program and OSINT are simpler one-shot pattern (pull data, one
LLM call). Every `[tool call] ...` line printed during a run is that decision
trail, not a fixed script.

The synthesis step uses a stronger model (Sonnet) to cross-check all five agents'
findings and explicitly flag contradictions — e.g. positive contract news paired
with unexplained negative price action — rather than just concatenating everything
into one report. A second pass then fact-checks the draft's own claims against the
underlying agent outputs: unsupported claims are dropped and overstated ones are
softened, with every check recorded in the report so nothing disappears silently.

### Point-in-time analysis

Every tool and agent accepts an optional `as_of_date`, which restricts it to data
that would genuinely have been available on that date (not just labeled with an
old date) — e.g. SEC filings are filtered by when they were *filed*, not the
fiscal period they cover, so a quarter that ended before the cutoff but wasn't
filed until after it is correctly excluded. This is what makes point-in-time
backtesting meaningful: `--as-of` picks a past date and the whole pipeline — tool
calls, agent reasoning, and synthesis — is restricted to what was knowable then.

## Setup

```bash
pip install -r requirements.txt  # or conda install the equivalents
cp .env.example .env             # then add your real API keys
```

You'll need:
- `ANTHROPIC_API_KEY` — https://console.anthropic.com
- `ALPHAVANTAGE_API_KEY` — https://www.alphavantage.co/support/#api-key (free tier)

SEC EDGAR, USASpending.gov, and GDELT require no API key.

## Usage

```bash
python demo/app.py LMT
python demo/app.py LMT RTX NOC RKLB
python demo/app.py LMT --as-of 2026-06-01   # point-in-time: only data knowable by that date
```

## Project structure

- `agents/` — one module per specialist agent, plus `orchestrator.py` (the LangGraph graph)
- `llm/agent_loop.py` — the generic tool-calling harness the filings/risk/sentiment agents run on
- `llm/client.py` — the only file that calls the Anthropic API directly
- `tools/` — thin wrappers around each external API (SEC EDGAR, Alpha Vantage,
  USASpending.gov, GDELT news + GDELT events, and a stubbed Sentinel Hub satellite
  imagery module for future work)
- `synthesis/debate.py` — the cross-check/contradiction-flagging step, plus its self-verification pass
- `schemas/models.py` — Pydantic schemas every agent's output must match
- `demo/app.py` — CLI entry point

## Status

Built as a course project for [course name] — Option #3 (build an app based on FOSS code).
All five agents implemented (OSINT covers geopolitical events; satellite imagery is
still stubbed). Known limitation: OSINT's underlying data source doesn't yet respect
`as_of_date`, so its output isn't point-in-time correct like the other four agents.
