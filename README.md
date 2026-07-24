# CrossCheck

A multi-agent equity research system built with LangGraph and the Anthropic API.
Four specialist agents independently analyze a company from different angles, then
a synthesis step cross-checks their findings for contradictions before generating
a concise research memo.

Currently scoped to defense/aerospace tickers: LMT, RTX, NOC, RKLB.

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
   │analyst │ │analyst ││analyst│analyst│ │(stubbed)│
   └────┬───┘ └────┬───┘└───┬──┘└──┬───┘ └────┬────┘
        └──────────┴────────┴──────┴──────────┘
                           │
                    ┌──────▼───────┐
                    │  Synthesis / │
                    │    Debate    │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Research memo│
                    └──────────────┘
```

Each specialist agent pulls data from a free public API, then uses Claude (Haiku,
for cost efficiency) to turn it into a structured output. The synthesis step uses
a stronger model (Sonnet) to cross-check all four agents' findings and explicitly
flag contradictions — e.g. positive contract news paired with unexplained negative
price action — rather than just concatenating everything into one report.

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
```

## Project structure

- `agents/` — one module per specialist agent, plus `orchestrator.py` (the LangGraph graph)
- `tools/` — thin wrappers around each external API
- `synthesis/debate.py` — the cross-check/contradiction-flagging step
- `schemas/models.py` — Pydantic schemas every agent's output must match
- `llm/client.py` — the only file that calls the Anthropic API directly
- `demo/app.py` — CLI entry point

## Status

Built as a course project for [course name] — Option #3 (build an app based on FOSS code).
4-agent version implemented; OSINT agent stubbed for future work.