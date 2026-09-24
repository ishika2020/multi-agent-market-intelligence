# Multi-Agent Market Intelligence

An agentic AI system that turns raw news coverage into a verified, evidence-backed
market intelligence report for any company — on demand or on a schedule.

Five agents run as a LangGraph state machine: **Research** and **Analysis** decide
for themselves which tools to call to gather and score data, **Insight** turns
findings into specific business claims (each tied to a source record), and a
**Critic** independently re-checks every claim against the original evidence
before anything ships — rejecting and sending work back to Insight when a claim
doesn't hold up, bounded to one retry so the system can't loop forever
unattended. The result is served through a REST API, persisted to a database,
visible in a dashboard, and can be scheduled to re-run automatically per company.

## How it works

```
News data (NewsAPI, or built-in sample data)
        │
        ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Research   │ ──▶ │  Analysis   │ ──▶ │  Insight    │ ──▶ │   Critic    │
│   Agent     │     │   Agent     │     │   Agent     │     │   Agent     │
└─────────────┘     └─────────────┘     └──────┬──────┘     └──────┬──────┘
                                                 ▲                   │
                                                 └── reject, retry ──┘
                                                        (bounded)
                                                                     │
                                                              approve/forced
                                                                     ▼
                                                              ┌─────────────┐
                                                              │   Report    │
                                                              └─────────────┘
```

- **Research Agent** — fetches raw records for the company (total volume, sources, date range) and flags data-quality concerns.
- **Analysis Agent** — computes sentiment (TextBlob + VADER ensemble), trend direction, and optionally correlates against real stock price movement (yfinance), deciding which of these it actually needs.
- **Insight Agent** — fetches evidence itself and produces claims (opportunities, risks, observations), each citing the specific `evidence_id` it's grounded in.
- **Critic Agent** — independently re-fetches every cited evidence_id and judges whether the claim actually holds up. Approves, or rejects with specific feedback and sends it back to Insight for one bounded revision.
- **Report Agent** — formats the final report, including a transparent QA section showing whether claims were approved or forced through unverified.

If no cited evidence survives the retry, the report says so explicitly rather
than presenting unverified claims as fact.

## Features

- **Real tool-calling agents** — each agent decides which tool(s) to call and when to stop; nothing is hardcoded into a fixed call sequence.
- **Evidence-grounded claims** — every insight traces back to a specific source record, and that link is independently verified before publishing.
- **Live news ingestion** — pulls from NewsAPI per company, with automatic graceful fallback to a built-in sample dataset if the source is unavailable.
- **Price/sentiment correlation** — pulls real stock price trend via yfinance to check whether news sentiment agrees or diverges from price movement.
- **Persistent history** — every run and report is stored (SQLite by default, drop-in Postgres via `DATABASE_URL`).
- **Per-agent observability** — latency, token usage, and success/failure logged per node per run, not just at the run level.
- **REST API** — trigger runs and retrieve past reports/run details programmatically.
- **Dashboard** — a Streamlit UI to trigger runs and browse report history, reading from the API.
- **Scheduling** — track a set of companies and re-run analysis automatically at a configurable interval.
- **Mock mode** — the full pipeline runs deterministically with no LLM calls when no API key is configured, useful for CI and offline testing.
- **Rate-limit resilient** — LLM calls back off and retry automatically on provider rate limits.
- **Containerized** — Dockerfile + docker-compose (with local Postgres), ready to deploy to Railway or Render.
- **CI** — GitHub Actions runs the full pipeline and API test suite in mock mode on every push.

## Project structure

```
agentic_market_intel/
  data.py           # dataset loading (NewsAPI / CSV / sample data / yfinance), cleaning, sentiment scoring
  tools.py          # tools agents call: fetch, sentiment, trend, lookup_evidence, market_data
  state.py          # graph state + schemas (Claim, InsightsOutput, CriticVerdict)
  llm.py            # LLM client + rate-limit retry handling
  graph.py          # the LangGraph state machine
  mock.py           # deterministic no-LLM pipeline (fallback / CI)
  report.py         # report formatting, including the QA/Critic section
  db.py             # database models (Run, Report, AgentCall)
  observability.py  # per-agent latency/token/success logging
  pipeline.py       # shared run-and-persist logic used by the CLI and API
  scheduler.py      # periodic per-company runs
  api.py            # FastAPI service
main.py             # CLI entrypoint
dashboard.py        # Streamlit dashboard (reads from the API)
tests/              # API test suite (mock mode)
Dockerfile, docker-compose.yml, railway.json
.github/workflows/ci.yml
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Add to `.env`:
- `GROQ_API_KEY` — free at [console.groq.com](https://console.groq.com). Without it, the system runs in mock mode.
- `NEWS_API_KEY` — free at [newsapi.org](https://newsapi.org). Without it, the system uses built-in sample data.

## Usage — CLI

```bash
python main.py --companies Tesla,Apple
python main.py --company Tesla --csv my_data.csv
```

Each run prints the report, saves it to `./market_reports/`, and persists it to the database.

## Usage — API

```bash
uvicorn agentic_market_intel.api:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/runs -H "Content-Type: application/json" -d '{"company":"Tesla"}'
curl http://127.0.0.1:8000/reports?company=Tesla
curl http://127.0.0.1:8000/runs/1        # per-agent latency/token/success breakdown
curl http://127.0.0.1:8000/health
```

Set `TRACKED_COMPANIES=Tesla,Apple` (and optionally `SCHEDULE_INTERVAL_HOURS`,
default 6) to have the API start a background scheduler on boot.

## Usage — Dashboard

```bash
uvicorn agentic_market_intel.api:app --reload   # in one terminal
streamlit run dashboard.py                      # in another
```

Trigger new runs, view the per-agent breakdown, and browse past reports. Set
`API_BASE_URL` if the API isn't on `http://127.0.0.1:8000`.

## Database

Defaults to a local SQLite file. To use Postgres (e.g. a free Supabase/Neon
instance), set:

```bash
DATABASE_URL=postgresql://user:pass@host:5432/dbname
```

## Running with Docker

```bash
docker compose up --build
```

Brings up the API on port 8000 backed by a local Postgres container (verified
working — includes a healthcheck so the API waits for Postgres to actually
be ready before connecting). Reads the rest of your config from `.env`.

## Deploying to Railway

1. Push this repo to GitHub (already done if you're reading this from there).
2. On [railway.app](https://railway.app), New Project → Deploy from GitHub repo → select this repo. Railway auto-detects the `Dockerfile`.
3. Add a Postgres database: New → Database → PostgreSQL. Railway injects `DATABASE_URL` automatically if you reference it, or copy its connection string into your service's variables.
4. In the service's Variables tab, set `GROQ_API_KEY`, `NEWS_API_KEY`, and optionally `TRACKED_COMPANIES`.
5. Railway assigns `$PORT` automatically — the Dockerfile's `CMD` already reads it.
6. Deploy. Your API is live at the generated `*.up.railway.app` domain; point the Streamlit dashboard's `API_BASE_URL` at it if you want the dashboard hosted separately.

## Tests

```bash
pytest tests/ -v
```

Runs entirely in mock mode — no API keys required.

## Roadmap

- pgvector-backed semantic search over past reports (needs a real Postgres instance to build against)
