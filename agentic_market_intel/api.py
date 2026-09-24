"""FastAPI service layer.

    uvicorn agentic_market_intel.api:app --reload
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

load_dotenv()

from agentic_market_intel.data import build_dataset, build_live_dataset
from agentic_market_intel.db import AgentCall, Report, Run, SessionLocal, init_db
from agentic_market_intel.pipeline import run_pipeline
from agentic_market_intel.scheduler import start_scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    tracked = os.environ.get("TRACKED_COMPANIES")
    if tracked:
        start_scheduler([c.strip() for c in tracked.split(",") if c.strip()])
    yield


app = FastAPI(title="Agentic Market Intelligence API", lifespan=lifespan)


class RunRequest(BaseModel):
    company: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/runs")
def trigger_run(req: RunRequest):
    """Trigger an on-demand analysis for a company. Runs synchronously —
    a live run typically takes 30-90s due to the multi-agent tool-calling
    loop plus any rate-limit backoff."""
    if not req.company.strip():
        raise HTTPException(status_code=400, detail="company is required")

    news_key = os.environ.get("NEWS_API_KEY")
    df = build_live_dataset([req.company], news_key) if news_key else build_dataset()

    try:
        return run_pipeline(df, req.company)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/runs/{run_id}")
def get_run(run_id: int):
    """Run metadata plus the per-agent latency/token/success breakdown —
    the observability story: every node execution is one queryable row."""
    session = SessionLocal()
    try:
        run = session.get(Run, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        calls = (
            session.query(AgentCall)
            .filter(AgentCall.run_id == run_id)
            .order_by(AgentCall.started_at.asc())
            .all()
        )
        return {
            "id": run.id,
            "company": run.company,
            "mode": run.mode,
            "status": run.status,
            "critic_approved": run.critic_approved,
            "revise_count": run.revise_count,
            "alert": run.alert,
            "duration_seconds": run.duration_seconds,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "finished_at": run.finished_at.isoformat() if run.finished_at else None,
            "agent_calls": [
                {
                    "node_name": c.node_name,
                    "latency_seconds": c.latency_seconds,
                    "tokens_total": c.tokens_total,
                    "success": c.success,
                    "error": c.error,
                }
                for c in calls
            ],
        }
    finally:
        session.close()


@app.get("/reports/{report_id}")
def get_report(report_id: int):
    session = SessionLocal()
    try:
        report = session.get(Report, report_id)
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        return {
            "id": report.id,
            "run_id": report.run_id,
            "company": report.company,
            "created_at": report.created_at.isoformat(),
            "report_text": report.report_text,
        }
    finally:
        session.close()


@app.get("/reports")
def list_reports(company: Optional[str] = Query(None), limit: int = Query(50, le=200)):
    session = SessionLocal()
    try:
        q = session.query(Report)
        if company:
            q = q.filter(Report.company.ilike(company))
        reports = q.order_by(Report.created_at.desc()).limit(limit).all()
        return [
            {"id": r.id, "run_id": r.run_id, "company": r.company, "created_at": r.created_at.isoformat()}
            for r in reports
        ]
    finally:
        session.close()
