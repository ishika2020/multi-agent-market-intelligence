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
from agentic_market_intel.db import Report, SessionLocal, init_db
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
