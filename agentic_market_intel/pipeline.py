"""Shared execution path used by both the CLI and the API — runs the graph
(live or mock), times it, and persists a Run + Report row. Keeping this in
one place means the FastAPI endpoint and the scheduler can't drift from
what `main.py` does.
"""

import time
from datetime import datetime

import pandas as pd

from agentic_market_intel.db import Report, Run, SessionLocal, init_db
from agentic_market_intel.llm import get_llm, is_live


def run_pipeline(df: pd.DataFrame, company: str) -> dict:
    init_db()
    session = SessionLocal()
    mode = "live" if is_live() else "mock"

    run = Run(company=company, mode=mode, started_at=datetime.utcnow(), status="running")
    session.add(run)
    session.commit()
    session.refresh(run)

    start = time.time()
    try:
        if mode == "live":
            from agentic_market_intel.graph import run_for_company
            state = run_for_company(df, get_llm(), company)
        else:
            from agentic_market_intel.mock import run_mock_pipeline
            state = run_mock_pipeline(df, company)

        critic_approved = bool(state["critic_verdict"].approved)

        run.finished_at = datetime.utcnow()
        run.duration_seconds = int(time.time() - start)
        run.critic_approved = critic_approved
        run.revise_count = state.get("revise_count", 0)
        run.alert = bool(state.get("alert", False))
        run.status = "approved" if critic_approved else "forced_unverified"
        session.commit()

        report = Report(run_id=run.id, company=company, report_text=state["final_report"])
        session.add(report)
        session.commit()
        session.refresh(report)

        return {
            "run_id": run.id,
            "report_id": report.id,
            "company": company,
            "mode": mode,
            "status": run.status,
            "critic_approved": critic_approved,
            "revise_count": run.revise_count,
            "alert": run.alert,
            "duration_seconds": run.duration_seconds,
            "report": state["final_report"],
        }
    except Exception as e:
        run.finished_at = datetime.utcnow()
        run.duration_seconds = int(time.time() - start)
        run.status = "failed"
        run.error = str(e)
        session.commit()
        raise
    finally:
        session.close()
