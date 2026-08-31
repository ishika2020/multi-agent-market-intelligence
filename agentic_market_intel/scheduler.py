"""APScheduler running in-process — no separate infra, per the design doc's
"start simple, mention Airflow as the upgrade path" guidance.
"""

import os

from apscheduler.schedulers.background import BackgroundScheduler

from agentic_market_intel.data import build_dataset, build_live_dataset
from agentic_market_intel.pipeline import run_pipeline

_scheduler: BackgroundScheduler | None = None


def _run_scheduled(company: str):
    news_key = os.environ.get("NEWS_API_KEY")
    df = build_live_dataset([company], news_key) if news_key else build_dataset()
    try:
        result = run_pipeline(df, company)
        print(f"[SCHEDULER] {company}: {result['status']} (run_id={result['run_id']})")
    except Exception as e:
        print(f"[SCHEDULER] {company}: run failed — {e}")


def start_scheduler(companies: list[str], interval_hours: int | None = None) -> BackgroundScheduler:
    global _scheduler
    if _scheduler is not None:
        return _scheduler

    interval_hours = interval_hours or int(os.environ.get("SCHEDULE_INTERVAL_HOURS", "6"))
    _scheduler = BackgroundScheduler()
    for company in companies:
        _scheduler.add_job(
            _run_scheduled, "interval", hours=interval_hours, args=[company], id=f"run_{company}",
        )
    _scheduler.start()
    print(f"[SCHEDULER] Tracking {companies} every {interval_hours}h")
    return _scheduler
