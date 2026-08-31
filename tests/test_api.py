"""Smoke tests for the FastAPI service. Runs entirely in mock mode — CI has
no GROQ_API_KEY/NEWS_API_KEY, so no live API calls or cost are involved.
"""

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_market_intel.db")

from fastapi.testclient import TestClient

from agentic_market_intel.api import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_trigger_run_mock_mode():
    resp = client.post("/runs", json={"company": "Tesla"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["company"] == "Tesla"
    assert body["mode"] == "mock"
    assert body["status"] in {"approved", "forced_unverified"}
    assert "report" in body and len(body["report"]) > 0


def test_get_and_list_reports():
    run_resp = client.post("/runs", json={"company": "Apple"})
    report_id = run_resp.json()["report_id"]

    get_resp = client.get(f"/reports/{report_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["company"] == "Apple"

    list_resp = client.get("/reports", params={"company": "Apple"})
    assert list_resp.status_code == 200
    assert any(r["id"] == report_id for r in list_resp.json())


def test_missing_report_404():
    resp = client.get("/reports/999999")
    assert resp.status_code == 404


def test_empty_company_rejected():
    resp = client.post("/runs", json={"company": "  "})
    assert resp.status_code == 400
