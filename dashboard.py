"""Streamlit dashboard — reads from the FastAPI service, doesn't touch the
pipeline or database directly, so it stays a thin client over the same API
any other consumer would use.

Run:
    uvicorn agentic_market_intel.api:app --reload   # in one terminal
    streamlit run dashboard.py                      # in another
"""

import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Market Intelligence", page_icon="📊", layout="wide")
st.title("📊 Multi-Agent Market Intelligence")

with st.sidebar:
    st.header("Run a new analysis")
    company_input = st.text_input("Company", value="Tesla")
    run_clicked = st.button("🚀 Run Analysis", use_container_width=True)

    st.caption(f"API: {API_BASE_URL}")
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=3)
        st.success("API reachable") if health.ok else st.error("API unhealthy")
    except requests.RequestException:
        st.error("API unreachable — is uvicorn running?")

if run_clicked and company_input.strip():
    with st.spinner(f"Running the agentic pipeline for {company_input}... this can take 30-90s in live mode."):
        try:
            resp = requests.post(f"{API_BASE_URL}/runs", json={"company": company_input.strip()}, timeout=300)
            resp.raise_for_status()
            result = resp.json()
        except requests.RequestException as e:
            st.error(f"Run failed: {e}")
            result = None

    if result:
        status = result["status"]
        badge = {"approved": "✅ APPROVED", "forced_unverified": "⚠️ FORCED THROUGH UNVERIFIED"}.get(status, status)
        st.subheader(f"{result['company']} — {badge}")
        cols = st.columns(4)
        cols[0].metric("Mode", result["mode"])
        cols[1].metric("Revisions used", result["revise_count"])
        cols[2].metric("Duration (s)", result["duration_seconds"])
        cols[3].metric("Alert", "🔴 Yes" if result["alert"] else "🟢 No")

        st.text_area("Report", result["report"], height=400)

        with st.expander("Per-agent breakdown"):
            try:
                run_detail = requests.get(f"{API_BASE_URL}/runs/{result['run_id']}", timeout=10).json()
                calls_df = pd.DataFrame(run_detail["agent_calls"])
                st.dataframe(calls_df, use_container_width=True)
            except requests.RequestException as e:
                st.warning(f"Could not load per-agent breakdown: {e}")

st.divider()
st.header("Past reports")

filter_company = st.text_input("Filter by company (optional)", value="")
try:
    params = {"company": filter_company} if filter_company.strip() else {}
    reports = requests.get(f"{API_BASE_URL}/reports", params=params, timeout=10).json()
except requests.RequestException as e:
    reports = []
    st.error(f"Could not load reports: {e}")

if not reports:
    st.info("No reports yet — run an analysis above.")
else:
    for r in reports:
        with st.expander(f"#{r['id']} — {r['company']} — {r['created_at']}"):
            try:
                detail = requests.get(f"{API_BASE_URL}/reports/{r['id']}", timeout=10).json()
                st.text_area("Report text", detail["report_text"], height=300, key=f"report_{r['id']}")
            except requests.RequestException as e:
                st.warning(f"Could not load report {r['id']}: {e}")
