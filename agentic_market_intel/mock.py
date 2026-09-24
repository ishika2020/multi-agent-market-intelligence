"""Deterministic fallback pipeline — no LLM calls.

This exists for the same reason the notebook's mock mode did: graceful
degradation when the live LLM dependency is down or unconfigured, and cheap
CI smoke-testing. The difference from the notebook is that it's an explicit
fallback with its own (simplified, rule-based) QA check, not the path that
silently ends up carrying most of the demo.
"""

import json

import pandas as pd

from agentic_market_intel.observability import log_agent_call
from agentic_market_intel.report import format_report
from agentic_market_intel.state import Claim, CriticVerdict, GraphState, InsightsOutput
from agentic_market_intel.tools import build_tools

MAX_REVISIONS = 1


def run_mock_pipeline(df: pd.DataFrame, company: str, run_id: int | None = None) -> GraphState:
    tools = build_tools(df)

    with log_agent_call(run_id, "research"):
        research_raw = json.loads(tools["fetch"].invoke({"company": company}))
        research_summary = (
            f"[MOCK] Found {research_raw['total_records']} records for {company} "
            f"from {len(research_raw['sources'])} source(s), spanning {research_raw['date_range']}."
        )

    with log_agent_call(run_id, "analysis"):
        sentiment = json.loads(tools["sentiment"].invoke({"company": company}))
        trend = json.loads(tools["trend"].invoke({"company": company}))
        market = json.loads(tools["market_data"].invoke({"company": company}))
        market_note = (
            f" Price trend: {market['price_trend']} ({market['pct_change']}% over {market['period']})."
            if "error" not in market else ""
        )
        analysis_summary = (
            f"[MOCK] Overall sentiment score {sentiment.get('overall_score')} "
            f"({sentiment.get('label_percentages')}). Trend: {trend.get('sentiment_trend')}. "
            f"Top keywords: {list(trend.get('top_keywords', {}).keys())}. "
            f"Negative rate: {trend.get('negative_rate_pct')}% (risk_flag={trend.get('risk_flag')})."
            f"{market_note}"
        )

    state: GraphState = {
        "company": company,
        "research_summary": research_summary,
        "analysis_summary": analysis_summary,
        "revise_count": 0,
    }

    revise_count = 0
    while True:
        with log_agent_call(run_id, "insight"):
            insights = _mock_insights(sentiment, trend)
        with log_agent_call(run_id, "critic"):
            verdict = _mock_critic(tools, insights)
        if not verdict.approved:
            revise_count += 1
        if verdict.approved or revise_count > MAX_REVISIONS:
            break

    state["insights"] = insights
    state["critic_verdict"] = verdict
    state["revise_count"] = revise_count
    state["final_report"] = format_report(state)
    state["alert"] = bool(trend.get("risk_flag"))
    return state


def _mock_insights(sentiment: dict, trend: dict) -> InsightsOutput:
    claims = []

    pos_ids = sentiment.get("most_positive_evidence_ids", [])
    if pos_ids:
        claims.append(Claim(
            claim="Recent coverage includes clearly positive developments that could support near-term sentiment recovery.",
            claim_type="opportunity",
            evidence_ids=pos_ids,
        ))

    neg_ids = sentiment.get("most_negative_evidence_ids", [])
    if neg_ids:
        claims.append(Claim(
            claim="Recent negative coverage represents a reputational or regulatory risk worth monitoring.",
            claim_type="risk",
            evidence_ids=neg_ids,
        ))

    claims.append(Claim(
        claim=f"Sentiment trend is currently {trend.get('sentiment_trend', 'Stable')}.",
        claim_type="observation",
        evidence_ids=list(set(pos_ids + neg_ids)) or ["EVID-0000"],
    ))

    recommendation = (
        "Monitor the flagged risk items closely; if negative rate stays above 35%, "
        "escalate to a full manual review before the next scheduled run."
        if trend.get("risk_flag")
        else "No immediate action required; continue routine monitoring."
    )

    return InsightsOutput(claims=claims, recommendation=recommendation)


def _mock_critic(tools: dict, insights: InsightsOutput) -> CriticVerdict:
    all_ids = sorted({eid for c in insights.claims for eid in c.evidence_ids})
    lookup = json.loads(tools["lookup_evidence"].invoke({"evidence_ids": ",".join(all_ids)})) if all_ids else {"missing_ids": []}
    missing = lookup.get("missing_ids", [])

    if missing:
        return CriticVerdict(
            approved=False,
            unsupported_claims=[c.claim for c in insights.claims if any(m in c.evidence_ids for m in missing)],
            reasoning=f"[MOCK] {len(missing)} cited evidence_id(s) do not exist in the dataset: {missing}",
        )

    return CriticVerdict(
        approved=True,
        unsupported_claims=[],
        reasoning="[MOCK] All cited evidence_ids resolved to real records.",
    )
