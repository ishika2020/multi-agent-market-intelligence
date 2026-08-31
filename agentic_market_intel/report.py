"""Formats the final report — including a QA transparency section.

The QA section is the point: a reader (or another system) can see whether
the Critic actually approved these claims, or whether they were forced
through after the bounded retry budget ran out. That distinction doesn't
exist anywhere in the original notebook's output.
"""

from datetime import datetime

from agentic_market_intel.state import GraphState


def format_report(state: GraphState) -> str:
    company = state["company"]
    insights = state["insights"]
    verdict = state["critic_verdict"]
    revise_count = state.get("revise_count", 0)

    lines = []
    lines.append(f"MARKET INTELLIGENCE REPORT: {company}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 72)
    lines.append("")
    lines.append("RESEARCH")
    lines.append(state.get("research_summary", "n/a"))
    lines.append("")
    lines.append("ANALYSIS")
    lines.append(state.get("analysis_summary", "n/a"))
    lines.append("")
    lines.append("KEY CLAIMS")
    for c in insights.claims:
        lines.append(f"  [{c.claim_type.upper()}] {c.claim}")
        lines.append(f"      evidence: {', '.join(c.evidence_ids)}")
    lines.append("")
    lines.append("RECOMMENDATION")
    lines.append(insights.recommendation)
    lines.append("")
    lines.append("QA / CRITIC VERDICT")
    if verdict.approved:
        lines.append(f"  Status: APPROVED (revisions used: {revise_count})")
    else:
        lines.append(f"  Status: FORCED THROUGH UNVERIFIED after {revise_count} revision(s) — treat with caution")
        lines.append(f"  Unsupported claims flagged: {verdict.unsupported_claims}")
    lines.append(f"  Reasoning: {verdict.reasoning}")

    return "\n".join(lines)
