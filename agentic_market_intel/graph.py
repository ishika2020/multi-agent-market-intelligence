"""The actual agentic core.

What makes this different from the notebook's CrewAI `Process.sequential`
chain:

1. Research and Analysis agents are ReAct loops (`create_react_agent`) that
   decide for themselves which tool(s) to call and when to stop — the task
   text below is a goal, not a scripted "call ToolX then ToolY" instruction.
2. The Insight agent must cite an `evidence_id` for every claim, and can
   call `lookup_evidence` itself to check its own reasoning before
   answering.
3. The Critic agent independently re-fetches the cited evidence and judges
   whether each claim actually holds up. Its verdict drives a real
   conditional edge: reject -> back to Insight (with the specific feedback)
   for one bounded retry, approve -> Report. This loop does not exist in
   the original notebook at all.
"""

import json

import pandas as pd
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

from agentic_market_intel.llm import invoke_with_retry
from agentic_market_intel.observability import log_agent_call
from agentic_market_intel.report import format_report
from agentic_market_intel.state import CriticVerdict, GraphState, InsightsOutput
from agentic_market_intel.tools import build_tools

MAX_REVISIONS = 1


def build_graph(df: pd.DataFrame, llm):
    tools = build_tools(df)

    research_agent = create_react_agent(
        llm,
        tools=[tools["fetch"]],
        prompt=(
            "You are a senior market research analyst. Given a company name, "
            "decide what data you need and use the available tool to fetch it. "
            "Summarize total records, date range, sources, and flag any data "
            "quality concerns (e.g. too few records, single-source bias, stale dates) "
            "in 2-4 sentences. Do not invent data you did not fetch."
        ),
    )

    analysis_agent = create_react_agent(
        llm,
        tools=[tools["sentiment"], tools["trend"], tools["market_data"]],
        prompt=(
            "You are a quantitative sentiment and trend analyst. Given a company name, "
            "decide which of the available tools you need (sentiment, trend, and/or "
            "market_data) to produce: overall sentiment score and distribution, trend "
            "direction, top keywords, any risk flag, and — if market_data is available "
            "for this company — whether the stock price trend agrees or diverges from "
            "the sentiment trend (a divergence is itself worth flagging). If market_data "
            "returns an error, note that price correlation was unavailable and move on; "
            "do not treat it as a data point. Summarize findings in 3-5 sentences, "
            "explicitly citing the numbers you found."
        ),
    )

    # Two-step design: let the ReAct loop reason and call tools freely (a forced
    # "must call the extraction tool now" step at the end of a tool-loop is
    # fragile — models sometimes wrap up with a prose/markdown answer instead),
    # then run a separate, focused structured-extraction call over its final
    # answer. Each step does one job, so each is more reliable.
    insight_agent = create_react_agent(
        llm,
        tools=[tools["fetch"], tools["lookup_evidence"]],
        prompt=(
            "You are a chief business intelligence strategist. You will be given research "
            "and analysis summaries as background context. Before writing any claims, call "
            "fetch_company_data yourself to get the real records and their evidence_id values "
            "for this company — do not assume an evidence_id exists just because it was "
            "mentioned in the background text. Then produce a list of specific business claims "
            "(opportunities, risks, or observations). Every claim MUST cite at least one "
            "evidence_id you actually retrieved from fetch_company_data, written exactly as "
            "EVID-XXXX. Use lookup_evidence if you need to double-check a specific record. "
            "Never fabricate an evidence_id. If you previously received critic feedback, fix "
            "exactly the claims it flagged. End with a plain-text list of the final claims and "
            "a one-paragraph recommendation."
        ),
    )
    insight_extractor = llm.with_structured_output(InsightsOutput)

    def research_node(state: GraphState) -> dict:
        with log_agent_call(state.get("run_id"), "research") as ctx:
            result = invoke_with_retry(research_agent.invoke, {
                "messages": [("user", f"Research company: {state['company']}")]
            })
            ctx["messages"] = result["messages"]
        return {"research_summary": result["messages"][-1].content}

    def analysis_node(state: GraphState) -> dict:
        with log_agent_call(state.get("run_id"), "analysis") as ctx:
            result = invoke_with_retry(analysis_agent.invoke, {
                "messages": [("user", f"Analyze sentiment and trends for: {state['company']}")]
            })
            ctx["messages"] = result["messages"]
        return {"analysis_summary": result["messages"][-1].content}

    def insight_node(state: GraphState) -> dict:
        prompt = (
            f"Company: {state['company']}\n\n"
            f"Research findings:\n{state['research_summary']}\n\n"
            f"Analysis findings:\n{state['analysis_summary']}\n"
        )
        if state.get("critic_verdict") is not None:
            cv = state["critic_verdict"]
            prompt += (
                f"\nA critic reviewed your previous claims and REJECTED them:\n"
                f"Reasoning: {cv.reasoning}\n"
                f"Unsupported claims: {cv.unsupported_claims}\n"
                f"Revise your claims to fix these specific issues.\n"
            )

        with log_agent_call(state.get("run_id"), "insight") as ctx:
            result = invoke_with_retry(insight_agent.invoke, {"messages": [("user", prompt)]})
            raw_answer = result["messages"][-1].content

            insights: InsightsOutput = invoke_with_retry(
                insight_extractor.invoke,
                "Extract the business claims from the analysis below into the required schema. "
                "Preserve every evidence_id exactly as written (format EVID-XXXX). Do not add or "
                "drop claims.\n\n"
                f"{raw_answer}"
            )
            ctx["messages"] = result["messages"]
        return {"insights": insights}

    def critic_node(state: GraphState) -> dict:
        insights: InsightsOutput = state["insights"]
        all_ids = sorted({eid for c in insights.claims for eid in c.evidence_ids})
        evidence_json = tools["lookup_evidence"].invoke({"evidence_ids": ",".join(all_ids)}) if all_ids else "{}"

        claims_text = "\n".join(
            f"- [{c.claim_type}] {c.claim} (cites: {', '.join(c.evidence_ids) or 'NONE'})"
            for c in insights.claims
        )

        critic_llm = llm.with_structured_output(CriticVerdict)
        with log_agent_call(state.get("run_id"), "critic"):
            verdict: CriticVerdict = invoke_with_retry(
                critic_llm.invoke,
                "You are a skeptical QA reviewer. Verify each claim below against the actual "
                "evidence text. Reject (approved=false) if: a claim cites no evidence_id, cites "
                "an evidence_id that is missing from the lookup, or the evidence text does not "
                "actually support the claim. Be strict — this report will be read unattended.\n\n"
                f"CLAIMS:\n{claims_text}\n\n"
                f"EVIDENCE LOOKUP RESULT:\n{evidence_json}"
            )
        return {"critic_verdict": verdict, "revise_count": state.get("revise_count", 0) + (0 if verdict.approved else 1)}

    def route_after_critic(state: GraphState) -> str:
        verdict: CriticVerdict = state["critic_verdict"]
        if verdict.approved or state.get("revise_count", 0) > MAX_REVISIONS:
            return "report"
        return "insight"

    def report_node(state: GraphState) -> dict:
        report_text = format_report(state)
        trend = json.loads(tools["trend"].invoke({"company": state["company"]}))
        return {"final_report": report_text, "alert": bool(trend.get("risk_flag"))}

    graph = StateGraph(GraphState)
    graph.add_node("research", research_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("insight", insight_node)
    graph.add_node("critic", critic_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("research")
    graph.add_edge("research", "analysis")
    graph.add_edge("analysis", "insight")
    graph.add_edge("insight", "critic")
    graph.add_conditional_edges("critic", route_after_critic, {"insight": "insight", "report": "report"})
    graph.add_edge("report", END)

    return graph.compile()


def run_for_company(df: pd.DataFrame, llm, company: str, run_id: int | None = None) -> GraphState:
    app = build_graph(df, llm)
    final_state = app.invoke({"company": company, "revise_count": 0, "run_id": run_id})
    return final_state
