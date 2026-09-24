"""Graph state + structured output schemas.

The Critic verdict and the conditional edge it drives are the actual
"agentic" part of this system: a real branch decided at run-time by an LLM
judgment call, not a hardcoded `task_a -> task_b -> task_c` chain.
"""

from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field


class Claim(BaseModel):
    claim: str = Field(description="One specific, checkable business claim.")
    claim_type: Literal["opportunity", "risk", "observation"]
    evidence_ids: list[str] = Field(description="evidence_id values this claim is grounded in. Must not be empty.")


class InsightsOutput(BaseModel):
    claims: list[Claim]
    recommendation: str = Field(description="One-paragraph strategic recommendation.")


class CriticVerdict(BaseModel):
    approved: bool = Field(description="True only if every claim is supported by its cited evidence and not stale/contradicted.")
    unsupported_claims: list[str] = Field(default_factory=list, description="Claim text of any claim that failed verification.")
    reasoning: str = Field(description="Brief explanation of the verdict, referencing specific claims.")


class GraphState(TypedDict, total=False):
    company: str
    run_id: int
    research_summary: str
    analysis_summary: str
    sentiment_json: str
    trend_json: str
    insights: InsightsOutput
    critic_verdict: CriticVerdict
    revise_count: int
    final_report: str
    alert: bool
