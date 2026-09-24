"""Per-agent latency/token/success logging — the granularity a run-level-only
`Run` row can't give you. This is the "how would you monitor this in prod"
answer: every node execution becomes one queryable row.
"""

import time
from contextlib import contextmanager

from agentic_market_intel.db import AgentCall, SessionLocal


def _sum_tokens(messages) -> int | None:
    total = 0
    found = False
    for m in messages or []:
        usage = getattr(m, "usage_metadata", None)
        if usage:
            total += usage.get("total_tokens", 0) or 0
            found = True
    return total if found else None


@contextmanager
def log_agent_call(run_id: int | None, node_name: str):
    """Usage:

        with log_agent_call(state.get("run_id"), "research") as ctx:
            result = research_agent.invoke(...)
            ctx["messages"] = result["messages"]  # optional, enables token counting
    """
    start = time.time()
    ctx: dict = {}
    error = None
    try:
        yield ctx
    except Exception as e:
        error = str(e)
        raise
    finally:
        if run_id is not None:
            session = SessionLocal()
            try:
                session.add(AgentCall(
                    run_id=run_id,
                    node_name=node_name,
                    latency_seconds=round(time.time() - start, 2),
                    tokens_total=_sum_tokens(ctx.get("messages")),
                    success=error is None,
                    error=error,
                ))
                session.commit()
            finally:
                session.close()
