import os
import time


def invoke_with_retry(fn, *args, max_retries: int = 4, base_delay: float = 8.0, **kwargs):
    """Retry an LLM call with backoff on Groq rate limits.

    The free tier's tokens-per-minute cap (8000 TPM at time of writing) is
    routinely hit by a 5-node agent graph, and Groq's error message tells you
    exactly how long to wait — this is the cost/rate guardrail called out in
    the design doc, not just a workaround for one flaky run.
    """
    from groq import RateLimitError

    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except RateLimitError:
            if attempt == max_retries:
                raise
            wait = base_delay * (attempt + 1)
            print(f"[RATE LIMIT] Groq TPM limit hit — waiting {wait:.0f}s before retry ({attempt + 1}/{max_retries})...")
            time.sleep(wait)


def get_llm(temperature: float = 0.2):
    """Returns a ChatGroq instance, or None if no API key is configured.

    None is the signal main.py uses to fall back to the deterministic mock
    pipeline (agentic_market_intel/mock.py) — the safety net for when the
    live LLM dependency is unavailable, not the primary demo path.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or not api_key.startswith("gsk_"):
        return None

    from langchain_groq import ChatGroq

    return ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=api_key,
        temperature=temperature,
    )


def is_live() -> bool:
    key = os.environ.get("GROQ_API_KEY", "")
    return key.startswith("gsk_")
