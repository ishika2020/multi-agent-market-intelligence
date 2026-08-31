"""LangChain tools the agents can choose to call.

Unlike the notebook (a bare global `df` filtered by hardcoded task instructions
that named the exact tool to use), these are built fresh per dataset via
`build_tools(df)` and closed over that dataframe — no module-level globals —
and every record carries a stable `evidence_id` so the Insight and Critic
agents can cite and verify claims against a specific source row instead of
just asserting things.
"""

import json
from collections import Counter

import pandas as pd
from langchain_core.tools import tool


def _subset_for(df: pd.DataFrame, company: str) -> pd.DataFrame:
    subset = df[df["company"].str.lower() == company.strip().lower()]
    return subset if not subset.empty else df


def build_tools(df: pd.DataFrame) -> dict:

    @tool
    def fetch_company_data(company: str) -> str:
        """Fetch raw news/market records for a company. Input: company name.
        Returns JSON with total_records, sources, date_range, and a list of
        records (each with an evidence_id, text, date, source, sentiment_label)."""
        subset = _subset_for(df, company)
        records = subset[["evidence_id", "text", "date", "source", "sentiment_label", "sentiment_avg"]].copy()
        records["date"] = records["date"].astype(str)
        return json.dumps({
            "company": company,
            "total_records": int(len(subset)),
            "sources": sorted(subset["source"].unique().tolist()),
            "date_range": f"{subset['date'].min()} to {subset['date'].max()}" if len(subset) else "n/a",
            "records": records.to_dict(orient="records"),
        }, indent=2)

    @tool
    def analyze_sentiment(company: str) -> str:
        """Compute aggregated sentiment stats for a company. Input: company name.
        Returns JSON with overall_score, label_percentages, and the evidence_ids
        of the most positive and most negative records."""
        subset = _subset_for(df, company)
        if subset.empty:
            return json.dumps({"error": "no data"})

        total = len(subset)
        counts = subset["sentiment_label"].value_counts()
        pct = {k: round(v / total * 100, 1) for k, v in counts.items()}
        for label in ["Positive", "Neutral", "Negative"]:
            pct.setdefault(label, 0.0)

        top_pos = subset.sort_values("sentiment_avg", ascending=False).head(2)
        top_neg = subset.sort_values("sentiment_avg", ascending=True).head(2)

        return json.dumps({
            "company": company,
            "overall_score": round(subset["sentiment_avg"].mean(), 4),
            "label_percentages": pct,
            "most_positive_evidence_ids": top_pos["evidence_id"].tolist(),
            "most_negative_evidence_ids": top_neg["evidence_id"].tolist(),
        }, indent=2)

    @tool
    def detect_trends(company: str) -> str:
        """Detect volume/sentiment trend direction and top keywords for a company.
        Input: company name. Returns JSON with sentiment_trend, top_keywords,
        and a risk_flag if negative coverage exceeds 35%."""
        subset = _subset_for(df, company).sort_values("date")
        if subset.empty:
            return json.dumps({"error": "no data"})

        n = len(subset)
        half = max(1, n // 2)
        first_half_avg = round(subset.iloc[:half]["sentiment_avg"].mean(), 4)
        second_half_avg = round(subset.iloc[half:]["sentiment_avg"].mean(), 4) if n > half else first_half_avg
        trend = "Improving" if second_half_avg > first_half_avg else (
            "Declining" if second_half_avg < first_half_avg else "Stable"
        )

        words = " ".join(subset["text_clean"].tolist()).lower().split()
        stopwords = {"the", "a", "an", "and", "to", "of", "in", "for", "on", "with", "is", "over", "amid", "faces"}
        keywords = Counter(w for w in words if len(w) > 3 and w not in stopwords)
        top_keywords = dict(keywords.most_common(5))

        neg_rate = round((subset["sentiment_label"] == "Negative").mean() * 100, 1)

        return json.dumps({
            "company": company,
            "sentiment_trend": trend,
            "first_half_avg": first_half_avg,
            "second_half_avg": second_half_avg,
            "top_keywords": top_keywords,
            "negative_rate_pct": neg_rate,
            "risk_flag": bool(neg_rate > 35.0),
        }, indent=2)

    @tool
    def lookup_evidence(evidence_ids: str) -> str:
        """Look up the original source text for one or more evidence_ids.
        Input: comma-separated evidence_ids, e.g. "EVID-0001,EVID-0003".
        Returns JSON list of {evidence_id, text, date, source} — use this to
        verify a claim actually traces back to real source material."""
        ids = [e.strip() for e in evidence_ids.split(",") if e.strip()]
        subset = df[df["evidence_id"].isin(ids)]
        records = subset[["evidence_id", "text", "date", "source"]].copy()
        records["date"] = records["date"].astype(str)
        found = set(records["evidence_id"])
        missing = [i for i in ids if i not in found]
        return json.dumps({"records": records.to_dict(orient="records"), "missing_ids": missing}, indent=2)

    return {
        "fetch": fetch_company_data,
        "sentiment": analyze_sentiment,
        "trend": detect_trends,
        "lookup_evidence": lookup_evidence,
    }
