"""CLI entrypoint.

Usage:
    python main.py --company Tesla
    python main.py --companies Tesla,Apple
    python main.py --company Tesla --csv my_data.csv
"""

import argparse
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

from agentic_market_intel.data import build_dataset, build_live_dataset
from agentic_market_intel.llm import is_live
from agentic_market_intel.pipeline import run_pipeline

if sys.platform == "win32":
    # LLM output can contain Unicode punctuation (narrow no-break spaces, em
    # dashes, etc.) that the default Windows console codepage can't encode.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Agentic market intelligence pipeline")
    parser.add_argument("--company", type=str, help="Single company to analyze")
    parser.add_argument("--companies", type=str, help="Comma-separated list of companies")
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV path (falls back to dummy dataset)")
    parser.add_argument("--out", type=str, default="./market_reports", help="Output directory for reports")
    args = parser.parse_args()

    if args.companies:
        companies = [c.strip() for c in args.companies.split(",") if c.strip()]
    elif args.company:
        companies = [args.company]
    else:
        companies = ["Tesla", "Apple"]

    news_api_key = os.environ.get("NEWS_API_KEY")
    if args.csv:
        df = build_dataset(args.csv)
    elif news_api_key:
        print(f"NEWS_API_KEY found - fetching live news for: {companies}")
        df = build_live_dataset(companies, news_api_key)
    else:
        df = build_dataset()

    if is_live():
        print(f"Live LLM mode (Groq). Analyzing: {companies}")
    else:
        print("No GROQ_API_KEY found - running in MOCK mode (deterministic, no LLM calls).")
        print(f"Analyzing: {companies}")

    os.makedirs(args.out, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for company in companies:
        print(f"\n{'=' * 72}\nRunning pipeline for: {company}\n{'=' * 72}")
        result = run_pipeline(df, company)
        print(result["report"])
        print(f"\n[DB] run_id={result['run_id']} report_id={result['report_id']} status={result['status']}")
        if result.get("alert"):
            print(f"[ALERT] {company} crossed the negative-coverage threshold - would trigger Slack/email alert.")

        path = os.path.join(args.out, f"{company.lower()}_{timestamp}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(result["report"])
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
