"""Data loading, cleaning, and sentiment pre-scoring.

Kept close to the original notebook's logic (TextBlob + VADER ensemble) since
that part of the prototype was fine — the problem was never the data prep,
it was the agent orchestration on top of it.
"""

import re
from datetime import date, timedelta

import pandas as pd
import requests
import yfinance as yf

COMPANY_TICKERS = {
    "tesla": "TSLA", "apple": "AAPL", "amazon": "AMZN", "google": "GOOGL",
    "alphabet": "GOOGL", "microsoft": "MSFT", "meta": "META", "facebook": "META",
    "nvidia": "NVDA", "netflix": "NFLX", "intel": "INTC", "amd": "AMD",
}
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_TEXT_COL_CANDIDATES = ["text", "review", "content", "headline", "description", "article"]


def load_dummy_data() -> pd.DataFrame:
    """Built-in fallback dataset — used only when no CSV/live source is provided."""
    today = date.today()
    rows = [
        {"text": "Tesla delivers record number of vehicles in Q3, beating analyst estimates.",
         "company": "Tesla", "source": "Reuters", "date": today - timedelta(days=6)},
        {"text": "Tesla recall notice issued for over 2 million vehicles due to autopilot safety concerns.",
         "company": "Tesla", "source": "Bloomberg", "date": today - timedelta(days=5)},
        {"text": "Tesla Cybertruck deliveries ramp up faster than expected, investors optimistic.",
         "company": "Tesla", "source": "CNBC", "date": today - timedelta(days=4)},
        {"text": "Tesla FSD beta receives positive reviews from early testers in urban driving.",
         "company": "Tesla", "source": "TechCrunch", "date": today - timedelta(days=3)},
        {"text": "Tesla stock dips amid concerns over Musk's political controversies.",
         "company": "Tesla", "source": "WSJ", "date": today - timedelta(days=2)},
        {"text": "Tesla announces India Gigafactory expansion, targeting fast-growing EV market.",
         "company": "Tesla", "source": "Economic Times", "date": today - timedelta(days=1)},
        {"text": "Tesla faces regulatory scrutiny over autopilot crash investigations.",
         "company": "Tesla", "source": "Reuters", "date": today},
        {"text": "Tesla investor sentiment mixed after mixed earnings call commentary.",
         "company": "Tesla", "source": "Bloomberg", "date": today},
        {"text": "Apple unveils new iPhone with improved battery life, reviews are largely positive.",
         "company": "Apple", "source": "The Verge", "date": today - timedelta(days=3)},
        {"text": "Apple faces antitrust lawsuit in EU over App Store practices.",
         "company": "Apple", "source": "Reuters", "date": today - timedelta(days=2)},
        {"text": "Apple services revenue hits all-time high, offsetting slower hardware sales.",
         "company": "Apple", "source": "CNBC", "date": today - timedelta(days=1)},
        {"text": "Apple supplier reports production delays ahead of holiday season.",
         "company": "Apple", "source": "Bloomberg", "date": today},
    ]
    return pd.DataFrame(rows)


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize columns, clean text, drop nulls."""
    df = df.copy()

    for col in _TEXT_COL_CANDIDATES:
        if col in df.columns and col != "text":
            df.rename(columns={col: "text"}, inplace=True)
            break

    if "text" not in df.columns:
        raise ValueError("Dataset must have a 'text' column (or: review, content, headline, description, article)")

    if "company" not in df.columns:
        df["company"] = "Unknown"
    if "date" not in df.columns:
        df["date"] = pd.Timestamp.today().date()
    if "source" not in df.columns:
        df["source"] = "Unknown"

    df["text"] = df["text"].astype(str).str.strip()
    df["text_clean"] = df["text"].apply(lambda t: re.sub(r"\s+", " ", re.sub(r"[^\w\s.,!?'-]", "", t)))
    df["word_count"] = df["text_clean"].apply(lambda t: len(t.split()))

    df.dropna(subset=["text"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Stable per-row evidence ID — every downstream claim traces back to one of these.
    df["evidence_id"] = [f"EVID-{i:04d}" for i in df.index]

    return df


def compute_sentiment_scores(df: pd.DataFrame) -> pd.DataFrame:
    """TextBlob polarity + VADER compound, ensembled into a single label."""
    df = df.copy()
    vader = SentimentIntensityAnalyzer()

    tb_scores, vd_scores, labels = [], [], []
    for text in df["text_clean"]:
        tb = round(TextBlob(text).sentiment.polarity, 4)
        vd = round(vader.polarity_scores(text)["compound"], 4)
        avg = round((tb + vd) / 2, 4)

        if avg >= 0.05:
            label = "Positive"
        elif avg <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        tb_scores.append(tb)
        vd_scores.append(vd)
        labels.append(label)

    df["sentiment_textblob"] = tb_scores
    df["sentiment_vader"] = vd_scores
    df["sentiment_avg"] = [round((a + b) / 2, 4) for a, b in zip(tb_scores, vd_scores)]
    df["sentiment_label"] = labels
    return df


def build_dataset(csv_path: str | None = None) -> pd.DataFrame:
    raw = load_csv(csv_path) if csv_path else load_dummy_data()
    df = preprocess_dataframe(raw)
    df = compute_sentiment_scores(df)
    return df


def fetch_news(company: str, api_key: str, page_size: int = 30) -> pd.DataFrame:
    """Fetch recent articles for a company from NewsAPI.org (free tier: 100 req/day)."""
    resp = requests.get(
        "https://newsapi.org/v2/everything",
        params={
            "q": company,
            "apiKey": api_key,
            "pageSize": page_size,
            "sortBy": "publishedAt",
            "language": "en",
        },
        timeout=15,
    )
    resp.raise_for_status()
    articles = resp.json().get("articles", [])

    rows = []
    for a in articles:
        text = a.get("description") or a.get("title")
        if not text:
            continue
        published = (a.get("publishedAt") or "")[:10] or str(date.today())
        rows.append({
            "text": text,
            "company": company,
            "date": published,
            "source": (a.get("source") or {}).get("name", "Unknown"),
        })
    return pd.DataFrame(rows)


def build_live_dataset(companies: list[str], news_api_key: str | None) -> pd.DataFrame:
    """Build a dataset from real news, per company, falling back to the dummy
    dataset for any company where the live fetch fails or returns nothing —
    graceful degradation instead of a hard crash mid-run."""
    frames = []
    dummy = load_dummy_data()

    for company in companies:
        company_df = pd.DataFrame()
        if news_api_key:
            try:
                company_df = fetch_news(company, news_api_key)
            except Exception as e:
                print(f"[WARN] NewsAPI fetch failed for '{company}' ({e}). Falling back to dummy data.")

        if company_df.empty:
            fallback = dummy[dummy["company"].str.lower() == company.lower()]
            if fallback.empty:
                print(f"[WARN] No live data and no dummy data available for '{company}' — skipping.")
                continue
            company_df = fallback

        frames.append(company_df)

    raw = pd.concat(frames, ignore_index=True) if frames else dummy
    df = preprocess_dataframe(raw)
    df = compute_sentiment_scores(df)
    return df


def fetch_price_trend(company: str, period: str = "1mo") -> dict:
    """Price/volume trend for a company via yfinance, to correlate against
    news sentiment. Free, no API key required. Returns an error dict (not an
    exception) for unmapped companies or network failures, so the calling
    tool can hand that back to the agent as data rather than crashing."""
    ticker_symbol = COMPANY_TICKERS.get(company.strip().lower())
    if not ticker_symbol:
        return {"error": f"No ticker mapping for '{company}'. Known: {sorted(COMPANY_TICKERS)}"}

    try:
        hist = yf.Ticker(ticker_symbol).history(period=period)
    except Exception as e:
        return {"error": f"yfinance fetch failed for {ticker_symbol}: {e}"}

    hist = hist.dropna(subset=["Close"])
    if hist.empty:
        return {"error": f"No usable price data returned for {ticker_symbol}"}

    start_price = float(hist["Close"].iloc[0])
    end_price = float(hist["Close"].iloc[-1])
    pct_change = round((end_price - start_price) / start_price * 100, 2)

    return {
        "ticker": ticker_symbol,
        "period": period,
        "start_price": round(start_price, 2),
        "end_price": round(end_price, 2),
        "pct_change": pct_change,
        "avg_volume": int(hist["Volume"].mean()),
        "price_trend": "Up" if pct_change > 1 else ("Down" if pct_change < -1 else "Flat"),
    }
