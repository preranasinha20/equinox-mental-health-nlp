# ingest.py
import os
import json
import time
from datetime import datetime
import pandas as pd
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

UPLOAD_DIR = "uploads"
JSON_PATH = os.path.join(UPLOAD_DIR, "reddit_posts.json")
CSV_PATH = os.path.join(UPLOAD_DIR, "reddit_posts.csv")

sia = SentimentIntensityAnalyzer()

# --- Ensure upload dir exists ---
def _ensure_upload_dir():
    os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Helper: combine and clean post data ---
def _posts_to_df(posts):
    df = pd.DataFrame(posts)
    if df.empty:
        return df

    # Convert timestamps and combine content
    df["created_dt_utc"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce", utc=True)
    df["created_date_utc"] = df["created_dt_utc"].dt.date
    df["content"] = (
        df["title"].fillna("").astype(str).str.strip() + " " +
        df["text"].fillna("").astype(str).str.strip()
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    # Sentiment analysis
    df["vader_compound"] = df["content"].astype(str).map(lambda t: sia.polarity_scores(t)["compound"])
    return df

# --- Main scraper (deeper and wider like your Phase-1 script) ---
def scrape_subreddits(reddit, subreddits=None, posts_per_sub=1000):
    """
    Scrape multiple subreddits deeply (top + hot + new + rising) for rich dataset.
    Returns a DataFrame and caches to uploads/.
    """
    _ensure_upload_dir()

    if subreddits is None:
        subreddits = [
            "mentalhealth", "selfimprovement", "college",
            "happiness", "Anxiety", "Depression", "relationships"
        ]

    all_posts = []

    for sub in subreddits:
        print(f"📡 Scraping subreddit: r/{sub} ...")
        sr = reddit.subreddit(sub)

        # TOP posts (weekly or all-time)
        for p in sr.top(limit=posts_per_sub // 4, time_filter="all"):
            all_posts.append(_extract_post(p, sub))

        # HOT
        for p in sr.hot(limit=posts_per_sub // 4):
            all_posts.append(_extract_post(p, sub))

        # NEW
        for p in sr.new(limit=posts_per_sub // 4):
            all_posts.append(_extract_post(p, sub))

        # RISING (if available)
        try:
            for p in sr.rising(limit=posts_per_sub // 4):
                all_posts.append(_extract_post(p, sub))
        except Exception:
            pass

        time.sleep(1)  # Avoid Reddit rate-limit

    print(f"✅ Collected total {len(all_posts)} posts.")
    df = _posts_to_df(all_posts)

    if not df.empty:
        df = df.drop_duplicates(subset=["title", "url", "author", "created_utc"]).reset_index(drop=True)

    # Cache JSON + CSV
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_posts, f, ensure_ascii=False, indent=2)
    if not df.empty:
        df.to_csv(CSV_PATH, index=False)

    return df, JSON_PATH, CSV_PATH


# --- Helper: extract safe fields ---
def _extract_post(p, sub):
    try:
        return {
            "subreddit": sub,
            "title": getattr(p, "title", ""),
            "score": getattr(p, "score", 0),
            "author": str(getattr(p, "author", "deleted")),
            "num_comments": getattr(p, "num_comments", 0),
            "created_utc": getattr(p, "created_utc", 0),
            "url": getattr(p, "url", ""),
            "text": getattr(p, "selftext", ""),
        }
    except Exception as e:
        return {"subreddit": sub, "error": str(e)}

# --- Cached data loader ---
def load_cached_df():
    """Load last scraped Reddit data from uploads."""
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _posts_to_df(data)
    return pd.DataFrame()
