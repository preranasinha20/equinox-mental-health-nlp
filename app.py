# app.py
import re
from datetime import datetime, timedelta
from collections import defaultdict, Counter

from flask import Flask, render_template, jsonify, request, redirect, url_for
import praw
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Local modules
from ingest import scrape_subreddits, load_cached_df
from aura import analyze_aura
from nlp_bert import analyze_emotions, emotion_analyzer  # reuse loaded HF pipeline
from personality import analyze_big5  # Big Five heuristic

# --- CONFIG ---
REDDIT_CLIENT_ID = "M1I1jvc74xBb2xr-QuK2zQ"
REDDIT_CLIENT_SECRET = "qSFZqF4lbrk5kZ9q3UP44n0RMBHrig"
REDDIT_USER_AGENT = "wellness-tracker-demo"

# --- APP INIT ---
app = Flask(__name__)
analyzer = SentimentIntensityAnalyzer()
user_posts = {}  # in-memory cache per username for the user dashboard

# --- Reddit Init ---
def init_reddit():
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )

# --- Text Preprocessing ---
def preprocess_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- Fetch Reddit Posts for a Specific User (dashboard + live patient fallback) ---
def fetch_user_submissions(username, limit=100):
    reddit = init_reddit()
    ruser = reddit.redditor(username)
    posts = []
    for s in ruser.submissions.new(limit=limit):
        text = (getattr(s, "title", "") or "") + " " + (getattr(s, "selftext", "") or "")
        clean_text = preprocess_text(text)
        created_utc = getattr(s, "created_utc", 0) or 0
        posts.append({
            "id": getattr(s, "id", ""),
            "title": getattr(s, "title", "") or "",
            "created": datetime.utcfromtimestamp(created_utc),
            "text": clean_text,
        })
    user_posts[username] = posts
    return posts

# --- Sentiment Analysis (VADER) ---
def analyze_posts(posts):
    results = []
    for p in posts:
        scores = analyzer.polarity_scores(p["text"])
        results.append({
            "id": p["id"],
            "title": p["title"],
            "created": p["created"],
            "compound": scores["compound"],
            "pos": scores["pos"],
            "neg": scores["neg"],
            "neu": scores["neu"],
        })
    return results

# --- Aggregate Daily Mood for dashboard ---
def get_daily_mood(username, days=60):
    posts = user_posts.get(username, [])
    if not posts:
        return []
    analyzed = analyze_posts(posts)
    cutoff = datetime.utcnow() - timedelta(days=days)
    buckets = {}
    for a in analyzed:
        if a["created"] < cutoff:
            continue
        d = a["created"].date().isoformat()
        buckets.setdefault(d, []).append(a["compound"])
    trend = [
        {"date": d, "avg_compound": sum(vals)/len(vals)}
        for d, vals in sorted(buckets.items())
    ]
    return trend

# =========================
# Helpers for Therapist Insights
# =========================

EMOTION_KEYS = ["joy", "love", "surprise", "anger", "sadness", "fear", "disgust", "neutral"]

def _analyze_emotion_trend(texts, dates):
    """
    Run BERT emotion classifier per post, aggregate by day, and
    return (labels, series_dict) for a stacked area chart.
    Values are normalized per day to sum to 1.0.
    """
    # 1) Score each post
    per_post = []
    for t in texts:
        t = (t or "").strip()
        if not t:
            per_post.append({})
            continue
        res = emotion_analyzer(t[:512])[0]  # list of dicts
        scores = {d["label"].lower(): float(d["score"]) for d in res}
        per_post.append(scores)

    # 2) Bucket by date
    buckets = defaultdict(list)  # date -> list of score dicts
    for d, sc in zip(dates, per_post):
        buckets[d].append(sc)

    # 3) Aggregate per day
    day_labels = sorted(buckets.keys())
    series = {k: [] for k in EMOTION_KEYS}
    for day in day_labels:
        agg = Counter()
        for sc in buckets[day]:
            agg.update(sc)
        total = sum(agg.values())
        for k in EMOTION_KEYS:
            val = float(agg.get(k, 0.0))
            series[k].append(round(val / total, 4) if total > 0 else 0.0)

    # prune empty series (all zeros)
    series = {k: v for k, v in series.items() if any(x > 0 for x in v)}
    return day_labels, series


def _assess_risks(texts, vader_avg, emo_daily_series):
    """
    Lightweight, explainable heuristics for clinical risk flags.
    Returns categorical levels: 'low' | 'moderate' | 'high'.
    Not a diagnosis.
    """
    corpus = " ".join(texts).lower()

    # keyword hints (non-exhaustive)
    kw_suicide = ["suicide", "kill myself", "end my life", "self harm", "self-harm", "cutting", "no reason to live"]
    kw_anxiety = ["panic", "worry", "anxious", "anxiety", "restless", "overthinking"]
    kw_ptsd    = ["flashback", "nightmare", "trauma", "abuse", "assault", "intrusive", "hypervigilant"]
    kw_schizo  = ["voices", "hallucination", "paranoid", "delusion", "thought broadcasting", "schizophrenia"]

    def score_keywords(kws):
        return sum(1 for k in kws if k in corpus)

    # emotion proportions - last day (if available)
    emo_keys = list(emo_daily_series.keys())
    last_vals = {k: (emo_daily_series[k][-1] if emo_daily_series.get(k) else 0.0) for k in emo_keys}
    sadness = last_vals.get("sadness", 0.0)
    fear    = last_vals.get("fear", 0.0)
    anger   = last_vals.get("anger", 0.0)
    disgust = last_vals.get("disgust", 0.0)
    surprise= last_vals.get("surprise", 0.0)
    neutral = last_vals.get("neutral", 0.0)

    def bucket(x):
        if x >= 0.66: return "high"
        if x >= 0.33: return "moderate"
        return "low"

    # Combine keyword + emotion + sentiment hints
    dep_score = 0.45*sadness + 0.25*(1-neutral) + 0.30*max(0.0, -vader_avg) + 0.05*score_keywords(["worthless","fatigue","insomnia","guilty"])
    anx_score = 0.40*fear + 0.20*surprise + 0.25*score_keywords(kw_anxiety) + 0.15*max(0.0, -vader_avg)
    ptsd_sc   = 0.45*score_keywords(kw_ptsd) + 0.30*fear + 0.25*anger
    sch_sc    = 0.55*score_keywords(kw_schizo) + 0.20*surprise + 0.25*disgust
    sui_sc    = 0.60*score_keywords(kw_suicide) + 0.25*sadness + 0.15*max(0.0, -vader_avg)

    # Normalize to ~[0,1] caps
    def norm(x, cap): return min(x / cap, 1.0)
    risks = {
        "depression": bucket(norm(dep_score, 1.2)),
        "anxiety": bucket(norm(anx_score, 1.2)),
        "ptsd": bucket(norm(ptsd_sc, 1.0)),
        "schizophrenia": bucket(norm(sch_sc, 1.0)),
        "suicidal": bucket(norm(sui_sc, 1.0)),
    }
    return risks

# =========================
# Frontend Pages
# =========================

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/dashboard")
def dashboard():
    username = request.args.get("username")
    if not username:
        return redirect(url_for("home"))
    return render_template("dashboard.html", username=username)

# --- Therapist Pages ---

@app.route("/therapist")
def therapist_dashboard():
    """Main Therapist Dashboard"""
    return render_template("therapist_dashboard.html")

@app.route("/patient/<string:author>")
def patient_detail(author):
    """
    Therapist insights page for a single Reddit author.
    If author not found in cached CSV, fetch fresh posts live from Reddit.
    """
    df = load_cached_df()
    user_df = pd.DataFrame()

    if not df.empty and "author" in df.columns:
        user_df = df[df["author"].str.lower() == author.lower()]

    # Fallback: live fetch if not in cache
    if user_df.empty:
        try:
            print(f"⚡ Fetching live Reddit data for {author} ...")
            posts = fetch_user_submissions(author, limit=200)
            if not posts:
                return render_template(
                    "patient_detail.html",
                    author=author,
                    aura={"aura": "(no data)", "description": "No public posts found for this user."},
                    big5={"openness": 0, "conscientiousness": 0, "extraversion": 0, "agreeableness": 0, "neuroticism": 0},
                    risks={"depression": "low", "anxiety": "low", "ptsd": "low", "schizophrenia": "low", "suicidal": "low"},
                    trend_labels=[], trend_series={}, trend_summary="No data available.",
                    quick_insight="This Reddit user has no recent posts or profile is private.",
                    session_tips=["Ask user to engage or share reflections to analyze trends."]
                )
            # Build DataFrame on the fly
            user_df = pd.DataFrame(posts)
            user_df["created_utc"] = user_df["created"].apply(lambda x: x.timestamp())
            # quick VADER if not present
            if "vader_compound" not in user_df.columns:
                user_df["vader_compound"] = user_df["text"].map(lambda t: analyzer.polarity_scores(t)["compound"])
        except Exception as e:
            print(f"❌ Error fetching live data: {e}")
            return render_template(
                "patient_detail.html",
                author=author,
                aura={"aura": "(error)", "description": "Unable to fetch data."},
                big5={"openness": 0, "conscientiousness": 0, "extraversion": 0, "agreeableness": 0, "neuroticism": 0},
                risks={"depression": "low", "anxiety": "low", "ptsd": "low", "schizophrenia": "low", "suicidal": "low"},
                trend_labels=[], trend_series={}, trend_summary="Error fetching data.",
                quick_insight=f"Failed to fetch Reddit data for {author}.",
                session_tips=["Retry after a few minutes or verify username spelling."]
            )

    # Prepare texts & dates
    text_col = (user_df["title"].fillna("") + " " + user_df.get("text", "").fillna("")).str.strip()
    texts = text_col.tolist()
    dates = pd.to_datetime(user_df["created_utc"], unit="s", errors="coerce", utc=True).dt.date.astype(str).tolist()

    # Aura
    aura = analyze_aura([{"text": t} for t in texts[:60]])

    # Daily stacked emotion trend
    trend_labels, trend_series = _analyze_emotion_trend(texts[:120], dates[:120])

    # VADER average (from df or quick calc)
    vader_avg = float(user_df.get("vader_compound", pd.Series([0])).astype(float).mean())

    # Big Five heuristic (0-100)
    big5 = analyze_big5(texts)

    # Clinical risk flags (heuristic)
    risks = _assess_risks(texts, vader_avg, trend_series)

    # Quick insight & trend summary (text)
    dom_emotion = None
    if trend_series:
        last_vals = {k: (trend_series[k][-1] if trend_series[k] else 0.0) for k in trend_series}
        dom_emotion = max(last_vals, key=last_vals.get)
    quick_insight = f"Recent language shows {dom_emotion or 'balanced'} affect; average sentiment {vader_avg:+.2f}. Aura suggests: {aura.get('aura','')}."

    def pct(x): return f"{round(x*100)}%"
    trend_summary = "Over recent posts, emotion mix shows " + \
        ", ".join([f"{k}: {pct(v[-1])}" for k, v in trend_series.items() if v]) + "."

    # Session tips
    tips = []
    if risks.get("suicidal") == "high":
        tips.append("Assess safety first; ask about intent, plan, means; provide crisis resources.")
    if risks.get("depression") in ("moderate", "high"):
        tips.append("Screen for MDD; explore sleep, appetite, anhedonia.")
    if risks.get("anxiety") in ("moderate", "high"):
        tips.append("Use grounding/breathing; identify triggers; consider CBT psychoeducation.")
    if risks.get("ptsd") in ("moderate", "high"):
        tips.append("Check trauma history and avoidance; stabilize before trauma processing.")
    if risks.get("schizophrenia") in ("moderate", "high"):
        tips.append("Clarify reality-testing issues; consider psychiatric referral.")
    if not tips:
        tips.append("Build rapport; reinforce strengths from positive/neutral periods.")
    tips.append("Validate emotions reflected in recent posts and set session goals.")

    return render_template(
        "patient_detail.html",
        author=author,
        aura=aura,
        big5=big5,
        risks=risks,
        trend_labels=trend_labels,
        trend_series=trend_series,
        trend_summary=trend_summary,
        quick_insight=quick_insight,
        session_tips=tips
    )

@app.route("/api/therapist/search")
def therapist_search():
    """Dynamic search and filter API for therapist dashboard"""
    name_query = request.args.get("name", "").lower()
    emotion_filter = request.args.get("emotion", "").lower()

    df = load_cached_df()
    if df.empty:
        return jsonify({"patients": [], "note": "Cache empty. Please scrape first."})

    # Aggregate per author
    patients = (
        df.groupby("author")
        .agg({"title": "count"})
        .rename(columns={"title": "post_count"})
        .reset_index()
    )
    # Dummy placeholder dominant emotion (can replace with real later)
    patients["dominant_emotion"] = [
        "happy" if i % 3 == 0 else "sad" if i % 3 == 1 else "calm"
        for i in range(len(patients))
    ]

    if name_query:
        patients = patients[patients["author"].str.lower().str.contains(name_query)]
    if emotion_filter:
        patients = patients[patients["dominant_emotion"].str.lower() == emotion_filter]

    result = patients.head(250).to_dict("records")
    return jsonify({"patients": result, "count": len(result)})

# =========================
# APIs used by the user dashboard
# =========================

@app.route("/api/fetch/<string:username>")
def api_fetch(username):
    """Fetch recent submissions for a username and cache them in memory."""
    try:
        posts = fetch_user_submissions(username, limit=200)
        return jsonify({"ok": True, "fetched": len(posts)})
    except Exception as e:
        # Common causes: bad Reddit credentials, non-existent user, rate limit
        return jsonify({"ok": False, "error": str(e), "fetched": 0}), 500

@app.route("/api/mood_trend/<string:username>")
def api_mood_trend(username):
    """Return daily average VADER compound sentiment for the last N days."""
    try:
        days = int(request.args.get("days", 60))
    except Exception:
        days = 60
    trend = get_daily_mood(username, days=days) or []
    return jsonify({"username": username, "trend": trend})

@app.route("/api/aura/<string:username>")
def api_aura(username):
    """Return aura for the user's cached posts."""
    posts = user_posts.get(username, [])
    payload = [{"text": (p.get("title","") + " " + p.get("text","")).strip()} for p in posts]
    try:
        result = analyze_aura(payload)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/emotions/<string:username>")
def api_emotions(username):
    """Return aggregated emotion distribution from BERT for the user's cached posts."""
    posts = user_posts.get(username, [])
    payload = [{"text": (p.get("title","") + " " + p.get("text","")).strip()} for p in posts[:40]]
    try:
        result = analyze_emotions(payload)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- RUN APP ---
if __name__ == "__main__":
    app.run(debug=True)
