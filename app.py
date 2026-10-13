# app.py
import os
import re
import logging
from difflib import get_close_matches
from functools import lru_cache, wraps
from datetime import datetime, timedelta
from collections import defaultdict, Counter

from flask import Flask, render_template, jsonify, request, redirect, url_for, session
from werkzeug.exceptions import HTTPException
import praw
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Local modules
from ingest import scrape_subreddits, load_cached_df
from aura import analyze_aura
from nlp_bert import analyze_emotions, get_emotion_analyzer  # reuse loaded HF pipeline
from personality import analyze_big5  # Big Five heuristic

# --- CONFIG ---
REDDIT_CLIENT_ID = 'M1I1jvc74xBb2xr-QuK2zQ'
REDDIT_CLIENT_SECRET = "qSFZqF4lbrk5kZ9q3UP44n0RMBHrig"
REDDIT_USER_AGENT = "wellness-tracker-demo"

# --- APP INIT ---
app = Flask(__name__)
# Set EQUINOX_SECRET_KEY in the environment for stable, signed sessions.
# Production startup fails closed if required configuration is missing.
IS_PRODUCTION = os.environ.get("EQUINOX_ENV", "").lower() == "production"
if IS_PRODUCTION and not os.environ.get("EQUINOX_SECRET_KEY"):
    raise RuntimeError("Set EQUINOX_SECRET_KEY before starting Equinox in production.")
if IS_PRODUCTION and not os.environ.get("EQUINOX_DEMO_PASSWORD"):
    raise RuntimeError("Set EQUINOX_DEMO_PASSWORD before starting Equinox in production.")
if IS_PRODUCTION and (not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET):
    raise RuntimeError("Set Reddit API credentials in the environment before starting Equinox in production.")
app.secret_key = os.environ.get("EQUINOX_SECRET_KEY", "local-demo-key-change-before-deploy")
DEMO_PASSWORD = os.environ.get("EQUINOX_DEMO_PASSWORD", "user123")
analyzer = SentimentIntensityAnalyzer()
user_posts = {}  # in-memory cache per username for the user dashboard
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("equinox")

GENERIC_API_ERROR = "Something went wrong. Please try again."


def api_safe(fn):
    """Return a generic JSON error for unexpected API failures; never leak exception text."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except HTTPException:
            raise
        except Exception:
            logger.exception("API request failed: %s", request.path)
            return jsonify({"ok": False, "error": GENERIC_API_ERROR}), 500
    return wrapper


@app.errorhandler(404)
def handle_404(exc):
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": "Not found."}), 404
    return exc


@app.errorhandler(500)
def handle_500(exc):
    logger.error("Internal server error", exc_info=exc)
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": GENERIC_API_ERROR}), 500
    return exc

# --- Reddit Init ---
def init_reddit():
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
        requestor_kwargs={"timeout": 6},
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

    try:
        ruser = reddit.redditor(username)

        posts = []

        for s in ruser.submissions.new(limit=limit):
            text = (getattr(s, "title", "") or "") + " " + (
                getattr(s, "selftext", "") or ""
            )

            posts.append({
                "id": getattr(s, "id", ""),
                "title": getattr(s, "title", "") or "",
                "created": datetime.utcfromtimestamp(
                    getattr(s, "created_utc", 0)
                ),
                "text": preprocess_text(text),
            })

        # Demo fallback must only use posts belonging to this exact author.
        # Never fill a missing profile with the first rows from the global dataset.
        if len(posts) == 0:
            df = load_cached_df()
            posts = []
            if not df.empty and "author" in df.columns:
                matched = df[df["author"].fillna("").astype(str).str.casefold() == str(username).casefold()].copy()
                if "created_utc" in matched.columns:
                    matched["_created_numeric"] = pd.to_numeric(matched["created_utc"], errors="coerce")
                    matched = matched.sort_values("_created_numeric", ascending=False)
                for idx, row in matched.head(limit).iterrows():
                    created = pd.to_datetime(row.get("created_utc"), unit="s", errors="coerce")
                    if pd.isna(created):
                        continue
                    posts.append({
                        "id": str(row.get("id", idx)),
                        "title": str(row.get("title", "") or ""),
                        "created": created.to_pydatetime(),
                        "text": preprocess_text(str(row.get("title", "") or "") + " " + str(row.get("text", "") or "")),
                    })

        user_posts[username] = posts
        return posts

    except Exception:
        logger.exception("Unable to load submissions")
        return []

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
        res = get_emotion_analyzer()(t[:512])[0]  # list of dicts
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

@app.route("/login/<role>", methods=["GET", "POST"])
def login(role):
    """Demo-only role login. Replace shared demo credentials before any real deployment."""
    if role not in ("user", "therapist"):
        return redirect(url_for("home"))
    error = None
    if request.method == "POST":
        password = request.form.get("password", "")
        identity = (request.form.get("identity", "") or "").strip()
        if password != DEMO_PASSWORD:
            error = "That demo password isn't correct. Check the credentials provided by your administrator."
        elif role == "user" and not identity:
            error = "Enter the Reddit username you want to explore."
        else:
            session.clear()
            session["role"] = role
            session["identity"] = identity or "Therapist"
            return redirect(url_for("dashboard") if role == "user" else url_for("therapist_dashboard"))
    return render_template("login.html", role=role, error=error, show_demo_credentials=not IS_PRODUCTION)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/dashboard")
def dashboard():
    if session.get("role") != "user" or not session.get("identity"):
        return redirect(url_for("login", role="user"))
    username = session["identity"]
    return render_template("dashboard.html", username=username)

# --- Therapist Pages ---

@app.route("/therapist")
def therapist_dashboard():
    """Main Therapist Dashboard"""
    if session.get("role") != "therapist":
        return redirect(url_for("login", role="therapist"))
    therapist_name = session.get("identity", "Therapist")
    return render_template("therapist_dashboard.html", therapist_name=therapist_name)

@app.route("/patient/<string:author>")
def patient_detail(author):
    """
    Therapist insights page for a single Reddit author.
    If author not found in cached CSV, fetch fresh posts live from Reddit.
    """
    if session.get("role") != "therapist":
        return redirect(url_for("login", role="therapist"))
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
        except Exception:
            logger.exception("Unable to fetch live patient data")
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

@lru_cache(maxsize=1)
def _cached_therapist_profiles():
    """Build profile summaries once; use cached VADER scores where available."""
    df = load_cached_df()
    if df.empty or "author" not in df.columns:
        return []
    df = df.dropna(subset=["author"]).copy()
    if df.empty:
        return []
    if "vader_compound" in df.columns:
        df["_score"] = pd.to_numeric(df["vader_compound"], errors="coerce")
    else:
        title = df.get("title", pd.Series("", index=df.index)).fillna("").astype(str)
        body = df.get("text", pd.Series("", index=df.index)).fillna("").astype(str)
        df["_score"] = (title + " " + body).map(lambda value: analyzer.polarity_scores(value)["compound"])
    summaries = []
    for author, group in df.groupby("author", sort=False):
        scores = group["_score"].dropna()
        average = float(scores.mean()) if not scores.empty else 0.0
        mood = "positive" if average >= 0.05 else ("negative" if average <= -0.05 else "neutral")
        summaries.append({"author": str(author), "post_count": int(len(group)),
                          "dominant_emotion": mood, "sentiment_score": round(average, 3)})
    summaries.sort(key=lambda item: (-item["post_count"], item["author"].casefold()))
    return summaries

@app.route("/api/therapist/search")
@api_safe
def therapist_search():
    """Fast search over cached profile summaries with username, sentiment and activity filters."""
    if session.get("role") != "therapist":
        return jsonify({"error": "Therapist sign-in required."}), 403
    name_query = request.args.get("name", "").strip().casefold()
    sentiment_filter = request.args.get("sentiment", "").strip().casefold()
    try:
        min_posts = max(1, min(int(request.args.get("min_posts", "1")), 10000))
    except (TypeError, ValueError):
        min_posts = 1
    profiles = _cached_therapist_profiles()
    results = [p for p in profiles
               if name_query in p["author"].casefold()
               and p["post_count"] >= min_posts
               and (sentiment_filter not in ("positive", "neutral", "negative")
                    or p["dominant_emotion"] == sentiment_filter)]
    results = results[:250]
    return jsonify({"patients": results, "count": len(results),
                    "dataset_profiles": len(profiles)})

# =========================
# APIs used by the user dashboard
# =========================

@app.route("/api/fetch/<string:username>")
@api_safe
def api_fetch(username):
    """Fetch recent submissions for the signed-in user's own username."""
    if session.get("role") != "user" or session.get("identity", "").casefold() != username.casefold():
        return jsonify({"error": "User sign-in required for this profile."}), 403
    posts = fetch_user_submissions(username, limit=40)
    source = "cached" if posts and any(str(p.get("id", "")).isdigit() for p in posts) else ("reddit" if posts else "unavailable")
    return jsonify({"ok": bool(posts), "fetched": len(posts), "source": source,
                    "message": "" if posts else "No public posts could be loaded for this username. Check the spelling or Reddit API connection."})

@app.route("/api/posts/<string:username>")
@api_safe
def api_recent_posts(username):
    """Return sanitized recent posts for the signed-in user's own profile."""
    if session.get("role") != "user" or session.get("identity", "").casefold() != username.casefold():
        return jsonify({"error": "User sign-in required for this profile."}), 403
    posts = user_posts.get(username, [])
    return jsonify({"posts": [
        {"title": (p.get("title") or "Untitled post")[:180],
         "date": p.get("created").strftime("%b %d, %Y") if p.get("created") else "Date unavailable",
         "text": (p.get("text") or "")[:320]}
        for p in sorted(posts, key=lambda item: item.get("created") or datetime.min, reverse=True)[:12]
    ]})

@app.route("/api/mood_trend/<string:username>")
@api_safe
def api_mood_trend(username):
    """Return daily average VADER compound sentiment for the signed-in user."""
    if session.get("role") != "user" or session.get("identity", "").casefold() != username.casefold():
        return jsonify({"error": "User sign-in required for this profile."}), 403
    try:
        days = int(request.args.get("days", 60))
    except Exception:
        days = 60
    trend = get_daily_mood(username, days=days) or []
    return jsonify({"username": username, "trend": trend})


@app.route("/api/aura/<string:username>")
@api_safe
def api_aura(username):
    """Return aura using the therapist page's data-selection logic."""
    if (
        session.get("role") != "user"
        or session.get("identity", "").casefold() != username.casefold()
    ):
        return jsonify({"error": "User sign-in required for this profile."}), 403

    df = load_cached_df()
    user_df = pd.DataFrame()

    if not df.empty and "author" in df.columns:
        user_df = df[
            df["author"].fillna("").astype(str).str.casefold()
            == username.casefold()
        ].copy()

    if user_df.empty:
        posts = fetch_user_submissions(username, limit=200)
        if not posts:
            return jsonify({
                "aura": "(no data)",
                "description": "No public posts found for this user."
            })
        user_df = pd.DataFrame(posts)

    titles = (
        user_df["title"].fillna("").astype(str)
        if "title" in user_df.columns
        else pd.Series("", index=user_df.index)
    )
    bodies = (
        user_df["text"].fillna("").astype(str)
        if "text" in user_df.columns
        else pd.Series("", index=user_df.index)
    )

    texts = (titles + " " + bodies).str.strip().tolist()
    result = analyze_aura([{"text": text} for text in texts[:60]])
    return jsonify(result)

@app.route("/api/emotions/<string:username>")
@api_safe
def api_emotions(username):
    """Return emotion distribution from BERT for the signed-in user's cached posts."""
    if session.get("role") != "user" or session.get("identity", "").casefold() != username.casefold():
        return jsonify({"error": "User sign-in required for this profile."}), 403
    posts = user_posts.get(username, [])
    payload = [{"text": (p.get("title","") + " " + p.get("text","")).strip()} for p in posts[:40]]
    result = analyze_emotions(payload)
    return jsonify(result)

# --- RUN APP ---
if __name__ == "__main__":
    app.run(debug=not IS_PRODUCTION)
