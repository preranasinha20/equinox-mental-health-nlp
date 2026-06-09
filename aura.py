# aura.py
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# --- Load model once globally for efficiency ---
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Aura mapping (Spotify-style colors + meaning) ---
AURA_MAP = {
    0: ("🌿 Calm Green", "Reflective and grounded — you often express thoughtfulness."),
    1: ("🔥 Radiant Orange", "Energetic and expressive — your posts show high engagement."),
    2: ("🌊 Tranquil Blue", "Balanced and introspective — calm tone with positive reflections."),
    3: ("🌪️ Stormy Gray", "You’ve shared signs of stress or emotional intensity recently."),
    4: ("🌸 Blossom Pink", "Compassionate and emotionally aware — empathetic tone detected."),
    5: ("🌞 Bright Yellow", "Optimistic and uplifting — your tone reflects positivity.")
}

# --- Text cleaning helper ---
def preprocess(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\[removed\]|\[deleted\]", "", text)
    text = re.sub(r"[^a-z0-9\s.,!?']", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# --- Main Aura Analysis Function ---
def analyze_aura(posts):
    """Given list of posts, returns aura and description."""
    if not posts:
        return {
            "aura": "🌊 Tranquil Blue",
            "description": "Balanced and introspective — calm tone with positive reflections.",
        }

    texts = [p["text"] for p in posts if p["text"].strip()]
    if len(texts) < 1:
        return {
            "aura": "🌊 Tranquil Blue",
            "description": "Balanced and introspective — calm tone with positive reflections.",
        }

    # === Step 1: Generate embeddings ===
    embeddings = model.encode(texts, show_progress_bar=False)

    # === Step 2: Cluster posts into 6 auras ===
    try:
        kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(embeddings)
        user_cluster = int(np.argmax(np.bincount(clusters)))
    except Exception:
        user_cluster = 2  # fallback to Tranquil Blue

    aura_color, aura_desc = AURA_MAP.get(
        user_cluster, ("🌊 Tranquil Blue", "Balanced and introspective.")
    )

    return {
        "aura": aura_color,
        "description": aura_desc,
    }
