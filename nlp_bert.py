# nlp_bert.py
from transformers import pipeline
from collections import defaultdict

# Lazy-load the model only when needed
emotion_analyzer = None

def get_emotion_analyzer():
    global emotion_analyzer

    if emotion_analyzer is None:
        emotion_analyzer = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True,
        )

    return emotion_analyzer

def analyze_emotions(posts):
    """
    Given a list of posts, returns aggregated emotion distribution over the week.
    Example output: {"joy": 0.54, "sadness": 0.12, ...}
    """
    emotions = defaultdict(list)

    for p in posts:
        text = p["text"].strip()
        if not text:
            continue

        # Analyze text (truncate long Reddit posts for performance)
        result = get_emotion_analyzer()(text[:512])

        # Aggregate emotion scores
        for score_obj in result[0]:
            label = score_obj["label"].lower()
            emotions[label].append(score_obj["score"])

    # Average scores
    avg_emotions = {label: round(sum(vals) / len(vals), 3) for label, vals in emotions.items() if vals}

    return avg_emotions
