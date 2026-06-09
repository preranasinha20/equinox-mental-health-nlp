**Equinox – AI-Powered Health & Wellness Analyzer**

*“Balancing minds, one insight at a time.”*

---

### **Overview**

Equinox is a clinical mental health intelligence platform that analyses a user's Reddit post history to generate emotional profiles, personality insights, and risk indicators — accessible through both a patient dashboard and a therapist interface.


### **Concept**
Mental health professionals currently have no passive insight into patient emotional states between sessions. Equinox bridges this gap by analysing publicly available social media language patterns to provide therapists with continuous, non-intrusive emotional monitoring.
The Aura system — the core innovation — generates a Spotify-style emotional fingerprint from a user's posting history. Just as Spotify analyses listening patterns to assign an audio aura, Equinox analyses language patterns to assign an emotional one.


### **Features**

**Patient View**

Aura — personalised emotional colour profile (Calm Green, Tranquil Blue, Radiant Orange, Stormy Gray, Blossom Pink, Bright Yellow)
Mood Trend over last 60 days (daily sentiment average)
Weekly Emotion Distribution radar chart (7 emotions)

**Insights View**

BERT-based Trend Evolution Graph tracking emotion intensity over time
Big Five Personality Insights (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism)
Clinical Risk Indicators — Depression, Anxiety, PTSD, Schizophrenia, Suicidal risk (heuristic, non-diagnostic)
Session Insights with therapist guidance suggestions

**Therapist Dashboard**

Search patients by Reddit username
View dominant emotion and post history summary
Session preparation insights per patient
---

### 🧩 **Architecture Overview**

```
┌─────────────────────────────────────┐
│   Reddit API (PRAW)                │  ← Data source
└──────────────┬──────────────────────┘
               │
               ▼
   ┌──────────────────────────────┐
   │  Flask Backend (app.py)      │
   │  + REST APIs (/api/*)        │
   └──────────────┬───────────────┘
                  │
     ┌──────────────────────────────┐
     │  NLP Models                  │
     │  • VADER (sentiment)         │
     │  • Sentence-BERT (embeddings)│
     │  • K-Means (aura clusters)   │
     │  • DistilRoBERTa (emotions)  │
     │  • Big Five (OCEAN)          │
     └──────────────┬───────────────┘
                    │
                    ▼
        ┌──────────────────────┐
        │  Frontend (HTML + JS)│
        │  • User Dashboard    │
        │  • Therapist Portal  │
        └──────────────────────┘
```

---
**Data Pipeline**
**Scrapes 7 subreddits:** mentalhealth, selfimprovement, college, happiness, Anxiety, Depression, relationships
**Four feed types per subreddit:** top, hot, new, rising — for a rich longitudinal dataset rather than just recent activity.


### **Installation & Setup**

#### Prerequisites

* Python ≥ 3.10
* pip (latest version)
* Reddit API credentials

### Steps

```bash
# 1️⃣ Clone the repository
git clone https://github.com/ombansal1/equinox.git
cd equinox

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Add Reddit API credentials
#    Inside app.py → replace:
#    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

# 4️⃣ Run the Flask app
python app.py
```

Then open → **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

**Core Workflow**

| Stage                        | Description                                     | Tools / Models                   |
| ---------------------------- | ----------------------------------------------- | -------------------------------- |
| **1. Data Collection**       | Scrape Reddit posts using PRAW                  | `ingest.py`                      |
| **2. Pre-processing**        | Clean & normalize text (lowercase, remove URLs) | Regex / Pandas                   |
| **3. Sentiment Analysis**    | Compute compound scores                         | VADER                            |
| **4. Embedding Generation**  | Create vector representations                   | Sentence-BERT (all-MiniLM-L6-v2) |
| **5. Clustering**            | Identify aura groups                            | K-Means                          |
| **6. Emotion Detection**     | Detect fine-grained emotions                    | DistilRoBERTa                    |
| **7. Personality Profiling** | Map linguistic features to OCEAN traits         | Big Five Model                   |
| **8. Visualization**         | Dashboards + Charts                             | Chart.js, Matplotlib             |

---

**Models Used**

| Model                | Type                | Purpose                                 |
| -------------------- | ------------------- | --------------------------------------- |
| **Naive Bayes**      | Probabilistic       | Baseline sentiment classification       |
| **VADER**            | Rule-based          | Social-media sentiment detection        |
| **Sentence-BERT**    | Transformer         | Semantic embeddings of text             |
| **K-Means**          | Unsupervised ML     | Emotional clustering / Aura detection   |
| **Big Five (OCEAN)** | Psychological Model | Personality trait prediction            |
| **DistilRoBERTa**    | Transformer         | Emotion classification (seven emotions) |

---

**Folder Structure**

```
equinox/
│
├── app.py                    # Flask backend
├── aura.py                   # Aura clustering logic
├── nlp_bert.py               # Emotion analysis (DistilRoBERTa)
├── ingest.py                 # Reddit scraper + cache
│
├── templates/                # HTML templates
│   ├── home.html
│   ├── dashboard.html
│   ├── therapist_dashboard.html
│   └── patient_detail.html
│
├── static/                   # Static assets (CSS / Images)
│   └── style.css
│
└── uploads/                  # Cached Reddit data (JSON / CSV)
```

---

### **Visualizations**

* **Mood Trend Graph:** Average VADER sentiment per day
* **Emotion Radar:** Weekly emotion distribution from BERT
* **Aura Card:** User’s dominant emotional cluster
* **Therapist Dashboard:** Patient search & dominant emotion summary

---

### **Reddit API Credentials**

Create a Reddit app at [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
and update `app.py`:

```python
REDDIT_CLIENT_ID = "your_client_id"
REDDIT_CLIENT_SECRET = "your_client_secret"
REDDIT_USER_AGENT = "Equinox-App-by-Om-Bansal"
```

---

**Known Limitations**

Minimum 10–15 posts recommended for reliable aura classification and mood trend analysis
Clinical risk indicators are heuristic and non-diagnostic — not a substitute for professional assessment
Model performance on non-English or code-switched text is untested
Reddit posts reflect public expression, not necessarily private emotional states

**Future Scope**

Fine-tune emotion model on mental health-specific Reddit corpora
Replace heuristic Big Five mapping with a trained personality classifier
Longitudinal deterioration detection — flag users whose emotional trajectory worsens over weeks
Therapist note integration for session continuity

**Contributors**

Prerana Sinha
Om Bansal
Pavan Bhapkar
