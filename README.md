# 🧠 Aradhya Recruiter Engine
### AI Candidate Ranking System — Hack2Skill India Runs Challenge

> *"Not keyword matching. Actual understanding."*

---

## 🎯 What This Does

Ranks job candidates the way a great recruiter would — by understanding the **full picture**, not just matching words.

## 🏗️ Architecture

```
Job Description (text)
        ↓
┌──────────────────────────────┐
│   LLM-Based JD Deep Parser   │  ← Extracts: skills, behavioral signals,
│   (Claude claude-opus-4-5)        │     culture indicators, responsibilities
└──────────────┬───────────────┘
               ↓
    ┌──────────────────────┐
    │  5-Signal Hybrid     │
    │  Scoring Engine      │
    └──────────────────────┘

Signal 1 (25%): Semantic Match      — sentence-transformers cosine similarity
Signal 2 (35%): LLM Assessment      — holistic recruiter-level judgment
Signal 3 (15%): Experience Fit      — band matching with graceful over/under scoring
Signal 4 (15%): Skill Coverage      — required + preferred skill overlap
Signal 5 (10%): Platform Activity   — GitHub, open source, hackathon signals

               ↓
    Weighted Hybrid Score (0–1)
               ↓
    Ranked CSV + Detailed JSON
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY=your_key_here

# Run with your dataset
python src/pipeline.py

# Output: output/ranked_candidates.csv
#         output/ranked_candidates_detailed.json
```

## 📁 Project Structure

```
aradhya-recruiter/
├── src/
│   └── pipeline.py          # Main engine
├── data/
│   ├── candidates.csv       # Input dataset (add when received)
│   └── job_description.txt  # JD (add when received)
├── output/
│   ├── ranked_candidates.csv
│   └── ranked_candidates_detailed.json
├── requirements.txt
└── README.md
```

## 📊 Output Format

| Rank | Candidate_ID | Name | Total_Score | Semantic_Match | LLM_Assessment | Experience_Fit | Skill_Coverage | Platform_Activity | Recruiter_Note |
|------|-------------|------|-------------|----------------|----------------|----------------|----------------|-------------------|----------------|
| 1 | C001 | Arjun Sharma | 0.847 | 0.823 | 0.871 | 1.0 | 0.900 | 0.750 | Strong ML background... |

## 🧠 Why This Beats Keyword Matching

| Approach | What It Misses |
|----------|---------------|
| Keyword filter | Synonyms, context, growth potential |
| TF-IDF | Semantic meaning, behavioral signals |
| Simple embeddings | Career trajectory, platform credibility |
| **This engine** | Nothing — 5 signals, LLM judgment |

---

*Built by Sarthak Gavade (AatmaCode) | Hack2Skill India Runs 2026*
