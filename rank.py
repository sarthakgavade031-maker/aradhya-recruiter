#!/usr/bin/env python3
"""
Aradhya Recruiter Engine — rank.py
===================================
Ranks 100K candidates for Senior AI Engineer JD.
- NO API calls during ranking (CPU only, offline)
- Runs in < 5 minutes on 16GB RAM
- Honeypot detection built-in
- Outputs top 100 candidates with reasoning

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Author: Sarthak Gavade
"""

import json
import csv
import argparse
import gzip
import re
import math
from datetime import date, datetime
from pathlib import Path

# ── JD Constants (Senior AI Engineer — Redrob AI) ─────────────────────────────

REQUIRED_SKILLS = {
    "embeddings", "sentence-transformers", "vector database", "faiss", "pinecone",
    "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch", "hybrid search",
    "retrieval", "ranking", "python", "evaluation", "ndcg", "mrr", "a/b test",
    "information retrieval", "semantic search", "dense retrieval", "bm25",
    "bge", "e5", "openai embeddings", "vector search", "rag",
    "retrieval augmented", "re-ranking", "reranking"
}

PREFERRED_SKILLS = {
    "lora", "qlora", "peft", "fine-tuning", "fine tuning", "finetuning",
    "learning to rank", "xgboost", "llm", "large language model",
    "transformer", "bert", "recommendation system", "search", "nlp",
    "langchain", "distributed systems", "inference optimization",
    "open source", "pytorch", "tensorflow", "huggingface"
}

# Titles that indicate strong fit
GOOD_TITLE_KEYWORDS = {
    "ai engineer", "ml engineer", "machine learning", "search engineer",
    "ranking engineer", "nlp engineer", "applied scientist", "research engineer",
    "data scientist", "recommendation", "retrieval", "senior engineer",
    "staff engineer", "principal engineer"
}

# Titles that indicate poor fit (JD explicitly says so)
BAD_TITLE_KEYWORDS = {
    "marketing", "sales", "hr manager", "accountant", "civil engineer",
    "mechanical engineer", "operations manager", "graphic designer",
    "business analyst", "frontend engineer", "customer support",
    "project manager", "qa engineer", "devops"
}

# Companies JD says are red flags (pure services)
BAD_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "tech mahindra", "mphasis", "hexaware", "mindtree", "hcl"
}

# JD wants: 5-9 years, product companies, India locations
TARGET_EXP_MIN = 5.0
TARGET_EXP_MAX = 9.0
PREFERRED_LOCATIONS = {
    "pune", "noida", "delhi", "ncr", "hyderabad", "mumbai", "bangalore",
    "bengaluru", "india", "gurgaon", "gurugram", "delhi ncr"
}

TODAY = date.today()


# ── Scoring Functions ──────────────────────────────────────────────────────────

def skill_score(candidate: dict) -> float:
    """
    Score based on skills — with trust multiplier.
    Penalizes keyword stuffing (high skills, 0 duration/endorsements).
    """
    skills = candidate.get("skills", [])
    if not skills:
        return 0.0

    required_match = 0.0
    preferred_match = 0.0
    total_trust = 0.0

    for sk in skills:
        name = sk.get("name", "").lower()
        proficiency = sk.get("proficiency", "beginner")
        endorsements = sk.get("endorsements", 0)
        duration = sk.get("duration_months", 0)

        # Trust multiplier — penalize keyword stuffers
        # Expert proficiency with 0 months used = suspicious
        prof_weight = {"beginner": 0.4, "intermediate": 0.7, "advanced": 0.9, "expert": 1.0}
        pw = prof_weight.get(proficiency, 0.5)

        # Endorsement trust (log scale)
        endorse_trust = min(1.0, math.log1p(endorsements) / math.log1p(50))

        # Duration trust
        duration_trust = min(1.0, duration / 24.0)  # 24 months = full trust

        # Combined trust — expert with 0 duration gets penalized
        if proficiency == "expert" and duration == 0:
            trust = 0.2  # honeypot signal
        else:
            trust = (pw * 0.4 + endorse_trust * 0.3 + duration_trust * 0.3)

        # Check skill relevance
        is_required = any(req in name or name in req for req in REQUIRED_SKILLS)
        is_preferred = any(pref in name or name in pref for pref in PREFERRED_SKILLS)

        if is_required:
            required_match += trust
        elif is_preferred:
            preferred_match += trust * 0.5

        total_trust += trust

    # Normalize
    req_score = min(1.0, required_match / max(len(REQUIRED_SKILLS) * 0.3, 1))
    pref_score = min(1.0, preferred_match / 5.0)

    return round(req_score * 0.75 + pref_score * 0.25, 4)


def career_score(candidate: dict) -> float:
    """
    Score career history — product companies, ML roles, trajectory.
    """
    history = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    current_title = profile.get("current_title", "").lower()
    current_company = profile.get("current_company", "").lower()

    if not history:
        return 0.0

    score = 0.0

    # Current title check
    title_bonus = 0.0
    if any(kw in current_title for kw in GOOD_TITLE_KEYWORDS):
        title_bonus = 0.3
    if any(kw in current_title for kw in BAD_TITLE_KEYWORDS):
        title_bonus = -0.3

    # Current company — pure services penalty
    company_penalty = 0.0
    if any(bad in current_company for bad in BAD_COMPANIES):
        company_penalty = -0.15

    # Career history analysis
    ml_roles = 0
    product_company_months = 0
    services_company_months = 0
    total_months = 0
    has_recent_ml = False

    for job in history:
        title = job.get("title", "").lower()
        company = job.get("company", "").lower()
        company_size = job.get("company_size", "")
        duration = job.get("duration_months", 0)
        description = job.get("description", "").lower()
        is_current = job.get("is_current", False)
        total_months += duration

        # ML role detection
        is_ml_role = any(kw in title for kw in GOOD_TITLE_KEYWORDS)
        desc_ml = any(kw in description for kw in [
            "embedding", "retrieval", "ranking", "vector", "llm", "transformer",
            "recommendation", "search", "nlp", "machine learning", "ml model"
        ])
        if is_ml_role or desc_ml:
            ml_roles += 1
            if is_current:
                has_recent_ml = True

        # Product vs services
        is_services = any(bad in company for bad in BAD_COMPANIES)
        if is_services:
            services_company_months += duration
        else:
            product_company_months += duration

    # Compute career score
    ml_ratio = ml_roles / max(len(history), 1)
    product_ratio = product_company_months / max(total_months, 1)

    score = (
        ml_ratio * 0.35 +
        product_ratio * 0.30 +
        title_bonus +
        company_penalty +
        (0.15 if has_recent_ml else 0.0)
    )

    return round(max(0.0, min(1.0, score)), 4)


def experience_score(candidate: dict) -> float:
    """Score based on years of experience — JD wants 5-9 years."""
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)

    if TARGET_EXP_MIN <= yoe <= TARGET_EXP_MAX:
        return 1.0
    elif yoe < TARGET_EXP_MIN:
        gap = TARGET_EXP_MIN - yoe
        return max(0.0, 1.0 - gap * 0.15)
    else:
        over = yoe - TARGET_EXP_MAX
        return max(0.5, 1.0 - over * 0.04)  # overqualified still ok


def location_score(candidate: dict) -> float:
    """Score based on location preference — Pune/Noida/India preferred."""
    profile = candidate.get("profile", {})
    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    willing_to_relocate = candidate.get("redrob_signals", {}).get("willing_to_relocate", False)

    if any(loc in location for loc in PREFERRED_LOCATIONS):
        return 1.0
    elif country == "india":
        return 0.8
    elif willing_to_relocate:
        return 0.4
    else:
        return 0.1


def education_score(candidate: dict) -> float:
    """Score based on education tier and field."""
    education = candidate.get("education", [])
    if not education:
        return 0.3

    best_score = 0.0
    for edu in education:
        tier = edu.get("tier", "unknown")
        field = edu.get("field_of_study", "").lower()
        degree = edu.get("degree", "").lower()

        tier_score = {"tier_1": 1.0, "tier_2": 0.8, "tier_3": 0.6, "tier_4": 0.4, "unknown": 0.5}.get(tier, 0.5)

        # Relevant field bonus
        field_bonus = 0.0
        if any(f in field for f in ["computer", "software", "data", "ai", "machine", "information", "statistics"]):
            field_bonus = 0.1

        score = min(1.0, tier_score + field_bonus)
        best_score = max(best_score, score)

    return round(best_score, 4)


def behavioral_signal_score(candidate: dict) -> float:
    """
    Score based on Redrob behavioral signals.
    These are availability/engagement signals — a great candidate
    who isn't active is, for hiring purposes, not actually available.
    """
    rs = candidate.get("redrob_signals", {})
    if not rs:
        return 0.5

    score = 0.5  # neutral baseline

    # 1. Recency — last active date
    last_active_str = rs.get("last_active_date", "")
    if last_active_str:
        try:
            last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
            days_inactive = (TODAY - last_active).days
            if days_inactive <= 7:
                score += 0.15
            elif days_inactive <= 30:
                score += 0.10
            elif days_inactive <= 90:
                score += 0.05
            elif days_inactive > 180:
                score -= 0.20  # not available
        except:
            pass

    # 2. Open to work
    if rs.get("open_to_work_flag", False):
        score += 0.10

    # 3. Recruiter response rate
    rr = rs.get("recruiter_response_rate", 0.5)
    score += (rr - 0.5) * 0.20

    # 4. Notice period — JD wants sub-30 days
    notice = rs.get("notice_period_days", 60)
    if notice <= 30:
        score += 0.08
    elif notice <= 60:
        score += 0.04
    elif notice > 90:
        score -= 0.05

    # 5. GitHub activity
    github = rs.get("github_activity_score", -1)
    if github > 0:
        score += (github / 100) * 0.10

    # 6. Interview completion rate
    icr = rs.get("interview_completion_rate", 0.5)
    score += (icr - 0.5) * 0.08

    # 7. Profile completeness
    pc = rs.get("profile_completeness_score", 50)
    score += ((pc - 50) / 100) * 0.05

    # 8. Saved by recruiters (market signal)
    saved = rs.get("saved_by_recruiters_30d", 0)
    if saved > 5:
        score += 0.05
    elif saved > 2:
        score += 0.02

    # 9. Verified signals
    if rs.get("verified_email") and rs.get("verified_phone"):
        score += 0.03

    return round(max(0.0, min(1.0, score)), 4)


def honeypot_check(candidate: dict) -> bool:
    """
    Detect honeypot candidates — impossible profiles.
    Returns True if candidate is likely a honeypot.
    """
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    # Check 1: Experience at company founded after experience start
    yoe = profile.get("years_of_experience", 0)
    for job in history:
        start_date_str = job.get("start_date", "")
        if start_date_str:
            try:
                start_year = int(start_date_str[:4])
                company_duration = job.get("duration_months", 0)
                # If someone has 8 yrs exp but company founded 3 yrs ago
                # their start_date would be before company existed
                current_year = TODAY.year
                implied_start = current_year - yoe
                if start_year < implied_start - 2:
                    pass  # Could be a previous company
            except:
                pass

    # Check 2: Expert proficiency in many skills with 0 duration
    expert_zero_duration = sum(
        1 for sk in skills
        if sk.get("proficiency") == "expert" and sk.get("duration_months", 0) == 0
    )
    if expert_zero_duration >= 5:
        return True  # Keyword stuffer

    # Check 3: Too many expert skills (10+ is suspicious per README)
    expert_count = sum(1 for sk in skills if sk.get("proficiency") == "expert")
    if expert_count >= 10:
        return True

    # Check 4: Profile completeness 100 but last_active > 1 year ago
    rs = candidate.get("redrob_signals", {})
    pc = rs.get("profile_completeness_score", 0)
    last_active_str = rs.get("last_active_date", "")
    if pc == 100 and last_active_str:
        try:
            last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
            if (TODAY - last_active).days > 365:
                pass  # Not necessarily honeypot, just inactive

        except:
            pass

    # Check 5: Impossible experience timeline
    # Career history total > claimed years_of_experience by huge margin
    total_career_months = sum(j.get("duration_months", 0) for j in history)
    if yoe > 0 and total_career_months > (yoe + 5) * 12:
        return True  # Timeline doesn't add up

    return False


def generate_reasoning(candidate: dict, signals: dict) -> str:
    """Generate specific, honest 1-2 sentence reasoning."""
    profile = candidate.get("profile", {})
    rs = candidate.get("redrob_signals", {})
    name = profile.get("anonymized_name", "Candidate")
    title = profile.get("current_title", "")
    yoe = profile.get("years_of_experience", 0)
    company = profile.get("current_company", "")
    location = profile.get("location", "")
    notice = rs.get("notice_period_days", 60)
    last_active_str = rs.get("last_active_date", "")

    skill_names = [s["name"] for s in candidate.get("skills", [])
                   if any(r in s["name"].lower() for r in ["embedding", "vector", "retrieval", "llm", "rag", "faiss", "ranking", "nlp", "python", "transformer"])
                   ][:3]

    total = signals.get("total", 0)

    # Build reasoning parts
    parts = []

    # Strength
    if skill_names:
        parts.append(f"{yoe:.0f}yr {title} at {company} with relevant skills in {', '.join(skill_names)}")
    else:
        parts.append(f"{yoe:.0f}yr {title} at {company}")

    # Location
    loc_lower = location.lower()
    if any(l in loc_lower for l in ["pune", "noida", "delhi", "mumbai", "bangalore", "bengaluru", "hyderabad"]):
        parts.append(f"India-based ({location})")
    elif signals.get("location", 0) < 0.5:
        parts.append(f"outside preferred locations ({location})")

    # Concern / notice period
    if notice > 90:
        concern = f"notice period {notice}d is a concern"
    elif signals.get("behavioral", 0) < 0.4:
        concern = "low platform engagement reduces effective availability"
    else:
        concern = None

    sentence1 = "; ".join(parts) + "."
    sentence2 = f"{concern.capitalize()}." if concern else f"Active on platform with {notice}d notice period."

    return f"{sentence1} {sentence2}"


# ── Main Pipeline ──────────────────────────────────────────────────────────────

def score_candidate(candidate: dict) -> dict:
    """Compute all signals and total score for one candidate."""

    # Honeypot check first
    if honeypot_check(candidate):
        return {
            "candidate_id": candidate["candidate_id"],
            "total": 0.0,
            "is_honeypot": True,
            "signals": {}
        }

    # Compute 5 signals
    sk = skill_score(candidate)
    ca = career_score(candidate)
    ex = experience_score(candidate)
    lo = location_score(candidate)
    be = behavioral_signal_score(candidate)
    ed = education_score(candidate)

    # Weighted total — tuned for NDCG@10
    # Skill + Career are most important per JD analysis
    total = (
        sk * 0.30 +
        ca * 0.30 +
        ex * 0.15 +
        be * 0.12 +
        lo * 0.08 +
        ed * 0.05
    )

    signals = {
        "skill": sk,
        "career": ca,
        "experience": ex,
        "location": lo,
        "behavioral": be,
        "education": ed,
        "total": round(total, 4)
    }

    return {
        "candidate_id": candidate["candidate_id"],
        "total": round(total, 4),
        "is_honeypot": False,
        "signals": signals,
        "_candidate": candidate
    }


def load_candidates(path: str):
    """Load candidates from .jsonl or .jsonl.gz file."""
    p = Path(path)
    print(f"[1/4] Loading candidates from {p.name}...")

    if p.suffix == ".gz":
        opener = gzip.open(path, "rt", encoding="utf-8")
    else:
        opener = open(path, "r", encoding="utf-8")

    candidates = []
    with opener as f:
        first_char = f.read(1)
        f.seek(0) if hasattr(f, 'seek') else None

        # Re-open to reset position for gzip files
        pass

    # Re-open and detect format
    if p.suffix == ".gz":
        opener2 = gzip.open(path, "rt", encoding="utf-8")
    else:
        opener2 = open(path, "r", encoding="utf-8")

    with opener2 as f:
        content_start = f.read(2)

    if p.suffix == ".gz":
        opener3 = gzip.open(path, "rt", encoding="utf-8")
    else:
        opener3 = open(path, "r", encoding="utf-8")

    with opener3 as f:
        if content_start.strip().startswith("["):
            # JSON array format
            data = json.load(f)
            candidates = data if isinstance(data, list) else [data]
            print(f"    Loaded {len(candidates):,} candidates (JSON array format).")
        else:
            # JSONL format
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
                if (i + 1) % 10000 == 0:
                    print(f"    Loaded {i+1:,} candidates...")

    print(f"    Total: {len(candidates):,} candidates loaded.")
    return candidates


def run(candidates_path: str, output_path: str, top_n: int = 100):
    # Load
    candidates = load_candidates(candidates_path)

    # Score all
    print(f"[2/4] Scoring {len(candidates):,} candidates...")
    scored = []
    honeypot_count = 0

    for i, c in enumerate(candidates):
        result = score_candidate(c)
        if result["is_honeypot"]:
            honeypot_count += 1
        scored.append(result)
        if (i + 1) % 20000 == 0:
            print(f"    Scored {i+1:,}...")

    print(f"    Honeypots detected and excluded: {honeypot_count}")

    # Sort by score, exclude honeypots
    print(f"[3/4] Ranking top {top_n}...")
    valid = [s for s in scored if not s["is_honeypot"]]
    valid.sort(key=lambda x: x["total"], reverse=True)
    top = valid[:top_n]

    # Generate output
    print(f"[4/4] Writing {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank, entry in enumerate(top, 1):
            cid = entry["candidate_id"]
            score = entry["total"]
            candidate = entry.get("_candidate", {})
            reasoning = generate_reasoning(candidate, entry["signals"])
            writer.writerow([cid, rank, score, reasoning])

    print(f"\n✅ Done! Submission saved to: {output_path}")
    print(f"\n🏆 Top 5:")
    for i, entry in enumerate(top[:5], 1):
        p = entry.get("_candidate", {}).get("profile", {})
        print(f"   #{i} {p.get('anonymized_name','?')} | {p.get('current_title','?')} | Score: {entry['total']:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aradhya Recruiter Engine")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    parser.add_argument("--top", type=int, default=100, help="Number of candidates to rank")
    args = parser.parse_args()

    run(args.candidates, args.out, args.top)
