"""
Demo Script — Quick walkthrough of the Smart Profile Builder + Eligibility Engine.

Run with:
    python -m bharat_access_hub.demo
  or:
    python demo.py   (from the 'AI for bharat' directory)
"""

import sys
import os

# Allow running from root directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bharat_access_hub.questionnaire import (
    build_profile, extend_profile,
    get_onboarding_questions, get_category_questions,
    suggest_tier2_categories, get_available_categories,
)
from bharat_access_hub.engine.eligibility import score_all_schemes, get_top_recommendations

# Plain ASCII output for Windows compatibility
GREEN  = ""
YELLOW = ""
CYAN   = ""
RED    = ""
BOLD   = ""
RESET  = ""
DIM    = ""

def divider(title=""):
    line = "=" * 65
    if title:
        print(f"\n{line}")
        print(f"  {title}")
        print(f"{line}")
    else:
        print(f"{'-' * 65}")


def print_score(match):
    score = match.eligibility_score
    if score >= 75:
        colour = GREEN
    elif score >= 40:
        colour = YELLOW
    else:
        colour = RED

    bar_filled = int(score / 10)
    bar = "█" * bar_filled + "░" * (10 - bar_filled)

    print(f"\n  {colour}{BOLD}{score:5.1f}%{RESET} {colour}{bar}{RESET}  {BOLD}{match.scheme_name}{RESET}")
    print(f"         Benefit : ₹{match.benefit_amount:,} ({match.benefit_type if hasattr(match, 'benefit_type') else ''})")
    print(f"         Category: {match.category.title()}")
    print(f"         Reason  : {match.reason}")
    if match.missing_criteria:
        missing_short = [m.split("—")[0].strip() for m in match.missing_criteria[:3]]
        print(f"         Missing : {DIM}{', '.join(missing_short)}{RESET}")


def run_demo():

    # ══════════════════════════════════════════════════════════════════════════
    # DEMO 1 — Farmer Profile
    # ══════════════════════════════════════════════════════════════════════════
    divider("DEMO 1: Rajesh Kumar — Small Farmer from Maharashtra")

    tier1_answers_farmer = {
        "name":              "Rajesh Kumar",
        "age":               35,
        "gender":            "male",
        "state":             "maharashtra",
        "area_type":         "rural",
        "category":          "obc",
        "education_level":   "10th",
        "employment_status": "farmer",
        "annual_income":     120000,
        "family_size":       5,
    }

    farmer_profile = build_profile(tier1_answers_farmer)
    print(f"\n  Profile built: {farmer_profile}")
    print(f"  Tier 1 completion: {farmer_profile.tier1_completion_pct}%")

    # Tier 1 only results
    print(f"\n  {BOLD}--- Tier 1 Recommendations (top 5) ---{RESET}")
    matches_t1 = get_top_recommendations(farmer_profile, top_n=5)
    for m in matches_t1:
        print_score(m)

    # Extend with Tier 2 agriculture answers
    tier2_agri_answers = {
        "owns_land":          True,
        "land_area_acres":    1.5,
        "farmer_type":        "small",
        "crop_type":          "food_grain",
        "irrigation_type":    "canal",
        "has_kisan_credit_card": False,
        "district":           "Pune",
    }
    farmer_profile = extend_profile(farmer_profile, "agriculture", tier2_agri_answers)
    print(f"\n  {BOLD}--- After Tier 2 (Agriculture) —{RESET}")
    matches_t2 = get_top_recommendations(farmer_profile, top_n=5)
    for m in matches_t2:
        print_score(m)


    # ══════════════════════════════════════════════════════════════════════════
    # DEMO 2 — Student Profile (OBC, Scholarship seeker)
    # ══════════════════════════════════════════════════════════════════════════
    divider("DEMO 2: Priya Sharma — OBC Student from UP")

    student_profile = build_profile({
        "name":              "Priya Sharma",
        "age":               20,
        "gender":            "female",
        "state":             "uttar_pradesh",
        "area_type":         "rural",
        "category":          "obc",
        "education_level":   "12th",
        "employment_status": "student",
        "annual_income":     180000,
        "family_size":       6,
    })
    print(f"\n  Profile built: {student_profile}")

    matches = get_top_recommendations(student_profile, top_n=5)
    print(f"\n  {BOLD}--- Top Recommendations ---{RESET}")
    for m in matches:
        print_score(m)

    # Tier 2 — Education
    student_profile = extend_profile(student_profile, "education", {
        "currently_enrolled":  True,
        "course_level":        "ug",
        "institution_type":    "government",
        "needs_scholarship":   True,
        "academic_percentage": 72.0,
    })
    print(f"\n  {BOLD}--- After Tier 2 (Education) ---{RESET}")
    for m in get_top_recommendations(student_profile, top_n=5):
        print_score(m)


    # ══════════════════════════════════════════════════════════════════════════
    # DEMO 3 — Urban High-Income (few matches, edge case)
    # ══════════════════════════════════════════════════════════════════════════
    divider("DEMO 3: Amit Mehta — Urban Employed, High Income")

    urban_profile = build_profile({
        "name":              "Amit Mehta",
        "age":               32,
        "gender":            "male",
        "state":             "delhi",
        "area_type":         "urban",
        "category":          "general",
        "education_level":   "postgraduate",
        "employment_status": "employed",
        "annual_income":     1200000,
        "family_size":       3,
    })
    print(f"\n  Profile built: {urban_profile}")

    matches = score_all_schemes(urban_profile, min_score=0.0, top_n=5)
    print(f"\n  {BOLD}--- Top Recommendations (expect low scores) ---{RESET}")
    for m in matches:
        print_score(m)


    # ══════════════════════════════════════════════════════════════════════════
    # DEMO 4 — Category Suggestion
    # ══════════════════════════════════════════════════════════════════════════
    divider("DEMO 4: Smart Tier 2 Category Suggestions")

    for profile, label in [
        (farmer_profile, "Rajesh (Farmer)"),
        (student_profile, "Priya (Student)"),
        (urban_profile,   "Amit (Urban employed)"),
    ]:
        print(f"\n  {BOLD}{label}:{RESET}")
        suggestions = suggest_tier2_categories(profile)
        for s in suggestions[:3]:
            print(f"   → {s['display_name']} — {DIM}{s['reason']}{RESET}")


    # ══════════════════════════════════════════════════════════════════════════
    # SHOW TIER 1 QUESTIONS (for integration reference)
    # ══════════════════════════════════════════════════════════════════════════
    divider("Tier 1 Questions (for reference)")
    for i, q in enumerate(get_onboarding_questions(), 1):
        opts = ""
        if q.options:
            opts = f"  [{', '.join(o['value'] for o in q.options[:3])}{'...' if len(q.options) > 3 else ''}]"
        print(f"  {i:2}. [{q.qtype:7}] {q.text}{opts}")

    divider("Done!")
    print("\n  Smart Profile Builder + Eligibility Engine is working correctly.\n")


if __name__ == "__main__":
    run_demo()
