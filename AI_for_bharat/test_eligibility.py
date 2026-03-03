"""
Test suite for the Eligibility Engine.
Run with: python test_eligibility.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bharat_access_hub.questionnaire import build_profile, extend_profile
from bharat_access_hub.engine.eligibility import score_scheme, score_all_schemes
from bharat_access_hub.data.schemes import get_scheme_by_id, get_all_schemes
from bharat_access_hub.questions.tier1 import get_tier1_questions
from bharat_access_hub.questions.tier2 import get_tier2_questions, TIER2_CATEGORIES

PASS = "PASS"
FAIL = "FAIL"
results = []

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((status, label, detail))
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))


print("\n" + "="*60)
print("  Bharat Access Hub — Eligibility Engine Test Suite")
print("="*60)

# ─── T1: Questionnaire structure ──────────────────────────────────────────────
print("\n[Section 1] Questionnaire Structure")

t1_qs = get_tier1_questions()
check("Tier 1 has exactly 10 questions", len(t1_qs) == 10, f"got {len(t1_qs)}")

for cat in TIER2_CATEGORIES:
    t2_qs = get_tier2_questions(cat)
    check(f"Tier 2 '{cat}' has 10 questions", len(t2_qs) == 10, f"got {len(t2_qs)}")

check("Scheme DB has >= 15 schemes", len(get_all_schemes()) >= 15, f"got {len(get_all_schemes())}")


# ─── T2: Profile building ─────────────────────────────────────────────────────
print("\n[Section 2] Profile Building")

farmer = build_profile({
    "name": "Rajesh Kumar",
    "age": 35,
    "gender": "male",
    "state": "maharashtra",
    "area_type": "rural",
    "category": "obc",
    "education_level": "10th",
    "employment_status": "farmer",
    "annual_income": 120000,
    "family_size": 5,
})
check("Farmer profile tier1_completion_pct = 100%", farmer.tier1_completion_pct == 100)
check("Annual income auto-derives monthly", farmer.monthly_income == 10000, f"got {farmer.monthly_income}")
# BPL auto-mark test: use a very low income profile
very_poor = build_profile({
    "name": "Test", "age": 30, "gender": "male", "state": "bihar",
    "area_type": "rural", "category": "sc", "education_level": "below_10th",
    "employment_status": "unemployed", "annual_income": 60000, "family_size": 6,
})
check("Sub-1L income auto-marks BPL", very_poor.bpl_card == True)
check("Above-1L income does NOT auto-mark BPL", farmer.bpl_card == False, f"farmer income=120k")
check("completed_tier1 flag is set", farmer.completed_tier1 == True)

# Extend with Tier 2 agriculture
farmer = extend_profile(farmer, "agriculture", {
    "owns_land": True,
    "land_area_acres": 1.5,
    "farmer_type": "small",
    "crop_type": "food_grain",
    "irrigation_type": "canal",
})
check("Tier 2 extension sets owns_land", farmer.owns_land == True)
check("Tier 2 category tracked", "agriculture" in farmer.completed_tier2_categories)


# ─── T3: Eligibility scoring ──────────────────────────────────────────────────
print("\n[Section 3] Eligibility Scores")

# PM-KISAN: rural farmer, land owned — should score high
pm_kisan = score_scheme(farmer, get_scheme_by_id("PM-KISAN"))
check(
    "PM-KISAN: rural farmer scores > 80%",
    pm_kisan.eligibility_score > 80,
    f"got {pm_kisan.eligibility_score}%"
)

# NSP Scholarship: OBC student, income < 2.5L
student = build_profile({
    "name": "Priya Sharma",
    "age": 20,
    "gender": "female",
    "state": "uttar_pradesh",
    "area_type": "rural",
    "category": "obc",
    "education_level": "12th",
    "employment_status": "student",
    "annual_income": 180000,
    "family_size": 6,
})
student = extend_profile(student, "education", {
    "currently_enrolled": True,
    "academic_percentage": 72.0,
})
nsp = score_scheme(student, get_scheme_by_id("NSP-POST-MATRIC"))
check(
    "NSP: OBC student scores > 70%",
    nsp.eligibility_score > 70,
    f"got {nsp.eligibility_score}%"
)

# Urban high-income should score low on rural farming schemes
urban = build_profile({
    "name": "Amit Mehta",
    "age": 32,
    "gender": "male",
    "state": "delhi",
    "area_type": "urban",
    "category": "general",
    "education_level": "postgraduate",
    "employment_status": "employed",
    "annual_income": 1200000,
    "family_size": 3,
})
pm_urban = score_scheme(urban, get_scheme_by_id("PM-KISAN"))
check(
    "PM-KISAN: urban employed scores < 55% (rural+farmer criteria fail)",
    pm_urban.eligibility_score < 55,
    f"got {pm_urban.eligibility_score}%"
)

# Ayushman Bharat: BPL should score high
bpl_profile = build_profile({
    "name": "Sunita Devi",
    "age": 45,
    "gender": "female",
    "state": "bihar",
    "area_type": "rural",
    "category": "sc",
    "education_level": "below_10th",
    "employment_status": "unemployed",
    "annual_income": 60000,
    "family_size": 7,
})
ayushman = score_scheme(bpl_profile, get_scheme_by_id("AYUSHMAN-BHARAT"))
check(
    "Ayushman Bharat: BPL family scores > 70%",
    ayushman.eligibility_score > 70,
    f"got {ayushman.eligibility_score}%"
)

# PMAY-G: Rural, no pucca house
pmay = score_scheme(bpl_profile, get_scheme_by_id("PMAY-G"))
check(
    "PMAY-G: rural BPL without pucca house scores > 60%",
    pmay.eligibility_score > 60,
    f"got {pmay.eligibility_score}%"
)


# ─── T4: Ranking ─────────────────────────────────────────────────────────────
print("\n[Section 4] Ranking & Sorting")

top5 = score_all_schemes(farmer, top_n=5)
check("score_all_schemes returns 5 results with top_n=5", len(top5) == 5)
check(
    "Results are sorted by ranking_score DESC",
    all(top5[i].ranking_score >= top5[i+1].ranking_score for i in range(len(top5)-1)),
)

high_only = score_all_schemes(farmer, min_score=60.0)
check(
    "min_score filter works — all results >= 60%",
    all(r.eligibility_score >= 60 for r in high_only),
    f"got {len(high_only)} results"
)

agri_only = score_all_schemes(farmer, category_filter="agriculture", top_n=10)
check(
    "category_filter returns only agriculture schemes",
    all(r.category == "agriculture" for r in agri_only),
)


# ─── Summary ─────────────────────────────────────────────────────────────────
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
print(f"\n{'='*60}")
print(f"  Results: {passed} passed, {failed} failed out of {len(results)} tests")
print(f"{'='*60}\n")

if failed > 0:
    print("  FAILED TESTS:")
    for status, label, detail in results:
        if status == FAIL:
            print(f"    - {label} ({detail})")
    sys.exit(1)
else:
    print("  ALL TESTS PASSED")
