"""
Eligibility Engine — Weighted scoring algorithm.

Computes a 0–100 eligibility score for each scheme against a user's profile.
Produces a ranked list of SchemeMatch objects sorted by composite ranking score.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..models.profile import UserProfile
from ..data.schemes import get_all_schemes, get_schemes_by_category


# ─── Education hierarchy for comparison ──────────────────────────────────────

EDUCATION_HIERARCHY = [
    "below_10th", "10th", "12th", "graduate", "postgraduate", "doctorate"
]


# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class SchemeMatch:
    """Result of running eligibility scoring for one scheme."""
    scheme_id: str
    scheme_name: str
    category: str
    eligibility_score: float        # 0–100
    ranking_score: float            # Composite: eligibility × benefit × urgency
    benefit_amount: int
    matched_criteria: List[str]
    missing_criteria: List[str]
    partially_matched: List[str]
    reason: str                     # Human-readable summary
    application_url: str
    required_documents: List[str]

    def __repr__(self):
        bar = "█" * int(self.eligibility_score / 10) + "░" * (10 - int(self.eligibility_score / 10))
        return (
            f"[{self.eligibility_score:5.1f}%] {bar}  {self.scheme_name}\n"
            f"          Benefit: ₹{self.benefit_amount:,} | "
            f"Matched: {self.matched_criteria} | "
            f"Missing: {self.missing_criteria}"
        )


# ─── Criterion evaluators ─────────────────────────────────────────────────────

def _eval_age(profile: UserProfile, criteria: dict) -> tuple[bool, str]:
    age_min = criteria.get("age_min", 0)
    age_max = criteria.get("age_max", 150)
    if age_min <= profile.age <= age_max:
        return True, f"age {profile.age} in [{age_min}–{age_max}]"
    return False, f"age {profile.age} outside [{age_min}–{age_max}]"


def _eval_income(profile: UserProfile, criteria: dict) -> tuple[bool, str]:
    income_max = criteria.get("income_max")
    if income_max is None:
        return True, "no income ceiling"
    if profile.annual_income <= income_max:
        return True, f"income ₹{profile.annual_income:,} ≤ ₹{income_max:,}"
    return False, f"income ₹{profile.annual_income:,} exceeds limit ₹{income_max:,}"


def _eval_state(profile: UserProfile, criteria: dict) -> tuple[bool, str]:
    allowed = criteria.get("states", ["all"])
    if "all" in allowed:
        return True, "open to all states"
    if profile.state in allowed:
        return True, f"state {profile.state!r} is eligible"
    return False, f"state {profile.state!r} not in eligible states"


def _eval_area_type(profile: UserProfile, criteria: dict) -> tuple[bool, str]:
    allowed = criteria.get("area_type", ["rural", "urban"])
    if not allowed or set(allowed) == {"rural", "urban"}:
        return True, "rural and urban both allowed"
    if profile.area_type in allowed:
        return True, f"area type '{profile.area_type}' matches"
    return False, f"area type '{profile.area_type}' not in {allowed}"


def _eval_category(profile: UserProfile, criteria: dict) -> tuple[bool, str]:
    allowed = criteria.get("category", [])
    if not allowed:
        return True, "open to all categories"
    if profile.category in allowed:
        return True, f"category '{profile.category}' matches"
    return False, f"category '{profile.category}' not in required {allowed}"


def _eval_education(profile: UserProfile, criteria: dict) -> tuple[bool, str]:
    required_levels = criteria.get("education_level", [])
    if not required_levels:
        return True, "no education requirement"
    try:
        user_idx = EDUCATION_HIERARCHY.index(profile.education_level)
    except ValueError:
        return False, f"unknown education level '{profile.education_level}'"
    for req in required_levels:
        try:
            req_idx = EDUCATION_HIERARCHY.index(req)
            if user_idx >= req_idx:
                return True, f"education '{profile.education_level}' meets '{req}'"
        except ValueError:
            continue
    return False, f"education '{profile.education_level}' does not meet any of {required_levels}"


def _eval_employment(profile: UserProfile, criteria: dict) -> tuple[bool, str]:
    allowed = criteria.get("employment_status", [])
    if not allowed:
        return True, "open to all employment types"
    if profile.employment_status in allowed:
        return True, f"employment '{profile.employment_status}' matches"
    return False, f"employment '{profile.employment_status}' not in {allowed}"


def _eval_gender(profile: UserProfile, criteria: dict) -> tuple[bool, str]:
    allowed = criteria.get("gender", [])
    if not allowed:
        return True, "open to all genders"
    if profile.gender in allowed:
        return True, f"gender '{profile.gender}' matches"
    return False, f"gender '{profile.gender}' not required ({allowed})"


def _eval_owns_land(profile: UserProfile, criteria: dict) -> tuple[bool, str]:
    required = criteria.get("owns_land")
    if required is None:
        return True, "land ownership not required"
    if required and profile.owns_land:
        return True, "owns land ✓"
    if not required:
        return True, "land ownership not required"
    return False, "land ownership required but not present"


def _eval_bpl(profile: UserProfile, criteria: dict) -> tuple[bool, str]:
    required = criteria.get("bpl_card")
    if required is None:
        return True, "BPL card not required"
    # BPL: must have card OR income below ₹1 lakh
    has_bpl = profile.bpl_card or profile.annual_income < 100000
    if required and has_bpl:
        return True, "BPL status confirmed"
    if not required:
        return True, "BPL not required"
    return False, "requires BPL status (card or income < ₹1L)"


def _eval_owns_home(profile: UserProfile, criteria: dict) -> tuple[bool, str]:
    # owns_home: False means scheme is for people who DON'T own a home
    required_home = criteria.get("owns_home")
    if required_home is None:
        return True, "home ownership not checked"
    if required_home is False and not profile.owns_home:
        return True, "does not own home ✓ (required for this scheme)"
    if required_home is False and profile.owns_home:
        return False, "already owns a home (scheme for non-homeowners)"
    return True, "home ownership condition met"


def _eval_enrolled(profile: UserProfile, criteria: dict) -> tuple[bool, str]:
    req = criteria.get("currently_enrolled")
    if req is None:
        return True, "enrollment not required"
    if req and profile.currently_enrolled:
        return True, "currently enrolled ✓"
    if req and not profile.currently_enrolled:
        return False, "must be currently enrolled"
    return True, ""


def _eval_academic_pct(profile: UserProfile, criteria: dict) -> tuple[bool, str]:
    req_min = criteria.get("academic_percentage_min")
    if req_min is None:
        return True, "no academic % requirement"
    if profile.academic_percentage >= req_min:
        return True, f"academic % {profile.academic_percentage} ≥ {req_min}"
    return False, f"academic % {profile.academic_percentage} < required {req_min}"


def _eval_chronic_illness(profile: UserProfile, criteria: dict) -> tuple[bool, str]:
    req = criteria.get("chronic_illness")
    if req is None:
        return True, "chronic illness not checked"
    if req and profile.chronic_illness:
        return True, "chronic illness confirmed"
    if req and not profile.chronic_illness:
        return False, "requires chronic illness declaration"
    return True, ""


def _eval_pucca_house(profile: UserProfile, criteria: dict) -> tuple[bool, str]:
    req = criteria.get("has_pucca_house")
    if req is None:
        return True, "house type not required"
    if req is False and not profile.has_pucca_house:
        return True, "does not have pucca house ✓"
    if req is False and profile.has_pucca_house:
        return False, "already has pucca house (scheme for kutcha/no house)"
    return True, ""


# ─── Criterion weight table ───────────────────────────────────────────────────

CRITERION_WEIGHTS: Dict[str, int] = {
    "income":           25,
    "area_type":        15,
    "employment":       15,
    "category":         10,
    "education":        10,
    "age":              10,
    "state":             5,
    "bpl":               5,
    "owns_land":         5,
    "owns_home":         5,
    "gender":            5,
    "enrolled":          3,
    "pucca_house":       3,
    "academic_pct":      3,
    "chronic_illness":   2,
}

# Map criterion name → evaluator function
CRITERION_EVALUATORS = {
    "income":           _eval_income,
    "area_type":        _eval_area_type,
    "employment":       _eval_employment,
    "category":         _eval_category,
    "education":        _eval_education,
    "age":              _eval_age,
    "state":            _eval_state,
    "bpl":              _eval_bpl,
    "owns_land":        _eval_owns_land,
    "owns_home":        _eval_owns_home,
    "gender":           _eval_gender,
    "enrolled":         _eval_enrolled,
    "pucca_house":      _eval_pucca_house,
    "academic_pct":     _eval_academic_pct,
    "chronic_illness":  _eval_chronic_illness,
}


# ─── Core scoring function ────────────────────────────────────────────────────

def score_scheme(profile: UserProfile, scheme: Dict[str, Any]) -> SchemeMatch:
    """
    Compute eligibility score for a single scheme.

    Returns a SchemeMatch with score (0–100), matched/missing criteria, and reasons.
    """
    criteria = scheme.get("eligibility_criteria", {})

    # Which criterion evaluators are relevant for this scheme?
    relevant = {
        "income":           "income_max" in criteria,
        "area_type":        "area_type" in criteria,
        "employment":       "employment_status" in criteria,
        "category":         "category" in criteria,
        "education":        "education_level" in criteria,
        "age":              "age_min" in criteria or "age_max" in criteria,
        "state":            "states" in criteria,
        "bpl":              "bpl_card" in criteria,
        "owns_land":        "owns_land" in criteria,
        "owns_home":        "owns_home" in criteria,
        "gender":           "gender" in criteria,
        "enrolled":         "currently_enrolled" in criteria,
        "pucca_house":      "has_pucca_house" in criteria,
        "academic_pct":     "academic_percentage_min" in criteria,
        "chronic_illness":  "chronic_illness" in criteria,
    }

    total_weight = 0
    matched_weight = 0
    matched_criteria = []
    missing_criteria = []

    for crit_name, is_relevant in relevant.items():
        if not is_relevant:
            continue

        weight = CRITERION_WEIGHTS.get(crit_name, 5)
        total_weight += weight

        evaluator = CRITERION_EVALUATORS[crit_name]
        passed, detail = evaluator(profile, criteria)

        if passed:
            matched_weight += weight
            matched_criteria.append(f"{crit_name} ({detail})")
        else:
            missing_criteria.append(f"{crit_name} — {detail}")

    # Calculate base eligibility score
    if total_weight == 0:
        base_score = 100.0  # No restrictions = fully eligible
    else:
        base_score = (matched_weight / total_weight) * 100

    # ─── Compute urgency factor ───────────────────────────────────────────────
    deadline = scheme.get("deadline")
    urgency_factor = 1.0
    if deadline:
        try:
            days_remaining = (datetime.fromisoformat(deadline) - datetime.utcnow()).days
            if days_remaining <= 7:
                urgency_factor = 1.5
            elif days_remaining <= 30:
                urgency_factor = 1.2
            elif days_remaining <= 90:
                urgency_factor = 1.0
            else:
                urgency_factor = 0.8
        except (ValueError, TypeError):
            urgency_factor = 1.0

    # ─── Compute ranking score ────────────────────────────────────────────────
    benefit = scheme.get("benefit_amount", 0)
    normalized_benefit = min(benefit / 1000000, 1.0)   # Normalise to [0,1] against ₹10L max

    ranking_score = (
        base_score * 0.5
        + normalized_benefit * 100 * 0.3
        + urgency_factor * 100 * 0.2
    )

    # ─── Build human-readable reason ─────────────────────────────────────────
    if base_score >= 80:
        reason = f"Strong match! You meet {len(matched_criteria)} key criteria."
    elif base_score >= 50:
        reason = f"Partial match. {len(missing_criteria)} criteria need attention."
    elif base_score > 0:
        reason = f"Low eligibility. Major criteria unmet: {', '.join(missing_criteria[:2])}."
    else:
        reason = "Not eligible based on current profile."

    return SchemeMatch(
        scheme_id=scheme["scheme_id"],
        scheme_name=scheme["name"],
        category=scheme["category"],
        eligibility_score=round(base_score, 1),
        ranking_score=round(ranking_score, 2),
        benefit_amount=benefit,
        matched_criteria=matched_criteria,
        missing_criteria=missing_criteria,
        partially_matched=[],
        reason=reason,
        application_url=scheme.get("application_url", ""),
        required_documents=scheme.get("required_documents", []),
    )


def score_all_schemes(
    profile: UserProfile,
    category_filter: Optional[str] = None,
    min_score: float = 0.0,
    top_n: Optional[int] = None,
) -> List[SchemeMatch]:
    """
    Score all schemes against a profile and return ranked results.

    Args:
        profile:         The user's profile.
        category_filter: Optional — filter to one category (e.g. 'agriculture').
        min_score:       Only return schemes with eligibility_score >= this.
        top_n:           Return only the top N results.

    Returns:
        List of SchemeMatch objects sorted by ranking_score descending.
    """
    if category_filter:
        schemes = get_schemes_by_category(category_filter)
    else:
        schemes = get_all_schemes()

    results = [score_scheme(profile, scheme) for scheme in schemes]

    # Filter by minimum score
    results = [r for r in results if r.eligibility_score >= min_score]

    # Sort by ranking score (desc)
    results.sort(key=lambda r: r.ranking_score, reverse=True)

    if top_n:
        results = results[:top_n]

    return results


def get_top_recommendations(
    profile: UserProfile,
    top_n: int = 5,
    min_score: float = 30.0,
) -> List[SchemeMatch]:
    """
    Get the top N recommended schemes for a user (with score >= 30 by default).
    Convenience wrapper around score_all_schemes.
    """
    return score_all_schemes(profile, min_score=min_score, top_n=top_n)
