"""
Questionnaire API — Callable functions to build and extend user profiles.

These are the public-facing functions that a frontend (web/mobile) or
CLI will call to:
  1. Get questions
  2. Submit answers
  3. Build / extend a profile
"""

from typing import Any, Dict, List, Tuple

from .models.profile import UserProfile
from .questions.tier1 import get_tier1_questions, validate_tier1_answer, Question
from .questions.tier2 import get_tier2_questions, TIER2_CATEGORIES


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1
# ─────────────────────────────────────────────────────────────────────────────

def get_onboarding_questions() -> List[Question]:
    """
    Return the 10 universal Tier 1 questions for first-time onboarding.
    Call this on first login to populate the questionnaire.
    """
    return get_tier1_questions()


def validate_answers(answers: Dict[str, Any]) -> Tuple[bool, Dict[str, str]]:
    """
    Validate a dict of Tier 1 answers.

    Args:
        answers: {question_key: value} dict from the frontend.

    Returns:
        (all_valid: bool, errors: {key: error_message})
    """
    questions = get_tier1_questions()
    errors: Dict[str, str] = {}

    for q in questions:
        value = answers.get(q.key)
        valid, msg = validate_tier1_answer(q, value)
        if not valid:
            errors[q.key] = msg

    return (len(errors) == 0), errors


def build_profile(answers: Dict[str, Any]) -> UserProfile:
    """
    Build a UserProfile from Tier 1 answers.

    Args:
        answers: dict of {question_key: answer_value} from Tier 1 questionnaire.
                 Keys match UserProfile fields directly.

    Returns:
        A UserProfile with Tier 1 fields populated.

    Example:
        profile = build_profile({
            "name": "Rajesh Kumar",
            "age": 35,
            "gender": "male",
            "state": "maharashtra",
            "area_type": "rural",
            "category": "obc",
            "education_level": "12th",
            "employment_status": "farmer",
            "annual_income": 120000,
            "family_size": 5,
        })
    """
    # Normalise string fields to lowercase
    normalised = {}
    str_fields = {"gender", "state", "area_type", "category",
                  "education_level", "employment_status"}
    for k, v in answers.items():
        if k in str_fields and isinstance(v, str):
            normalised[k] = v.lower().strip()
        else:
            normalised[k] = v

    # Derive monthly income from annual
    annual = normalised.get("annual_income", 0)
    normalised["monthly_income"] = int(annual / 12) if annual else 0

    # Mark BPL if income is very low
    if annual < 100000:
        normalised["bpl_card"] = True

    # Filter out keys that aren't UserProfile fields
    valid_fields = set(UserProfile.__dataclass_fields__.keys())
    filtered = {k: v for k, v in normalised.items() if k in valid_fields}

    profile = UserProfile(**filtered)
    profile.completed_tier1 = True
    return profile


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2
# ─────────────────────────────────────────────────────────────────────────────

def get_category_questions(category: str) -> List[Question]:
    """
    Return the 10 Tier 2 questions for a given category.

    Args:
        category: One of 'agriculture', 'education', 'housing', 'employment', 'health'

    Returns:
        List of 10 Question objects.
    """
    return get_tier2_questions(category)


def get_available_categories() -> Dict[str, str]:
    """
    Return the category codes and display names available for Tier 2.

    Returns:
        {'agriculture': '🌾 Agriculture & Farming', ...}
    """
    return TIER2_CATEGORIES


def extend_profile(
    profile: UserProfile,
    category: str,
    answers: Dict[str, Any],
) -> UserProfile:
    """
    Extend a profile with Tier 2 answers for a given category.

    This MUTATES the profile in-place (and also returns it for chaining).

    Args:
        profile:  An existing UserProfile (must have completed_tier1=True).
        category: The category the user selected (e.g. 'agriculture').
        answers:  Dict of {question_key: answer_value} from Tier 2 questionnaire.

    Returns:
        The updated UserProfile.

    Example:
        profile = extend_profile(profile, "agriculture", {
            "owns_land": True,
            "land_area_acres": 1.5,
            "farmer_type": "small",
            "crop_type": "food_grain",
            "irrigation_type": "canal",
            "has_kisan_credit_card": False,
        })
    """
    if not profile.completed_tier1:
        raise ValueError("Complete Tier 1 profile before extending with Tier 2 answers.")

    if category not in TIER2_CATEGORIES:
        raise ValueError(
            f"Unknown category '{category}'. "
            f"Valid categories: {list(TIER2_CATEGORIES.keys())}"
        )

    valid_fields = set(UserProfile.__dataclass_fields__.keys())

    for key, value in answers.items():
        if key in valid_fields:
            # Normalise strings
            if isinstance(value, str):
                value = value.lower().strip()
            setattr(profile, key, value)

    if category not in profile.completed_tier2_categories:
        profile.completed_tier2_categories.append(category)

    return profile


# ─────────────────────────────────────────────────────────────────────────────
# Suggested Tier 2 categories based on Tier 1 profile
# ─────────────────────────────────────────────────────────────────────────────

def suggest_tier2_categories(profile: UserProfile) -> List[Dict[str, str]]:
    """
    Based on Tier 1 answers, suggest the most relevant Tier 2 categories.

    Returns:
        List of {category, display_name, reason} dicts, ordered by relevance.
    """
    suggestions = []

    if profile.employment_status == "farmer" or profile.owns_land:
        suggestions.append({
            "category": "agriculture",
            "display_name": TIER2_CATEGORIES["agriculture"],
            "reason": "You identified as a farmer — unlock more farming schemes.",
        })

    if profile.employment_status == "student" or profile.education_level in ("10th", "12th", "graduate"):
        suggestions.append({
            "category": "education",
            "display_name": TIER2_CATEGORIES["education"],
            "reason": "Scholarship and education aid schemes may apply to you.",
        })

    if profile.annual_income < 300000 or not hasattr(profile, "owns_home") or not profile.owns_home:
        suggestions.append({
            "category": "housing",
            "display_name": TIER2_CATEGORIES["housing"],
            "reason": "Housing assistance (PMAY) may be available to you.",
        })

    if profile.employment_status in ("unemployed", "self_employed"):
        suggestions.append({
            "category": "employment",
            "display_name": TIER2_CATEGORIES["employment"],
            "reason": "Skill training, loans, and livelihood schemes are available.",
        })

    if profile.annual_income < 200000 or profile.bpl_card:
        suggestions.append({
            "category": "health",
            "display_name": TIER2_CATEGORIES["health"],
            "reason": "Free healthcare coverage (Ayushman Bharat) may be available.",
        })

    # Any remaining categories not yet suggested
    for cat, name in TIER2_CATEGORIES.items():
        if not any(s["category"] == cat for s in suggestions):
            suggestions.append({
                "category": cat,
                "display_name": name,
                "reason": "Explore additional schemes in this category.",
            })

    return suggestions
