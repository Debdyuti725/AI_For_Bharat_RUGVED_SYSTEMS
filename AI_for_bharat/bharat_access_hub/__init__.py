"""
Bharat Access Hub — Core Backend Package
Smart Profile Builder + Eligibility Matching Engine
"""

from .models.profile import UserProfile
from .questions.tier1 import get_tier1_questions
from .questions.tier2 import get_tier2_questions, TIER2_CATEGORIES
from .engine.eligibility import score_all_schemes, score_scheme
from .questionnaire import build_profile, extend_profile

__all__ = [
    "UserProfile",
    "get_tier1_questions",
    "get_tier2_questions",
    "TIER2_CATEGORIES",
    "score_all_schemes",
    "score_scheme",
    "build_profile",
    "extend_profile",
]
