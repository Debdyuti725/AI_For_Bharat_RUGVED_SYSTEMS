"""
UserProfile — Data model for a citizen's profile.
Holds Tier 1 (universal) and Tier 2 (contextual) data.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime
import json


@dataclass
class UserProfile:
    """
    Represents a citizen's complete profile.

    Tier 1 fields are collected upfront (10 questions, everyone).
    Tier 2 fields are collected based on category interest.
    """

    # ─── Identity ────────────────────────────────────────────────────────────
    name: str = ""
    age: int = 0
    gender: str = ""           # male / female / other

    # ─── Location ────────────────────────────────────────────────────────────
    state: str = ""            # e.g. "Maharashtra", "Tamil Nadu"
    district: str = ""
    area_type: str = ""        # "rural" / "urban"

    # ─── Demographics ────────────────────────────────────────────────────────
    category: str = ""         # "general" / "sc" / "st" / "obc"
    disability: bool = False

    # ─── Education ───────────────────────────────────────────────────────────
    education_level: str = ""
    # Options: "below_10th", "10th", "12th", "graduate", "postgraduate", "doctorate"

    # ─── Employment ──────────────────────────────────────────────────────────
    employment_status: str = ""
    # Options: "student", "unemployed", "self_employed", "employed", "farmer", "retired"

    # ─── Income ──────────────────────────────────────────────────────────────
    annual_income: int = 0     # In INR
    monthly_income: int = 0    # In INR

    # ─── Family ──────────────────────────────────────────────────────────────
    family_size: int = 1
    dependents: int = 0
    bpl_card: bool = False     # Below Poverty Line card

    # ─── Tier 2 — Agriculture (unlocked if interested in farming schemes) ────
    owns_land: bool = False
    land_area_acres: float = 0.0
    farmer_type: str = ""      # "marginal" (<1 acre), "small" (1-2), "large" (>2)
    crop_type: str = ""        # "food_grain", "horticulture", "cash_crop", etc.
    has_kisan_credit_card: bool = False
    irrigation_type: str = ""  # "rainfed", "irrigated", "drip"

    # ─── Tier 2 — Education (unlocked if interested in education schemes) ────
    currently_enrolled: bool = False
    institution_type: str = ""     # "government", "private", "aided"
    course_level: str = ""         # "school", "ug", "pg", "diploma", "vocational"
    needs_scholarship: bool = False
    academic_percentage: float = 0.0

    # ─── Tier 2 — Housing (unlocked if interested in housing schemes) ────────
    owns_home: bool = False
    current_housing: str = ""   # "owned", "rented", "slum", "homeless"
    has_pucca_house: bool = False

    # ─── Tier 2 — Employment/Skill (unlocked if interested in jobs/training) ─
    is_job_seeker: bool = False
    skill_set: list = field(default_factory=list)
    wants_skill_training: bool = False
    has_mudra_loan: bool = False

    # ─── Tier 2 — Health (unlocked if interested in health schemes) ──────────
    has_ayushman_card: bool = False
    chronic_illness: bool = False
    disability_percentage: int = 0   # 0-100

    # ─── Meta ─────────────────────────────────────────────────────────────────
    completed_tier1: bool = False
    completed_tier2_categories: list = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    preferred_language: str = "en"

    # ─── Computed properties ─────────────────────────────────────────────────

    @property
    def tier1_completion_pct(self) -> int:
        """Returns 0-100 completion % of Tier 1 required fields."""
        required = [
            self.name, str(self.age) if self.age else "",
            self.gender, self.state, self.area_type,
            self.category, self.education_level,
            self.employment_status,
            str(self.annual_income) if self.annual_income else "",
            str(self.family_size)
        ]
        filled = sum(1 for v in required if v and v not in ("0", ""))
        return int((filled / len(required)) * 100)

    @property
    def is_farmer(self) -> bool:
        return self.employment_status == "farmer" or self.owns_land

    @property
    def is_student(self) -> bool:
        return self.employment_status == "student" or self.currently_enrolled

    @property
    def is_bpl(self) -> bool:
        return self.bpl_card or self.annual_income < 100000

    def to_dict(self) -> Dict[str, Any]:
        """Serialize profile to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
        }

    def to_json(self) -> str:
        """Serialize profile to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        """Deserialize profile from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def __repr__(self):
        return (
            f"UserProfile(name={self.name!r}, age={self.age}, state={self.state!r}, "
            f"category={self.category!r}, income=₹{self.annual_income:,}, "
            f"employment={self.employment_status!r}, tier1={self.tier1_completion_pct}%)"
        )
