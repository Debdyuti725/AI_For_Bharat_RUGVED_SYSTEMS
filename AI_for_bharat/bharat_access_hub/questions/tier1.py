"""
Tier 1 Questions — 10 universal questions asked of every user at first login.
These provide enough signal to compute baseline eligibility across all schemes.
"""

from dataclasses import dataclass
from typing import List, Optional, Any


@dataclass
class Question:
    """A single question in the questionnaire."""
    key: str                    # Maps to UserProfile field
    text: str                   # The question text (English)
    text_hi: str                # Hindi translation
    qtype: str                  # "text", "number", "choice", "bool"
    options: Optional[List[dict]] = None  # For "choice" type
    validation: Optional[dict] = None    # min/max for numbers, etc.
    help_text: str = ""
    required: bool = True

    def display(self) -> str:
        return f"[{self.key}] {self.text}"


# ─────────────────────────────────────────────────────────────────────────────
# TIER 1 — 10 Universal Questions (asked of everyone)
# ─────────────────────────────────────────────────────────────────────────────

TIER1_QUESTIONS: List[Question] = [

    Question(
        key="name",
        text="What is your full name?",
        text_hi="आपका पूरा नाम क्या है?",
        qtype="text",
        help_text="Used for personalisation only.",
    ),

    Question(
        key="age",
        text="How old are you?",
        text_hi="आपकी उम्र कितनी है?",
        qtype="number",
        validation={"min": 14, "max": 100},
        help_text="Age in years. Many schemes have age limits.",
    ),

    Question(
        key="gender",
        text="What is your gender?",
        text_hi="आपका लिंग क्या है?",
        qtype="choice",
        options=[
            {"value": "male",   "label": "Male",   "label_hi": "पुरुष"},
            {"value": "female", "label": "Female", "label_hi": "महिला"},
            {"value": "other",  "label": "Other",  "label_hi": "अन्य"},
        ],
    ),

    Question(
        key="state",
        text="Which state do you live in?",
        text_hi="आप किस राज्य में रहते हैं?",
        qtype="choice",
        options=[
            {"value": "andhra_pradesh",     "label": "Andhra Pradesh"},
            {"value": "assam",              "label": "Assam"},
            {"value": "bihar",              "label": "Bihar"},
            {"value": "chhattisgarh",       "label": "Chhattisgarh"},
            {"value": "delhi",              "label": "Delhi"},
            {"value": "goa",                "label": "Goa"},
            {"value": "gujarat",            "label": "Gujarat"},
            {"value": "haryana",            "label": "Haryana"},
            {"value": "himachal_pradesh",   "label": "Himachal Pradesh"},
            {"value": "jharkhand",          "label": "Jharkhand"},
            {"value": "karnataka",          "label": "Karnataka"},
            {"value": "kerala",             "label": "Kerala"},
            {"value": "madhya_pradesh",     "label": "Madhya Pradesh"},
            {"value": "maharashtra",        "label": "Maharashtra"},
            {"value": "odisha",             "label": "Odisha"},
            {"value": "punjab",             "label": "Punjab"},
            {"value": "rajasthan",          "label": "Rajasthan"},
            {"value": "tamil_nadu",         "label": "Tamil Nadu"},
            {"value": "telangana",          "label": "Telangana"},
            {"value": "uttar_pradesh",      "label": "Uttar Pradesh"},
            {"value": "uttarakhand",        "label": "Uttarakhand"},
            {"value": "west_bengal",        "label": "West Bengal"},
            {"value": "other",              "label": "Other / UT"},
        ],
        help_text="Many state-specific schemes exist.",
    ),

    Question(
        key="area_type",
        text="Do you live in a Rural or Urban area?",
        text_hi="आप ग्रामीण या शहरी क्षेत्र में रहते हैं?",
        qtype="choice",
        options=[
            {"value": "rural", "label": "Rural (Village / Gram Panchayat)", "label_hi": "ग्रामीण"},
            {"value": "urban", "label": "Urban (City / Town)",              "label_hi": "शहरी"},
        ],
        help_text="MGNREGS, PM-KISAN and many schemes are rural-only.",
    ),

    Question(
        key="category",
        text="What is your social category?",
        text_hi="आपकी सामाजिक श्रेणी क्या है?",
        qtype="choice",
        options=[
            {"value": "general", "label": "General",                 "label_hi": "सामान्य"},
            {"value": "obc",     "label": "OBC (Other Backward Class)", "label_hi": "अन्य पिछड़ा वर्ग"},
            {"value": "sc",      "label": "SC (Scheduled Caste)",     "label_hi": "अनुसूचित जाति"},
            {"value": "st",      "label": "ST (Scheduled Tribe)",     "label_hi": "अनुसूचित जनजाति"},
        ],
        help_text="SC/ST/OBC reserved categories get access to additional schemes.",
    ),

    Question(
        key="education_level",
        text="What is your highest education level?",
        text_hi="आपकी सबसे उच्च शिक्षा योग्यता क्या है?",
        qtype="choice",
        options=[
            {"value": "below_10th",   "label": "Below 10th",        "label_hi": "10वीं से कम"},
            {"value": "10th",         "label": "10th Pass",          "label_hi": "10वीं पास"},
            {"value": "12th",         "label": "12th Pass",          "label_hi": "12वीं पास"},
            {"value": "graduate",     "label": "Graduate (UG)",      "label_hi": "स्नातक"},
            {"value": "postgraduate", "label": "Post-Graduate (PG)", "label_hi": "स्नातकोत्तर"},
            {"value": "doctorate",    "label": "Doctorate / PhD",    "label_hi": "डॉक्टरेट"},
        ],
    ),

    Question(
        key="employment_status",
        text="What best describes your current employment?",
        text_hi="आपकी वर्तमान रोजगार स्थिति क्या है?",
        qtype="choice",
        options=[
            {"value": "student",       "label": "Student",             "label_hi": "छात्र"},
            {"value": "unemployed",    "label": "Unemployed / Job Seeker", "label_hi": "बेरोजगार"},
            {"value": "farmer",        "label": "Farmer / Agricultural Worker", "label_hi": "किसान"},
            {"value": "self_employed", "label": "Self-Employed / Business", "label_hi": "स्वरोजगार"},
            {"value": "employed",      "label": "Salaried / Employed", "label_hi": "नौकरीपेशा"},
            {"value": "retired",       "label": "Retired",             "label_hi": "सेवानिवृत्त"},
        ],
        help_text="This helps match employment-specific schemes (MGNREGS, Mudra, etc.)",
    ),

    Question(
        key="annual_income",
        text="What is your approximate annual household income (in ₹)?",
        text_hi="आपकी वार्षिक पारिवारिक आय कितनी है (₹ में)?",
        qtype="number",
        validation={"min": 0, "max": 10000000},
        help_text="Income threshold is the #1 eligibility filter across all schemes. Enter 0 if no income.",
    ),

    Question(
        key="family_size",
        text="How many members are in your family (including yourself)?",
        text_hi="आपके परिवार में कितने सदस्य हैं (आप सहित)?",
        qtype="number",
        validation={"min": 1, "max": 20},
        help_text="Used to calculate per-capita income and BPL status.",
    ),
]


def get_tier1_questions() -> List[Question]:
    """Return the list of 10 universal Tier 1 questions."""
    return TIER1_QUESTIONS


def validate_tier1_answer(question: Question, value: Any) -> tuple[bool, str]:
    """
    Validate a single answer against the question's rules.
    Returns (is_valid, error_message).
    """
    if question.required and (value is None or value == "" or value == 0):
        if question.key != "annual_income":  # 0 income is valid
            return False, f"'{question.text}' is required."

    if question.qtype == "number" and question.validation:
        try:
            val = int(value)
            if "min" in question.validation and val < question.validation["min"]:
                return False, f"Must be at least {question.validation['min']}."
            if "max" in question.validation and val > question.validation["max"]:
                return False, f"Must be at most {question.validation['max']}."
        except (ValueError, TypeError):
            return False, "Please enter a valid number."

    if question.qtype == "choice" and question.options:
        valid_values = [o["value"] for o in question.options]
        if value not in valid_values:
            return False, f"Invalid choice. Valid options: {valid_values}"

    return True, ""
