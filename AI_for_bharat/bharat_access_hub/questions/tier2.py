"""
Tier 2 Questions — 10 contextual questions per category.
Unlocked after Tier 1 when user expresses interest in a specific scheme category.

Categories:
  - agriculture   🌾 Farming schemes
  - education     🎓 Scholarships / study
  - housing       🏠 Housing schemes
  - employment    💼 Jobs / skill training
  - health        🏥 Health schemes
"""

from .tier1 import Question
from typing import List, Dict


TIER2_CATEGORIES = {
    "agriculture": "🌾 Agriculture & Farming",
    "education":   "🎓 Education & Scholarships",
    "housing":     "🏠 Housing & Infrastructure",
    "employment":  "💼 Employment & Skill Training",
    "health":      "🏥 Health & Wellness",
}


# ─────────────────────────────────────────────────────────────────────────────
# AGRICULTURE — 10 Questions
# ─────────────────────────────────────────────────────────────────────────────

AGRICULTURE_QUESTIONS: List[Question] = [

    Question(
        key="owns_land",
        text="Do you own or cultivate agricultural land?",
        text_hi="क्या आपके पास कृषि भूमि है?",
        qtype="bool",
        help_text="Required for PM-KISAN, Kisan Credit Card, and most farming schemes.",
    ),

    Question(
        key="land_area_acres",
        text="How much land do you own/cultivate? (in acres)",
        text_hi="आपके पास कितनी ज़मीन है? (एकड़ में)",
        qtype="number",
        validation={"min": 0, "max": 500},
        help_text="Marginal farmers (<1 acre) get higher priority under most schemes.",
    ),

    Question(
        key="farmer_type",
        text="Which type of farmer are you?",
        text_hi="आप किस प्रकार के किसान हैं?",
        qtype="choice",
        options=[
            {"value": "marginal", "label": "Marginal Farmer (< 1 acre)",     "label_hi": "सीमांत किसान"},
            {"value": "small",    "label": "Small Farmer (1–2 acres)",        "label_hi": "लघु किसान"},
            {"value": "medium",   "label": "Medium Farmer (2–5 acres)",       "label_hi": "मध्यम किसान"},
            {"value": "large",    "label": "Large Farmer (> 5 acres)",        "label_hi": "बड़े किसान"},
            {"value": "tenant",   "label": "Tenant / Sharecropper",           "label_hi": "किरायेदार किसान"},
        ],
    ),

    Question(
        key="crop_type",
        text="What type of crops do you primarily grow?",
        text_hi="आप मुख्यतः कौन सी फसल उगाते हैं?",
        qtype="choice",
        options=[
            {"value": "food_grain",   "label": "Food Grains (Wheat, Rice, Maize)", "label_hi": "खाद्यान्न"},
            {"value": "horticulture", "label": "Horticulture (Fruits, Vegetables)", "label_hi": "बागवानी"},
            {"value": "cash_crop",    "label": "Cash Crops (Cotton, Sugarcane)", "label_hi": "नकदी फसल"},
            {"value": "pulses",       "label": "Pulses & Oilseeds",            "label_hi": "दालें"},
            {"value": "mixed",        "label": "Mixed / Multiple crops",       "label_hi": "मिश्रित"},
        ],
    ),

    Question(
        key="irrigation_type",
        text="What is your primary source of irrigation?",
        text_hi="आपकी मुख्य सिंचाई का स्रोत क्या है?",
        qtype="choice",
        options=[
            {"value": "rainfed",  "label": "Rainfed (no irrigation)",     "label_hi": "वर्षा आधारित"},
            {"value": "canal",    "label": "Canal / River",               "label_hi": "नहर / नदी"},
            {"value": "borewell", "label": "Borewell / Tubewell",         "label_hi": "बोरवेल"},
            {"value": "drip",     "label": "Drip / Sprinkler",            "label_hi": "ड्रिप सिंचाई"},
            {"value": "pond",     "label": "Pond / Tank",                 "label_hi": "तालाब"},
        ],
    ),

    Question(
        key="has_kisan_credit_card",
        text="Do you have a Kisan Credit Card (KCC)?",
        text_hi="क्या आपके पास किसान क्रेडिट कार्ड (KCC) है?",
        qtype="bool",
        help_text="KCC is required for subsidised credit and some loan waiver schemes.",
    ),

    Question(
        key="bpl_card",
        text="Do you have a BPL (Below Poverty Line) ration card?",
        text_hi="क्या आपके पास बीपीएल राशन कार्ड है?",
        qtype="bool",
    ),

    Question(
        key="disability",
        text="Do you have any physical or mental disability?",
        text_hi="क्या आपको कोई शारीरिक या मानसिक विकलांगता है?",
        qtype="bool",
        help_text="Disabled farmers may get additional support under PMFBY.",
    ),

    Question(
        key="has_ayushman_card",
        text="Are you enrolled under Ayushman Bharat (PM-JAY)?",
        text_hi="क्या आप आयुष्मान भारत में नामांकित हैं?",
        qtype="bool",
    ),

    Question(
        key="district",
        text="Which district do you live in?",
        text_hi="आप किस जिले में रहते हैं?",
        qtype="text",
        help_text="Some district-specific crop / drought relief schemes require this.",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# EDUCATION — 10 Questions
# ─────────────────────────────────────────────────────────────────────────────

EDUCATION_QUESTIONS: List[Question] = [

    Question(
        key="currently_enrolled",
        text="Are you currently enrolled in an educational institution?",
        text_hi="क्या आप वर्तमान में किसी शिक्षण संस्थान में नामांकित हैं?",
        qtype="bool",
    ),

    Question(
        key="course_level",
        text="What level of course are you enrolled in / applying for?",
        text_hi="आप किस स्तर के पाठ्यक्रम में नामांकित हैं?",
        qtype="choice",
        options=[
            {"value": "school",     "label": "School (Class 1–12)",     "label_hi": "स्कूल"},
            {"value": "diploma",    "label": "Diploma / ITI",            "label_hi": "डिप्लोमा"},
            {"value": "ug",         "label": "Under-Graduate (UG)",      "label_hi": "स्नातक"},
            {"value": "pg",         "label": "Post-Graduate (PG)",       "label_hi": "स्नातकोत्तर"},
            {"value": "vocational", "label": "Vocational Training",      "label_hi": "व्यावसायिक प्रशिक्षण"},
            {"value": "other",      "label": "Other",                    "label_hi": "अन्य"},
        ],
    ),

    Question(
        key="institution_type",
        text="Is your institution government or private?",
        text_hi="आपका संस्थान सरकारी है या निजी?",
        qtype="choice",
        options=[
            {"value": "government", "label": "Government",          "label_hi": "सरकारी"},
            {"value": "aided",      "label": "Government-Aided",    "label_hi": "सरकारी सहायता प्राप्त"},
            {"value": "private",    "label": "Private (unaided)",   "label_hi": "निजी"},
        ],
    ),

    Question(
        key="needs_scholarship",
        text="Are you looking for a scholarship or financial aid?",
        text_hi="क्या आप छात्रवृत्ति या वित्तीय सहायता चाहते हैं?",
        qtype="bool",
    ),

    Question(
        key="academic_percentage",
        text="What is your last academic score / percentage?",
        text_hi="आपका अंतिम अकादमिक प्रतिशत क्या है?",
        qtype="number",
        validation={"min": 0, "max": 100},
        help_text="Some merit-based scholarships require minimum 50–60%.",
    ),

    Question(
        key="bpl_card",
        text="Do you have a BPL ration card?",
        text_hi="क्या आपके पास बीपीएल राशन कार्ड है?",
        qtype="bool",
        help_text="BPL students get priority for need-based scholarships.",
    ),

    Question(
        key="disability",
        text="Do you have any disability?",
        text_hi="क्या आपको कोई विकलांगता है?",
        qtype="bool",
        help_text="Specially-abled students qualify for higher scholarship amounts.",
    ),

    Question(
        key="disability_percentage",
        text="If yes, what is the disability percentage? (0 if none)",
        text_hi="यदि हाँ, तो विकलांगता प्रतिशत कितना है?",
        qtype="number",
        validation={"min": 0, "max": 100},
        required=False,
    ),

    Question(
        key="annual_income",
        text="Confirm your annual family income (₹):",
        text_hi="अपनी वार्षिक पारिवारिक आय की पुष्टि करें (₹):",
        qtype="number",
        validation={"min": 0, "max": 10000000},
        help_text="Most scholarships have income ceiling of ₹2.5–8 lakhs per year.",
    ),

    Question(
        key="district",
        text="Which district are you from?",
        text_hi="आप किस जिले से हैं?",
        qtype="text",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# HOUSING — 10 Questions
# ─────────────────────────────────────────────────────────────────────────────

HOUSING_QUESTIONS: List[Question] = [

    Question(
        key="owns_home",
        text="Do you currently own a house?",
        text_hi="क्या आपके पास अपना घर है?",
        qtype="bool",
    ),

    Question(
        key="current_housing",
        text="What is your current housing situation?",
        text_hi="आपकी वर्तमान आवास स्थिति क्या है?",
        qtype="choice",
        options=[
            {"value": "owned",    "label": "Own house (pucca)",          "label_hi": "खुद का पक्का घर"},
            {"value": "kutcha",   "label": "Own house (kutcha / mud)",    "label_hi": "कच्चा घर"},
            {"value": "rented",   "label": "Rented accommodation",       "label_hi": "किराये का घर"},
            {"value": "slum",     "label": "Slum / Informal Settlement",  "label_hi": "झुग्गी-झोपड़ी"},
            {"value": "homeless", "label": "No permanent housing",        "label_hi": "बेघर"},
        ],
    ),

    Question(
        key="has_pucca_house",
        text="Is your current house pucca (concrete/brick)?",
        text_hi="क्या आपका वर्तमान घर पक्का है?",
        qtype="bool",
    ),

    Question(
        key="bpl_card",
        text="Do you hold a BPL ration card?",
        text_hi="क्या आपके पास बीपीएल राशन कार्ड है?",
        qtype="bool",
        help_text="BPL households are prioritised under PMAY.",
    ),

    Question(
        key="owns_land",
        text="Do you own a plot/land to build a house?",
        text_hi="क्या आपके पास घर बनाने के लिए ज़मीन है?",
        qtype="bool",
        help_text="PMAY Gramin requires land ownership.",
    ),

    Question(
        key="annual_income",
        text="Confirm your annual household income (₹):",
        text_hi="अपनी वार्षिक पारिवारिक आय की पुष्टि करें (₹):",
        qtype="number",
        validation={"min": 0, "max": 10000000},
        help_text="PMAY Urban: EWS < ₹3L, LIG < ₹6L, MIG-I < ₹12L.",
    ),

    Question(
        key="family_size",
        text="Confirm your family size:",
        text_hi="अपने परिवार के आकार की पुष्टि करें:",
        qtype="number",
        validation={"min": 1, "max": 20},
    ),

    Question(
        key="disability",
        text="Is there anyone with disability in your household?",
        text_hi="क्या आपके घर में कोई विकलांग व्यक्ति है?",
        qtype="bool",
    ),

    Question(
        key="district",
        text="Which district is your property located in?",
        text_hi="आपकी संपत्ति किस जिले में है?",
        qtype="text",
    ),

    Question(
        key="has_ayushman_card",
        text="Are you registered under Ayushman Bharat?",
        text_hi="क्या आप आयुष्मान भारत में पंजीकृत हैं?",
        qtype="bool",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# EMPLOYMENT / SKILL — 10 Questions
# ─────────────────────────────────────────────────────────────────────────────

EMPLOYMENT_QUESTIONS: List[Question] = [

    Question(
        key="is_job_seeker",
        text="Are you actively looking for a job?",
        text_hi="क्या आप सक्रिय रूप से नौकरी की तलाश में हैं?",
        qtype="bool",
    ),

    Question(
        key="wants_skill_training",
        text="Are you interested in free skill training / vocational courses?",
        text_hi="क्या आप मुफ्त कौशल प्रशिक्षण / व्यावसायिक पाठ्यक्रम में रुचि रखते हैं?",
        qtype="bool",
        help_text="Pradhan Mantri Kaushal Vikas Yojana (PMKVY) offers free certified training.",
    ),

    Question(
        key="has_mudra_loan",
        text="Have you taken a MUDRA / business loan before?",
        text_hi="क्या आपने पहले मुद्रा / व्यवसाय ऋण लिया है?",
        qtype="bool",
    ),

    Question(
        key="bpl_card",
        text="Do you hold a BPL ration card?",
        text_hi="क्या आपके पास बीपीएल राशन कार्ड है?",
        qtype="bool",
    ),

    Question(
        key="annual_income",
        text="Confirm annual household income (₹):",
        text_hi="वार्षिक पारिवारिक आय की पुष्टि करें (₹):",
        qtype="number",
        validation={"min": 0, "max": 10000000},
    ),

    Question(
        key="disability",
        text="Do you have any disability?",
        text_hi="क्या आपको कोई विकलांगता है?",
        qtype="bool",
        help_text="Specially-abled persons get priority under Stand-Up India and other schemes.",
    ),

    Question(
        key="owns_home",
        text="Do you own a home or business premises?",
        text_hi="क्या आपके पास घर या व्यावसायिक परिसर है?",
        qtype="bool",
        help_text="Relevant for PM SVANidhi and Stand Up India.",
    ),

    Question(
        key="district",
        text="Which district are you in?",
        text_hi="आप किस जिले में हैं?",
        qtype="text",
    ),

    Question(
        key="education_level",
        text="Confirm your education level:",
        text_hi="अपनी शिक्षा स्तर की पुष्टि करें:",
        qtype="choice",
        options=[
            {"value": "below_10th",   "label": "Below 10th"},
            {"value": "10th",         "label": "10th Pass"},
            {"value": "12th",         "label": "12th Pass"},
            {"value": "graduate",     "label": "Graduate"},
            {"value": "postgraduate", "label": "Post-Graduate"},
        ],
    ),

    Question(
        key="employment_status",
        text="Are you a street vendor / hawker?",
        text_hi="क्या आप एक स्ट्रीट वेंडर / फेरीवाले हैं?",
        qtype="bool",
        help_text="PM SVANidhi provides working capital loans to street vendors.",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH — 10 Questions
# ─────────────────────────────────────────────────────────────────────────────

HEALTH_QUESTIONS: List[Question] = [

    Question(
        key="has_ayushman_card",
        text="Are you enrolled under Ayushman Bharat PM-JAY?",
        text_hi="क्या आप आयुष्मान भारत PM-JAY में नामांकित हैं?",
        qtype="bool",
        help_text="PM-JAY provides ₹5 lakh health coverage per family.",
    ),

    Question(
        key="bpl_card",
        text="Do you hold a BPL / SECC class ration card?",
        text_hi="क्या आपके पास बीपीएल राशन कार्ड है?",
        qtype="bool",
        help_text="BPL families automatically qualify for PM-JAY.",
    ),

    Question(
        key="annual_income",
        text="Confirm annual household income (₹):",
        text_hi="वार्षिक पारिवारिक आय की पुष्टि करें (₹):",
        qtype="number",
        validation={"min": 0, "max": 10000000},
    ),

    Question(
        key="disability",
        text="Does any member of your family have a disability?",
        text_hi="क्या आपके परिवार के किसी सदस्य को विकलांगता है?",
        qtype="bool",
    ),

    Question(
        key="disability_percentage",
        text="If yes, what is the disability percentage? (0 if none)",
        text_hi="यदि हाँ, तो विकलांगता प्रतिशत क्या है?",
        qtype="number",
        validation={"min": 0, "max": 100},
        required=False,
    ),

    Question(
        key="chronic_illness",
        text="Does any family member suffer from chronic illness (cancer, kidney disease, etc.)?",
        text_hi="क्या कोई सदस्य गंभीर बीमारी से पीड़ित है?",
        qtype="bool",
        help_text="Rashtriya Arogya Nidhi provides up to ₹15 lakh for life-threatening illnesses.",
    ),

    Question(
        key="family_size",
        text="Confirm your family size:",
        text_hi="अपने परिवार के आकार की पुष्टि करें:",
        qtype="number",
        validation={"min": 1, "max": 20},
    ),

    Question(
        key="district",
        text="Which district are you in?",
        text_hi="आप किस जिले में हैं?",
        qtype="text",
    ),

    Question(
        key="current_housing",
        text="What is your housing situation?",
        text_hi="आपकी आवास स्थिति क्या है?",
        qtype="choice",
        options=[
            {"value": "owned",    "label": "Own house"},
            {"value": "kutcha",   "label": "Kutcha / temporary house"},
            {"value": "rented",   "label": "Rented"},
            {"value": "slum",     "label": "Slum"},
            {"value": "homeless", "label": "No home"},
        ],
    ),

    Question(
        key="gender",
        text="Confirm gender (for maternal health schemes):",
        text_hi="लिंग की पुष्टि करें (मातृत्व स्वास्थ्य योजनाओं के लिए):",
        qtype="choice",
        options=[
            {"value": "male",   "label": "Male"},
            {"value": "female", "label": "Female"},
            {"value": "other",  "label": "Other"},
        ],
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Lookup table
# ─────────────────────────────────────────────────────────────────────────────

_TIER2_MAP: Dict[str, List[Question]] = {
    "agriculture": AGRICULTURE_QUESTIONS,
    "education":   EDUCATION_QUESTIONS,
    "housing":     HOUSING_QUESTIONS,
    "employment":  EMPLOYMENT_QUESTIONS,
    "health":      HEALTH_QUESTIONS,
}


def get_tier2_questions(category: str) -> List[Question]:
    """
    Return the 10 contextual questions for the given category.

    Args:
        category: One of 'agriculture', 'education', 'housing', 'employment', 'health'

    Returns:
        List of 10 Question objects.

    Raises:
        ValueError: If category is not recognised.
    """
    if category not in _TIER2_MAP:
        raise ValueError(
            f"Unknown category '{category}'. "
            f"Valid options: {list(_TIER2_MAP.keys())}"
        )
    return _TIER2_MAP[category]
