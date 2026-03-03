"""
Multi-Language Support — Language definitions, detection, and translation helpers.

Supported languages:
  en  English
  hi  Hindi
  kn  Kannada
  te  Telugu
  ta  Tamil
  bn  Bengali
  mr  Marathi
  gu  Gujarati
  ml  Malayalam
  pa  Punjabi
"""

from typing import Dict, List, Optional

# ─── Supported Languages ─────────────────────────────────────────────────────

SUPPORTED_LANGUAGES = {
    "en": {"name": "English",   "native": "English",   "flag": "EN"},
    "hi": {"name": "Hindi",     "native": "हिन्दी",     "flag": "HI"},
    "kn": {"name": "Kannada",   "native": "ಕನ್ನಡ",      "flag": "KN"},
    "te": {"name": "Telugu",    "native": "తెలుగు",     "flag": "TE"},
    "ta": {"name": "Tamil",     "native": "தமிழ்",      "flag": "TA"},
    "bn": {"name": "Bengali",   "native": "বাংলা",      "flag": "BN"},
    "mr": {"name": "Marathi",   "native": "मराठी",      "flag": "MR"},
    "gu": {"name": "Gujarati",  "native": "ગુજરાતી",    "flag": "GU"},
    "ml": {"name": "Malayalam", "native": "മലയാളം",    "flag": "ML"},
    "pa": {"name": "Punjabi",   "native": "ਪੰਜਾਬੀ",     "flag": "PA"},
}

DEFAULT_LANGUAGE = "en"


def get_language_list() -> List[Dict]:
    """Get list of all supported languages."""
    return [
        {"code": code, **info}
        for code, info in SUPPORTED_LANGUAGES.items()
    ]


def get_language_name(code: str) -> str:
    """Get language name from code."""
    lang = SUPPORTED_LANGUAGES.get(code)
    return lang["name"] if lang else "English"


def get_language_native_name(code: str) -> str:
    """Get native language name from code."""
    lang = SUPPORTED_LANGUAGES.get(code)
    return lang["native"] if lang else "English"


def is_supported(code: str) -> bool:
    """Check if a language code is supported."""
    return code in SUPPORTED_LANGUAGES


# ─── Language instruction for LLM ────────────────────────────────────────────

LANGUAGE_INSTRUCTIONS = {
    "en": "Respond in English.",
    "hi": "Respond in Hindi (हिन्दी में जवाब दें). Use Devanagari script.",
    "kn": "Respond in Kannada (ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರಿಸಿ). Use Kannada script.",
    "te": "Respond in Telugu (తెలుగులో జవాబు ఇవ్వండి). Use Telugu script.",
    "ta": "Respond in Tamil (தமிழில் பதிலளிக்கவும்). Use Tamil script.",
    "bn": "Respond in Bengali (বাংলায় উত্তর দিন). Use Bengali script.",
    "mr": "Respond in Marathi (मराठीत उत्तर द्या). Use Devanagari script.",
    "gu": "Respond in Gujarati (ગુજરાતીમાં જવાબ આપો). Use Gujarati script.",
    "ml": "Respond in Malayalam (മലയാളത്തിൽ ഉത്തരം നൽകുക). Use Malayalam script.",
    "pa": "Respond in Punjabi (ਪੰਜਾਬੀ ਵਿੱਚ ਜਵਾਬ ਦਿਓ). Use Gurmukhi script.",
}


def get_language_instruction(code: str) -> str:
    """Get the LLM instruction for responding in a specific language."""
    return LANGUAGE_INSTRUCTIONS.get(code, LANGUAGE_INSTRUCTIONS["en"])


# ─── UI Labels (for frontend) ────────────────────────────────────────────────

UI_LABELS = {
    "en": {
        "welcome": "Welcome to Bharat Access Hub",
        "login": "Sign In",
        "signup": "Sign Up",
        "profile": "My Profile",
        "dashboard": "Dashboard",
        "chat": "AI Chat",
        "schemes": "Explore Schemes",
        "search": "Search...",
        "logout": "Logout",
        "score": "Eligibility Score",
        "apply": "How to Apply",
        "documents": "Required Documents",
        "high_match": "High Match",
        "partial_match": "Partial Match",
        "low_match": "Low Match",
    },
    "hi": {
        "welcome": "भारत एक्सेस हब में आपका स्वागत है",
        "login": "लॉग इन करें",
        "signup": "साइन अप करें",
        "profile": "मेरी प्रोफ़ाइल",
        "dashboard": "डैशबोर्ड",
        "chat": "AI चैट",
        "schemes": "योजनाएं खोजें",
        "search": "खोजें...",
        "logout": "लॉग आउट",
        "score": "पात्रता स्कोर",
        "apply": "आवेदन कैसे करें",
        "documents": "आवश्यक दस्तावेज़",
        "high_match": "उच्च मिलान",
        "partial_match": "आंशिक मिलान",
        "low_match": "कम मिलान",
    },
    "kn": {
        "welcome": "ಭಾರತ್ ಆಕ್ಸೆಸ್ ಹಬ್‌ಗೆ ಸ್ವಾಗತ",
        "login": "ಲಾಗಿನ್",
        "signup": "ಸೈನ್ ಅಪ್",
        "profile": "ನನ್ನ ಪ್ರೊಫೈಲ್",
        "dashboard": "ಡ್ಯಾಶ್‌ಬೋರ್ಡ್",
        "chat": "AI ಚಾಟ್",
        "schemes": "ಯೋಜನೆಗಳನ್ನು ಅನ್ವೇಷಿಸಿ",
        "search": "ಹುಡುಕಿ...",
        "logout": "ಲಾಗ್ ಔಟ್",
        "score": "ಅರ್ಹತೆ ಸ್ಕೋರ್",
        "apply": "ಹೇಗೆ ಅರ್ಜಿ ಸಲ್ಲಿಸುವುದು",
        "documents": "ಅಗತ್ಯ ದಾಖಲೆಗಳು",
        "high_match": "ಹೆಚ್ಚಿನ ಹೊಂದಾಣಿಕೆ",
        "partial_match": "ಭಾಗಶಃ ಹೊಂದಾಣಿಕೆ",
        "low_match": "ಕಡಿಮೆ ಹೊಂದಾಣಿಕೆ",
    },
    "te": {
        "welcome": "భారత్ యాక్సెస్ హబ్‌కు స్వాగతం",
        "login": "లాగిన్",
        "signup": "సైన్ అప్",
        "profile": "నా ప్రొఫైల్",
        "dashboard": "డాష్‌బోర్డ్",
        "chat": "AI చాట్",
        "schemes": "పథకాలను అన్వేషించండి",
        "search": "వెతకండి...",
        "logout": "లాగ్ అవుట్",
        "score": "అర్హత స్కోర్",
        "apply": "దరఖాస్తు ఎలా చేయాలి",
        "documents": "అవసరమైన పత్రాలు",
        "high_match": "అధిక సరిపోలిక",
        "partial_match": "పాక్షిక సరిపోలిక",
        "low_match": "తక్కువ సరిపోలిక",
    },
}


def get_ui_labels(lang_code: str) -> Dict[str, str]:
    """Get UI labels for a language. Falls back to English."""
    return UI_LABELS.get(lang_code, UI_LABELS["en"])
