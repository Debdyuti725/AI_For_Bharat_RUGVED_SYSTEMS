"""
FastAPI Backend — REST API for Bharat Access Hub.

Features:
  - Auth: Sign up / Sign in with JWT tokens
  - Profile: Build & extend with Tier 1/2 questions (saved to SQLite)
  - Eligibility: Score all schemes for authenticated user
  - Chat: RAG chatbot with conversation history

Run with:
    set GROQ_API_KEY=gsk_your_key
    uvicorn bharat_access_hub.api:app --reload --port 8000
"""

import os
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

# Auto-load .env file
def _load_dotenv():
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())

_load_dotenv()

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .questionnaire import (
    build_profile, extend_profile,
    get_onboarding_questions, get_category_questions,
    get_available_categories, suggest_tier2_categories,
)
from .engine.eligibility import score_all_schemes
from .models.profile import UserProfile
from .database import (
    init_db, save_profile, get_profile,
    save_chat_message, get_chat_history, clear_chat_history,
)
from .auth import signup, login, get_current_user
from .languages import get_language_list, get_ui_labels, is_supported, SUPPORTED_LANGUAGES
from .data.schemes import get_all_schemes, get_scheme_by_id


# ─── Pydantic models ─────────────────────────────────────────────────────────

class SignUpRequest(BaseModel):
    email: str = Field(..., example="vedh@example.com")
    password: str = Field(..., min_length=6, example="mypassword123")
    name: str = Field("", example="Vedh Sontha")


class LoginRequest(BaseModel):
    email: str = Field(..., example="vedh@example.com")
    password: str = Field(..., example="mypassword123")


class Tier1Answers(BaseModel):
    name: str = Field(..., example="Rajesh Kumar")
    age: int = Field(..., ge=14, le=100, example=35)
    gender: str = Field(..., example="male")
    state: str = Field(..., example="maharashtra")
    area_type: str = Field(..., example="rural")
    category: str = Field(..., example="obc")
    education_level: str = Field(..., example="10th")
    employment_status: str = Field(..., example="farmer")
    annual_income: int = Field(..., ge=0, example=120000)
    family_size: int = Field(..., ge=1, le=20, example=5)


class Tier2ExtendRequest(BaseModel):
    category: str = Field(..., example="agriculture")
    answers: Dict[str, Any] = Field(..., description="Tier 2 answers")


class ChatRequest(BaseModel):
    message: str = Field(..., example="What schemes am I eligible for?")
    language: str = Field("en", example="hi", description="Response language code (en, hi, kn, te, ta, bn, mr, gu, ml, pa)")


class QuestionResponse(BaseModel):
    key: str
    text: str
    text_hi: str
    qtype: str
    options: Optional[List[dict]] = None
    validation: Optional[dict] = None
    help_text: str = ""
    required: bool = True


class SchemeMatchResponse(BaseModel):
    scheme_id: str
    scheme_name: str
    category: str
    eligibility_score: float
    ranking_score: float
    benefit_amount: int
    matched_criteria: List[str]
    missing_criteria: List[str]
    reason: str
    application_url: str
    required_documents: List[str]


# ─── Auth dependency ──────────────────────────────────────────────────────────

def get_authenticated_user(authorization: str = Header(None)):
    """Extract and validate JWT from Authorization header."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header. Use 'Bearer <token>'.")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization format. Use 'Bearer <token>'.")

    token = parts[1]
    try:
        user = get_current_user(token)
        return user
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


# ─── App setup ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    print("[Startup] Database initialized.")
    # Vector store loads lazily on first use (embedding model takes time)
    import threading
    def _bg_build():
        try:
            from .engine.vector_store import build_vector_store
            build_vector_store()
            print("[Background] Vector store ready.")
        except Exception as e:
            print(f"[Background] Vector store skipped: {e}")
    threading.Thread(target=_bg_build, daemon=True).start()
    print("[Startup] Server ready! Vector store loading in background...")
    yield


app = FastAPI(
    title="Bharat Access Hub API",
    description="AI-powered government scheme discovery. Sign up, build your profile, get matched schemes, chat with AI.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files (CSS, JS)
import pathlib
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

FRONTEND_DIR = pathlib.Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ─── Root — serves the website ───────────────────────────────────────────────

@app.get("/", tags=["Frontend"], include_in_schema=False)
def serve_frontend():
    """Serve the main website."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Bharat Access Hub API is running", "docs": "/docs"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "version": "1.0.0"}


# ═════════════════════════════════════════════════════════════════════════════
# AUTH ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/api/auth/signup", tags=["Auth"])
def api_signup(req: SignUpRequest):
    """Create a new account. Returns a JWT token."""
    try:
        result = signup(req.email, req.password, req.name)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/auth/login", tags=["Auth"])
def api_login(req: LoginRequest):
    """Sign in with email + password. Returns a JWT token."""
    try:
        result = login(req.email, req.password)
        return result
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


@app.get("/api/auth/me", tags=["Auth"])
def api_me(user=Depends(get_authenticated_user)):
    """Get current user info (requires token)."""
    return {
        "user_id": user["id"],
        "email": user["email"],
        "name": user["name"],
        "created_at": user["created_at"],
    }


# ═════════════════════════════════════════════════════════════════════════════
# QUESTIONS (public — no auth needed)
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/questions/tier1", response_model=List[QuestionResponse], tags=["Questions"])
def get_tier1():
    """Get the 10 universal Tier 1 questions."""
    return [
        QuestionResponse(
            key=q.key, text=q.text, text_hi=q.text_hi, qtype=q.qtype,
            options=q.options, validation=q.validation,
            help_text=q.help_text, required=q.required,
        )
        for q in get_onboarding_questions()
    ]


@app.get("/api/questions/tier2/{category}", response_model=List[QuestionResponse], tags=["Questions"])
def get_tier2(category: str):
    """Get the 10 Tier 2 questions for a category."""
    try:
        return [
            QuestionResponse(
                key=q.key, text=q.text, text_hi=q.text_hi, qtype=q.qtype,
                options=q.options, validation=q.validation,
                help_text=q.help_text, required=q.required,
            )
            for q in get_category_questions(category)
        ]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/categories", tags=["Questions"])
def list_categories():
    """Get available Tier 2 categories."""
    return get_available_categories()


# ═════════════════════════════════════════════════════════════════════════════
# PROFILE (auth required — saved to DB)
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/api/profile", tags=["Profile"])
def create_profile(answers: Tier1Answers, user=Depends(get_authenticated_user)):
    """Build profile from Tier 1 answers and save to database."""
    profile = build_profile(answers.model_dump())

    # Save to DB
    save_profile(
        user_id=user["id"],
        profile_dict=profile.to_dict(),
        tier1_complete=True,
        tier2_categories=[],
    )

    return {
        "message": "Profile created and saved!",
        "profile": profile.to_dict(),
        "tier1_completion": profile.tier1_completion_pct,
        "suggested_categories": suggest_tier2_categories(profile),
    }


@app.post("/api/profile/extend", tags=["Profile"])
def extend_user_profile(req: Tier2ExtendRequest, user=Depends(get_authenticated_user)):
    """Extend profile with Tier 2 answers and save to database."""
    # Load existing profile from DB
    db_profile = get_profile(user["id"])
    if not db_profile:
        raise HTTPException(status_code=400, detail="Complete Tier 1 profile first.")

    profile = UserProfile.from_dict(db_profile["profile_json"])
    profile.completed_tier1 = True

    try:
        updated = extend_profile(profile, req.category, req.answers)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Save updated profile to DB
    save_profile(
        user_id=user["id"],
        profile_dict=updated.to_dict(),
        tier1_complete=True,
        tier2_categories=updated.completed_tier2_categories,
    )

    return {
        "message": f"Profile extended with {req.category} answers!",
        "profile": updated.to_dict(),
        "completed_categories": updated.completed_tier2_categories,
    }


@app.get("/api/profile", tags=["Profile"])
def get_my_profile(user=Depends(get_authenticated_user)):
    """Get the current user's saved profile."""
    db_profile = get_profile(user["id"])
    if not db_profile:
        return {"message": "No profile yet. Complete Tier 1 questionnaire first.", "profile": None}
    return {
        "profile": db_profile["profile_json"],
        "tier1_complete": bool(db_profile["tier1_complete"]),
        "tier2_categories": db_profile["tier2_categories"],
    }


# ═════════════════════════════════════════════════════════════════════════════
# ELIGIBILITY (auth required — uses saved profile)
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/eligibility", tags=["Eligibility"])
def get_eligibility(
    category_filter: Optional[str] = None,
    min_score: float = 0.0,
    top_n: Optional[int] = None,
    user=Depends(get_authenticated_user),
):
    """Get eligibility scores using the saved profile."""
    db_profile = get_profile(user["id"])
    if not db_profile:
        raise HTTPException(status_code=400, detail="Complete your profile first.")

    profile = UserProfile.from_dict(db_profile["profile_json"])
    results = score_all_schemes(profile, category_filter=category_filter, min_score=min_score, top_n=top_n)

    return [
        SchemeMatchResponse(
            scheme_id=r.scheme_id, scheme_name=r.scheme_name,
            category=r.category, eligibility_score=r.eligibility_score,
            ranking_score=r.ranking_score, benefit_amount=r.benefit_amount,
            matched_criteria=r.matched_criteria, missing_criteria=r.missing_criteria,
            reason=r.reason, application_url=r.application_url,
            required_documents=r.required_documents,
        )
        for r in results
    ]


# ═════════════════════════════════════════════════════════════════════════════
# CHAT (auth required — uses saved profile + saves history)
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/api/chat", tags=["Chat"])
def chat(req: ChatRequest, user=Depends(get_authenticated_user)):
    """Chat with the AI assistant. Uses your saved profile for personalized answers."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY not configured on server. Set it as an environment variable."
        )

    try:
        from .engine.chatbot import ChatBot

        # Load profile from DB
        db_profile = get_profile(user["id"])
        profile = None
        if db_profile:
            profile = UserProfile.from_dict(db_profile["profile_json"])

        # Create or reuse chatbot
        lang = getattr(req, 'language', 'en') or 'en'
        bot = ChatBot(groq_api_key=api_key, language=lang)
        if profile:
            bot.set_profile(profile)

        # Load existing chat history from DB
        history = get_chat_history(user["id"], limit=10)
        bot.chat_history = [
            {"user": h["message"], "assistant": ""} if h["role"] == "user"
            else {"user": "", "assistant": h["message"]}
            for h in history
        ]

        # Get response
        response = bot.chat(req.message)

        # Save to DB
        save_chat_message(user["id"], "user", req.message)
        save_chat_message(user["id"], "assistant", response)

        return {
            "response": response,
            "user_name": user["name"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.get("/api/chat/history", tags=["Chat"])
def get_history(limit: int = 20, user=Depends(get_authenticated_user)):
    """Get conversation history."""
    history = get_chat_history(user["id"], limit=limit)
    return {"history": history}


@app.delete("/api/chat/history", tags=["Chat"])
def delete_history(user=Depends(get_authenticated_user)):
    """Clear conversation history."""
    clear_chat_history(user["id"])
    return {"message": "Chat history cleared."}


# =============================================================================
# LANGUAGE
# =============================================================================

@app.get("/api/languages", tags=["Language"])
def list_languages():
    """Get all supported languages."""
    return get_language_list()


@app.get("/api/languages/{code}/labels", tags=["Language"])
def get_labels(code: str):
    """Get UI labels for a language."""
    if not is_supported(code):
        raise HTTPException(status_code=400, detail=f"Language '{code}' not supported.")
    return get_ui_labels(code)


# =============================================================================
# SCHEME EXPLORER
# =============================================================================

@app.get("/api/schemes", tags=["Schemes"])
def list_schemes(category: Optional[str] = None):
    """Browse all schemes, optionally filter by category."""
    schemes = get_all_schemes()
    if category:
        schemes = [s for s in schemes if s["category"] == category.lower()]
    return [
        {
            "scheme_id": s["scheme_id"],
            "name": s["name"],
            "category": s["category"],
            "benefit_amount": s.get("benefit_amount", 0),
            "description": s.get("description", "")[:200],
            "application_url": s.get("application_url", ""),
        }
        for s in schemes
    ]


@app.get("/api/schemes/{scheme_id}", tags=["Schemes"])
def get_scheme_detail(scheme_id: str):
    """Get full details of a specific scheme."""
    try:
        scheme = get_scheme_by_id(scheme_id)
        return scheme
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Scheme '{scheme_id}' not found.")


@app.get("/api/schemes/search/{query}", tags=["Schemes"])
def search_schemes_endpoint(query: str, k: int = 5):
    """Semantic search for schemes using natural language."""
    try:
        from .engine.vector_store import search_schemes
        results = search_schemes(query, k=k)
        return [
            {
                "scheme_id": r.metadata.get("scheme_id", ""),
                "section": r.metadata.get("section", ""),
                "content": r.page_content[:300],
                "score": r.metadata.get("score", 0),
            }
            for r in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


# =============================================================================
# DOCUMENT INTELLIGENCE
# =============================================================================

from fastapi import UploadFile, File, Form

@app.post("/api/document/upload", tags=["Document"])
async def upload_document(
    file: UploadFile = File(...),
    language: str = Form("en"),
    user=Depends(get_authenticated_user),
):
    """Upload a PDF document for AI-powered explanation."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 10MB.")

    try:
        from .engine.document_ai import extract_text_from_pdf, explain_document

        file_bytes = await file.read()
        extracted_text = extract_text_from_pdf(file_bytes)

        if not extracted_text or len(extracted_text) < 10:
            return {
                "filename": file.filename,
                "status": "error",
                "message": "Could not extract text from this PDF. It may be scanned/image-based.",
                "extracted_text": "",
                "analysis": None,
            }

        api_key = os.environ.get("GROQ_API_KEY")
        analysis = explain_document(extracted_text, language=language, groq_api_key=api_key)

        return {
            "filename": file.filename,
            "status": "success",
            "pages_text_length": len(extracted_text),
            "extracted_text": extracted_text[:2000],
            "analysis": analysis,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document processing error: {str(e)}")


# =============================================================================
# TEXT-TO-SPEECH (Google Translate TTS)
# =============================================================================

import io
from fastapi.responses import StreamingResponse

class TTSRequest(BaseModel):
    text: str = Field(..., max_length=1000, description="Text to speak")
    language: str = Field("en", description="Language code (en, hi, kn, te, ta, bn, mr, gu, ml, pa)")

# Map our language codes to gTTS language codes
GTTS_LANG_MAP = {
    "en": "en", "hi": "hi", "kn": "kn", "te": "te", "ta": "ta",
    "bn": "bn", "mr": "mr", "gu": "gu", "ml": "ml", "pa": "pa",
}

@app.post("/api/tts", tags=["TTS"])
def text_to_speech(req: TTSRequest):
    """Convert text to speech using Google Translate TTS. Supports all 10 Indian languages."""
    try:
        from gtts import gTTS

        lang = GTTS_LANG_MAP.get(req.language, "en")
        cleaned = req.text[:1000].strip()
        if not cleaned:
            raise HTTPException(status_code=400, detail="Text cannot be empty.")

        tts = gTTS(text=cleaned, lang=lang)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=tts.mp3"},
        )
    except ImportError:
        raise HTTPException(status_code=500, detail="gTTS is not installed. Run: pip install gTTS")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")


# =============================================================================
# JOBS
# =============================================================================

import json as _json
from pathlib import Path as _Path

def _load_jobs():
    jobs_path = _Path(__file__).parent / "data" / "jobs.json"
    if not jobs_path.exists():
        return []
    with open(jobs_path, encoding="utf-8") as f:
        data = _json.load(f)
    return data.get("jobs", [])

@app.get("/api/jobs", tags=["Jobs"])
def get_jobs(
    city: Optional[str] = None,
    category: Optional[str] = None,
    keyword: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
):
    """Browse scraped job listings. Filter by city, category, or keyword."""
    jobs = _load_jobs()

    if city:
        jobs = [j for j in jobs if city.lower() in j.get("city", "").lower()
                or city.lower() in j.get("location", "").lower()]
    if category:
        jobs = [j for j in jobs if category.lower() in j.get("category", "").lower()]
    if keyword:
        jobs = [j for j in jobs if keyword.lower() in j.get("title", "").lower()]

    total = len(jobs)
    page = jobs[offset : offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "jobs": page,
    }

@app.get("/api/jobs/categories", tags=["Jobs"])
def get_job_categories():
    """Get all available job categories."""
    jobs = _load_jobs()
    from collections import Counter
    cats = Counter(j.get("category", "General") for j in jobs)
    return [{"category": k, "count": v} for k, v in cats.most_common()]

@app.get("/api/jobs/cities", tags=["Jobs"])
def get_job_cities():
    """Get all available cities."""
    jobs = _load_jobs()
    from collections import Counter
    cities = Counter(j.get("city", "") for j in jobs if j.get("city"))
    return [{"city": k, "count": v} for k, v in cities.most_common()]
