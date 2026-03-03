"""
Document Intelligence Module — PDF upload, text extraction, AI explanation.

Features:
  - Extract text from uploaded PDFs
  - Use LLM to explain document in simple language
  - Identify key fields and their meanings
"""

import os
import io
from typing import Optional


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes using pdfplumber or fallback."""
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            return text.strip()
    except ImportError:
        # Fallback: try PyPDF2
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            return text.strip()
        except ImportError:
            return "[Error] No PDF library installed. Install pdfplumber: pip install pdfplumber"


def explain_document(text: str, language: str = "en", groq_api_key: str = None) -> dict:
    """Use LLM to explain a document in simple language."""
    if not groq_api_key:
        groq_api_key = os.environ.get("GROQ_API_KEY")
    
    if not groq_api_key:
        return {
            "summary": "GROQ_API_KEY not set. Cannot analyze document.",
            "key_fields": [],
            "action_required": [],
        }
    
    from langchain_groq import ChatGroq
    
    lang_map = {
        "en": "English", "hi": "Hindi", "kn": "Kannada", "te": "Telugu",
        "ta": "Tamil", "bn": "Bengali", "mr": "Marathi", "gu": "Gujarati",
        "ml": "Malayalam", "pa": "Punjabi",
    }
    lang_name = lang_map.get(language, "English")
    
    prompt = f"""You are a helpful government document assistant. Analyze this document and explain it in simple {lang_name}.

DOCUMENT TEXT:
{text[:4000]}

Provide your response in this exact format:

SUMMARY: [2-3 sentence summary of what this document is about, in {lang_name}]

KEY FIELDS:
- [Field name]: [What it means and what to fill, in {lang_name}]
- [Field name]: [What it means and what to fill, in {lang_name}]

ACTIONS REQUIRED:
- [Step 1: What the person needs to do, in {lang_name}]
- [Step 2: Next step, in {lang_name}]

DOCUMENTS NEEDED:
- [Document 1]
- [Document 2]

Important: Use simple, everyday language that someone with basic education can understand. Respond in {lang_name}."""

    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=groq_api_key,
            temperature=0.3,
            max_tokens=2000,
        )
        response = llm.invoke(prompt)
        content = response.content
        
        # Parse the response
        result = {
            "summary": "",
            "key_fields": [],
            "actions_required": [],
            "documents_needed": [],
            "raw_explanation": content,
        }
        
        # Simple parsing
        sections = content.split("\n\n")
        for section in sections:
            if section.upper().startswith("SUMMARY:"):
                result["summary"] = section.split(":", 1)[1].strip()
            elif "KEY FIELDS" in section.upper():
                lines = section.split("\n")[1:]
                result["key_fields"] = [l.strip("- ").strip() for l in lines if l.strip().startswith("-")]
            elif "ACTIONS REQUIRED" in section.upper():
                lines = section.split("\n")[1:]
                result["actions_required"] = [l.strip("- ").strip() for l in lines if l.strip().startswith("-")]
            elif "DOCUMENTS NEEDED" in section.upper():
                lines = section.split("\n")[1:]
                result["documents_needed"] = [l.strip("- ").strip() for l in lines if l.strip().startswith("-")]
        
        if not result["summary"]:
            result["summary"] = content[:300]
        
        return result
        
    except Exception as e:
        return {
            "summary": f"Error analyzing document: {str(e)}",
            "key_fields": [],
            "actions_required": [],
            "documents_needed": [],
            "raw_explanation": "",
        }
