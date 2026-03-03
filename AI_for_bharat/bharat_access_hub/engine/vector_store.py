"""
Scheme Document Loader + FAISS Vector Store Builder.

Loads the 15 hardcoded schemes as LangChain Documents,
generates embeddings using a free local model (HuggingFace),
and stores them in a FAISS vector store for semantic search.

No AWS required — runs entirely locally.
"""

import os
import json
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from ..data.schemes import get_all_schemes


# ─── Build LangChain Documents from scheme data ──────────────────────────────

def _scheme_to_documents(scheme: dict) -> List[Document]:
    """
    Convert a single scheme dict into multiple Documents (chunks).
    Each scheme produces 3-4 documents for better retrieval granularity.
    """
    docs = []
    sid = scheme["scheme_id"]
    name = scheme["name"]
    cat = scheme["category"]

    # Doc 1: Overview
    overview = (
        f"Scheme: {name}\n"
        f"Category: {cat}\n"
        f"Description: {scheme['description']}\n"
        f"Benefit: Rs.{scheme['benefit_amount']:,} ({scheme['benefit_type']})\n"
    )
    if scheme.get("deadline"):
        overview += f"Application Deadline: {scheme['deadline']}\n"
    if scheme.get("application_url"):
        overview += f"Apply at: {scheme['application_url']}\n"
    docs.append(Document(
        page_content=overview,
        metadata={"scheme_id": sid, "section": "overview", "category": cat}
    ))

    # Doc 2: Eligibility criteria (human-readable)
    criteria = scheme.get("eligibility_criteria", {})
    elig_text = f"Eligibility criteria for {name}:\n"
    if criteria.get("age_min") or criteria.get("age_max"):
        elig_text += f"- Age: {criteria.get('age_min', 'any')} to {criteria.get('age_max', 'any')} years\n"
    if criteria.get("income_max"):
        elig_text += f"- Maximum annual income: Rs.{criteria['income_max']:,}\n"
    elif criteria.get("income_max") is None:
        elig_text += "- No income ceiling\n"
    if criteria.get("employment_status"):
        elig_text += f"- Employment: {', '.join(criteria['employment_status'])}\n"
    if criteria.get("category"):
        elig_text += f"- Social category: {', '.join(criteria['category'])}\n"
    if criteria.get("states"):
        states = criteria["states"]
        elig_text += f"- States: {'All India' if 'all' in states else ', '.join(states)}\n"
    if criteria.get("area_type"):
        elig_text += f"- Area: {', '.join(criteria['area_type'])}\n"
    if criteria.get("owns_land") is True:
        elig_text += "- Must own agricultural land\n"
    if criteria.get("bpl_card") is True:
        elig_text += "- Must have BPL card or below poverty line income\n"
    if criteria.get("currently_enrolled") is True:
        elig_text += "- Must be currently enrolled in an educational institution\n"
    if criteria.get("education_level"):
        elig_text += f"- Minimum education: {', '.join(criteria['education_level'])}\n"
    docs.append(Document(
        page_content=elig_text,
        metadata={"scheme_id": sid, "section": "eligibility", "category": cat}
    ))

    # Doc 3: Required documents
    req_docs = scheme.get("required_documents", [])
    if req_docs:
        docs_text = f"Documents required for {name}:\n"
        for d in req_docs:
            docs_text += f"- {d}\n"
        docs.append(Document(
            page_content=docs_text,
            metadata={"scheme_id": sid, "section": "documents", "category": cat}
        ))

    # Doc 4: How to apply
    apply_text = (
        f"How to apply for {name}:\n"
        f"1. Visit the official portal: {scheme.get('application_url', 'N/A')}\n"
        f"2. Gather required documents: {', '.join(req_docs[:3])}\n"
        f"3. Fill the application form online\n"
        f"4. Submit and note the application reference number\n"
    )
    if scheme.get("deadline"):
        apply_text += f"Note: Application deadline is {scheme['deadline']}. Apply before this date.\n"
    docs.append(Document(
        page_content=apply_text,
        metadata={"scheme_id": sid, "section": "how_to_apply", "category": cat}
    ))

    return docs


def load_all_scheme_documents() -> List[Document]:
    """Load all 15 schemes as LangChain Documents."""
    all_docs = []
    for scheme in get_all_schemes():
        all_docs.extend(_scheme_to_documents(scheme))
    return all_docs


# ─── Vector Store ─────────────────────────────────────────────────────────────

# Use HuggingFace embeddings (free, local, no API key needed)
def _get_embeddings():
    """Get embedding model — tries HuggingFace first, falls back to simple."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Cache the vector store in memory
_vector_store = None
_STORE_PATH = os.path.join(os.path.dirname(__file__), "..", ".vector_store")


def build_vector_store(force_rebuild: bool = False) -> FAISS:
    """
    Build or load the FAISS vector store.

    First run: embeds all scheme documents and saves to disk.
    Subsequent runs: loads from disk (fast).
    """
    global _vector_store

    if _vector_store is not None and not force_rebuild:
        return _vector_store

    embeddings = _get_embeddings()
    abs_path = os.path.abspath(_STORE_PATH)

    if os.path.exists(abs_path) and not force_rebuild:
        print("[VectorStore] Loading from disk...")
        _vector_store = FAISS.load_local(abs_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print("[VectorStore] Building from scheme data...")
        docs = load_all_scheme_documents()
        print(f"[VectorStore] Embedding {len(docs)} documents...")
        _vector_store = FAISS.from_documents(docs, embeddings)
        os.makedirs(abs_path, exist_ok=True)
        _vector_store.save_local(abs_path)
        print(f"[VectorStore] Saved to {abs_path}")

    return _vector_store


def search_schemes(query: str, k: int = 5) -> List[Document]:
    """Search the vector store for relevant scheme chunks."""
    store = build_vector_store()
    return store.similarity_search(query, k=k)
