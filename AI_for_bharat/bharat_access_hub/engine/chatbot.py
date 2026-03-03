"""
RAG Chatbot Engine — LangChain + Groq (Llama 3.1).

Orchestrates:
  1. Retrieve relevant scheme docs from FAISS vector store
  2. Inject user profile + eligibility scores as context
  3. Generate personalized response via Groq LLM
"""

import os
from typing import Optional, List, Dict, Any

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from .vector_store import search_schemes, build_vector_store
from .eligibility import score_all_schemes, SchemeMatch
from ..models.profile import UserProfile
from ..languages import get_language_instruction, get_language_name


# ─── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the AI assistant for Bharat Access Hub, India's largest government scheme discovery platform with 100+ central and state schemes.

Your role: Help Indian citizens understand, discover, and apply for government welfare schemes they qualify for.

BEHAVIOR RULES:
1. ALWAYS respond in the language specified in the LANGUAGE INSTRUCTION below. This overrides everything else. Do NOT respond in Hindi unless the language instruction says Hindi.
2. Be warm, empathetic, and use simple language — many users may have limited education.
3. ALWAYS cite specific scheme names (in bold) when recommending.
4. Provide actionable steps: what documents they need, where to apply, and deadlines.
5. If unsure, say so — never make up scheme details.
6. Use the user's profile (age, income, state, occupation) to personalize recommendations.
7. When listing schemes, use numbered lists with benefit amounts.
8. For each scheme mention: name, benefit amount, key eligibility, how to apply.
9. Keep answers complete but scannable (use bullet points and bold text).
10. If the user asks about something NOT related to government schemes, politely redirect.

You have access to:
- The user's profile (demographics, location, income)
- Eligibility scores for 100+ schemes
- A knowledge base of scheme details

Use ALL available context to give the most relevant, personalized advice."""


RAG_PROMPT_TEMPLATE = """You are answering a question from an Indian citizen about government schemes.

━━━ USER PROFILE ━━━
{user_profile}

━━━ TOP ELIGIBLE SCHEMES (pre-computed scores) ━━━
{eligibility_summary}

━━━ RELEVANT SCHEME DETAILS (from knowledge base) ━━━
{retrieved_context}

━━━ CONVERSATION HISTORY ━━━
{chat_history}

━━━ USER QUESTION ━━━
{question}

━━━ MANDATORY LANGUAGE ━━━
{language_instruction}

INSTRUCTIONS:
- Answer the question using the scheme information provided above.
- Prioritize schemes the user is actually eligible for (from the scores).
- Include specific details: benefit amounts, required documents, application links.
- Be concise but thorough. Use bullet points and bold scheme names.
- If the question is about a specific scheme, provide detailed info about that scheme.
- If no schemes match the query, suggest the closest alternatives.
- CRITICAL: Your ENTIRE response MUST be in {language_name}. Do NOT use Hindi or any other language unless {language_name} IS Hindi.

Your response (in {language_name} ONLY):"""


# ─── Chatbot class ────────────────────────────────────────────────────────────

class ChatBot:
    """
    RAG-powered chatbot using LangChain + Groq.

    Usage:
        bot = ChatBot(groq_api_key="gsk_...")
        bot.set_profile(profile)
        response = bot.chat("Tell me about farming schemes")
    """

    def __init__(self, groq_api_key: Optional[str] = None, language: str = "en"):
        api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "Groq API key required. Set GROQ_API_KEY env var or pass groq_api_key parameter. "
                "Get a free key at https://console.groq.com/keys"
            )

        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=api_key,
            temperature=0.5,
            max_tokens=2048,
        )

        self.language = language
        self._build_prompt()

        self.profile: Optional[UserProfile] = None
        self.eligibility_results: List[SchemeMatch] = []
        self.chat_history: List[Dict[str, str]] = []

        # Pre-build vector store
        build_vector_store()

    def _build_prompt(self):
        """Build prompt template with language instruction."""
        lang_instruction = get_language_instruction(self.language)
        lang_name = get_language_name(self.language)
        system = (
            SYSTEM_PROMPT
            + f"\n\n⚠️ MANDATORY LANGUAGE INSTRUCTION: {lang_instruction} "
            + f"You MUST respond ONLY in {lang_name}. "
            + f"Even if the user writes in English or another language, your response must be in {lang_name}."
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", RAG_PROMPT_TEMPLATE),
        ])

    def set_language(self, language: str):
        """Change the response language."""
        self.language = language
        self._build_prompt()

    def set_profile(self, profile: UserProfile):
        """Set or update the user profile. Recomputes eligibility scores."""
        self.profile = profile
        self.eligibility_results = score_all_schemes(profile, top_n=10)

    def _format_profile(self) -> str:
        if not self.profile:
            return "No profile set yet."
        p = self.profile
        return (
            f"Name: {p.name}, Age: {p.age}, Gender: {p.gender}\n"
            f"State: {p.state}, Area: {p.area_type}\n"
            f"Category: {p.category}, Education: {p.education_level}\n"
            f"Employment: {p.employment_status}\n"
            f"Annual Income: Rs.{p.annual_income:,}\n"
            f"Family Size: {p.family_size}\n"
            f"Owns Land: {p.owns_land}, BPL: {p.bpl_card}"
        )

    def _format_eligibility(self) -> str:
        if not self.eligibility_results:
            return "No eligibility scores computed yet."
        lines = []
        for m in self.eligibility_results[:10]:
            lines.append(
                f"- **{m.scheme_name}**: {m.eligibility_score}% eligible "
                f"(Benefit: Rs.{m.benefit_amount:,}) — {m.reason}"
            )
        return "\n".join(lines)

    def _format_history(self) -> str:
        if not self.chat_history:
            return "No previous conversation."
        lines = []
        for turn in self.chat_history[-5:]:  # Last 5 turns
            lines.append(f"User: {turn['user']}")
            lines.append(f"Assistant: {turn['assistant']}")
        return "\n".join(lines)

    def chat(self, question: str) -> str:
        """
        Send a message and get a RAG-powered response.

        Args:
            question: The user's question (any language).

        Returns:
            The assistant's response string.
        """
        # 1. Retrieve relevant scheme docs
        retrieved_docs = search_schemes(question, k=8)
        retrieved_context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

        # 2. Build the prompt
        chain = self.prompt | self.llm

        # 3. Generate response
        lang_instruction = get_language_instruction(self.language)
        lang_name = get_language_name(self.language)
        response = chain.invoke({
            "user_profile": self._format_profile(),
            "eligibility_summary": self._format_eligibility(),
            "retrieved_context": retrieved_context,
            "chat_history": self._format_history(),
            "question": question,
            "language_instruction": lang_instruction,
            "language_name": lang_name,
        })

        answer = response.content

        # 4. Store in history
        self.chat_history.append({
            "user": question,
            "assistant": answer,
        })

        return answer

    def get_chat_history(self) -> List[Dict[str, str]]:
        """Return the conversation history."""
        return self.chat_history

    def clear_history(self):
        """Clear conversation history."""
        self.chat_history = []
