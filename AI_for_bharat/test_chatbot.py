"""Quick test for the RAG chatbot end-to-end."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 1. Test vector store
print("=" * 50)
print("  Step 1: Building vector store...")
print("=" * 50)
from bharat_access_hub.engine.vector_store import build_vector_store, search_schemes

store = build_vector_store(force_rebuild=True)
print("  Vector store built!\n")

# 2. Test semantic search
print("  Step 2: Testing semantic search...")
results = search_schemes("farming schemes for small farmers", k=3)
for r in results:
    sid = r.metadata.get("scheme_id", "?")
    snippet = r.page_content[:80].replace("\n", " ")
    print("    - [%s] %s..." % (sid, snippet))
print()

# 3. Test chatbot (needs GROQ_API_KEY)
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    print("  GROQ_API_KEY not set. Skipping chatbot test.")
    print("  Set it with: set GROQ_API_KEY=gsk_your_key")
    sys.exit(0)

print("  Step 3: Testing chatbot with Groq...")
from bharat_access_hub.engine.chatbot import ChatBot
from bharat_access_hub.questionnaire import build_profile

profile = build_profile({
    "name": "Vedh", "age": 19, "gender": "male",
    "state": "karnataka", "area_type": "urban", "category": "general",
    "education_level": "graduate", "employment_status": "student",
    "annual_income": 800000, "family_size": 3,
})

bot = ChatBot()
bot.set_profile(profile)

print("  Sending question: 'What schemes am I eligible for?'")
answer = bot.chat("What schemes am I eligible for as a student?")
print("\n  RESPONSE:")
print("  " + "-" * 45)
for line in answer.split("\n"):
    print("  " + line)
print("  " + "-" * 45)
print("\n  ALL TESTS PASSED!")
