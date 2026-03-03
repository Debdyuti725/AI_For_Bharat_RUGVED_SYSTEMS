"""
Bharat Access Hub — Complete Interactive Experience.

Run with:
    python run.py

Flow:
    1. Sign Up or Sign In
    2. Tier 1 Questionnaire (10 questions)
    3. See your matched schemes
    4. Optionally unlock Tier 2 (10 more questions per category)
    5. Chat with AI assistant about your schemes
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Auto-load .env file (so user doesn't need to set env vars manually)
def load_dotenv():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())

load_dotenv()

from bharat_access_hub.auth import signup, login, get_current_user
from bharat_access_hub.database import init_db, save_profile, get_profile, save_chat_message, get_chat_history
from bharat_access_hub.questionnaire import (
    build_profile, extend_profile,
    get_onboarding_questions, get_category_questions,
    suggest_tier2_categories,
)
from bharat_access_hub.engine.eligibility import score_all_schemes
from bharat_access_hub.models.profile import UserProfile


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def divider(title=""):
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
    else:
        print(f"{'-' * 60}")


def ask_question(q, index, total):
    """Ask a single question and return the validated answer."""
    print(f"\n  Question {index}/{total}")
    print(f"  {q.text}")
    if q.text_hi:
        print(f"  ({q.text_hi})")
    if q.help_text:
        print(f"  Hint: {q.help_text}")

    if q.qtype == "choice" and q.options:
        print()
        for i, opt in enumerate(q.options, 1):
            label_hi = opt.get("label_hi", "")
            hi_part = f" ({label_hi})" if label_hi else ""
            try:
                print(f"    {i}. {opt['label']}{hi_part}")
            except UnicodeEncodeError:
                print(f"    {i}. {opt['label']}")
        while True:
            try:
                choice = input(f"\n  Enter number (1-{len(q.options)}): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(q.options):
                    return q.options[idx]["value"]
                print("  Invalid choice, try again.")
            except (ValueError, KeyboardInterrupt):
                print("  Please enter a valid number.")

    elif q.qtype == "number":
        vmin = q.validation.get("min", 0) if q.validation else 0
        vmax = q.validation.get("max", 99999999) if q.validation else 99999999
        while True:
            try:
                val = input(f"\n  Enter a number ({vmin}-{vmax}): ").strip()
                num = int(val)
                if vmin <= num <= vmax:
                    return num
                print(f"  Must be between {vmin} and {vmax}.")
            except (ValueError, KeyboardInterrupt):
                print("  Please enter a valid number.")

    elif q.qtype == "bool":
        while True:
            val = input(f"\n  Yes or No? (y/n): ").strip().lower()
            if val in ("y", "yes", "1", "true"):
                return True
            if val in ("n", "no", "0", "false"):
                return False
            print("  Please enter y or n.")

    else:
        while True:
            val = input(f"\n  Your answer: ").strip()
            if val:
                return val
            if not q.required:
                return ""
            print("  This field is required.")


def show_results(profile):
    """Show eligibility results."""
    divider("YOUR SCHEME MATCHES")
    results = score_all_schemes(profile, min_score=0.0)

    for i, m in enumerate(results, 1):
        score = m.eligibility_score
        bar_filled = int(score / 10)
        bar = "#" * bar_filled + "." * (10 - bar_filled)

        if score >= 75:
            tag = "** HIGH MATCH **"
        elif score >= 50:
            tag = "Partial Match"
        else:
            tag = "Low Match"

        print(f"\n  {i:2}. [{score:5.1f}%] [{bar}]  {m.scheme_name}")
        print(f"      Benefit: Rs.{m.benefit_amount:,} | Category: {m.category.title()} | {tag}")
        print(f"      {m.reason}")

    high = sum(1 for r in results if r.eligibility_score >= 75)
    mid = sum(1 for r in results if 50 <= r.eligibility_score < 75)
    print(f"\n  Total: {len(results)} schemes | High matches (>75%): {high} | Partial (50-75%): {mid}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FLOW
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    init_db()
    clear()

    print()
    print("  ######################################################")
    print("  #                                                    #")
    print("  #          BHARAT ACCESS HUB                         #")
    print("  #          AI-Powered Government Scheme Discovery    #")
    print("  #                                                    #")
    print("  ######################################################")

    # ─── STEP 1: AUTH ──────────────────────────────────────────
    divider("STEP 1: Sign In / Sign Up")
    print("\n  1. Sign In (existing account)")
    print("  2. Sign Up (new account)")

    while True:
        choice = input("\n  Choose (1 or 2): ").strip()
        if choice in ("1", "2"):
            break
        print("  Please enter 1 or 2.")

    user_id = None
    user_name = ""

    if choice == "2":
        # SIGN UP
        print("\n  --- Create New Account ---")
        name = input("  Your full name: ").strip()
        email = input("  Email: ").strip()
        while True:
            password = input("  Password (min 6 chars): ").strip()
            if len(password) >= 6:
                break
            print("  Password too short!")

        try:
            result = signup(email, password, name)
            user_id = result["user_id"]
            user_name = name
            print(f"\n  Account created! Welcome, {name}!")
        except ValueError as e:
            print(f"\n  Error: {e}")
            print("  Trying to log in instead...")
            try:
                result = login(email, password)
                user_id = result["user_id"]
                user_name = result["name"]
                print(f"  Logged in as {user_name}!")
            except ValueError as e2:
                print(f"  Login failed: {e2}")
                return
    else:
        # SIGN IN
        print("\n  --- Sign In ---")
        email = input("  Email: ").strip()
        password = input("  Password: ").strip()

        try:
            result = login(email, password)
            user_id = result["user_id"]
            user_name = result["name"]
            print(f"\n  Welcome back, {user_name}!")
        except ValueError as e:
            print(f"\n  Login failed: {e}")
            return

    # ─── Check existing profile ────────────────────────────────
    existing_profile = get_profile(user_id)
    profile = None

    if existing_profile and existing_profile["tier1_complete"]:
        profile = UserProfile.from_dict(existing_profile["profile_json"])
        print(f"\n  Found your existing profile!")
        print(f"  {profile}")

        redo = input("\n  Use existing profile? (y/n): ").strip().lower()
        if redo in ("n", "no"):
            profile = None

    # ─── STEP 2: TIER 1 QUESTIONNAIRE ─────────────────────────
    if profile is None:
        divider("STEP 2: Tell Us About Yourself (10 Questions)")
        print("  These questions help us find the best government schemes for you.\n")

        questions = get_onboarding_questions()
        answers = {}

        for i, q in enumerate(questions, 1):
            answer = ask_question(q, i, len(questions))
            answers[q.key] = answer
            print(f"  >> Saved: {q.key} = {answer}")

        profile = build_profile(answers)
        print(f"\n  Profile built! Completion: {profile.tier1_completion_pct}%")

        # Save to DB
        save_profile(
            user_id=user_id,
            profile_dict=profile.to_dict(),
            tier1_complete=True,
            tier2_categories=[],
        )
        print("  Profile saved to database!")

    # ─── STEP 3: SHOW RESULTS ─────────────────────────────────
    show_results(profile)

    # ─── STEP 4: TIER 2 (OPTIONAL) ────────────────────────────
    while True:
        divider("STEP 3: Unlock More Targeted Questions (Optional)")
        suggestions = suggest_tier2_categories(profile)
        already_done = profile.completed_tier2_categories
        available = [s for s in suggestions if s["category"] not in already_done]

        if not available:
            print("\n  You've completed all categories!")
            break

        for i, s in enumerate(available, 1):
            print(f"    {i}. {s['category'].title()} - {s['reason']}")
        print(f"    0. Skip - go to AI Chat")

        try:
            choice = input(f"\n  Pick a category (0-{len(available)}): ").strip()
            idx = int(choice) - 1
            if idx == -1:
                break
            if 0 <= idx < len(available):
                cat = available[idx]["category"]
                cat_questions = get_category_questions(cat)
                cat_answers = {}

                print(f"\n  --- {cat.upper()} Questions ---")
                for j, q in enumerate(cat_questions, 1):
                    answer = ask_question(q, j, len(cat_questions))
                    cat_answers[q.key] = answer
                    print(f"  >> Saved: {q.key} = {answer}")

                profile = extend_profile(profile, cat, cat_answers)

                # Save updated profile
                save_profile(
                    user_id=user_id,
                    profile_dict=profile.to_dict(),
                    tier1_complete=True,
                    tier2_categories=profile.completed_tier2_categories,
                )
                show_results(profile)
            else:
                print("  Invalid choice.")
        except (ValueError, KeyboardInterrupt):
            break

    # ─── STEP 5: AI CHAT ──────────────────────────────────────
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        divider("AI CHAT (Skipped)")
        print("\n  GROQ_API_KEY not set. To enable AI chat, run:")
        print('  set GROQ_API_KEY=gsk_your_key_here')
        print("  Then run this program again.\n")
    else:
        divider("STEP 4: Chat with AI Assistant")
        print("  Ask me anything about government schemes!")
        print("  I know your profile and eligibility scores.")
        print("  Type 'quit' or 'exit' to end.\n")

        try:
            from bharat_access_hub.engine.chatbot import ChatBot
            bot = ChatBot(groq_api_key=api_key)
            bot.set_profile(profile)

            while True:
                try:
                    question = input("  You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break

                if not question:
                    continue
                if question.lower() in ("quit", "exit", "bye", "q"):
                    break

                print("  Thinking...")
                try:
                    answer = bot.chat(question)
                    save_chat_message(user_id, "user", question)
                    save_chat_message(user_id, "assistant", answer)
                    print(f"\n  AI: {answer}\n")
                except Exception as e:
                    print(f"  Error: {e}\n")

        except Exception as e:
            print(f"  Could not start chatbot: {e}")

    # ─── DONE ─────────────────────────────────────────────────
    divider("Thank you for using Bharat Access Hub!")
    print(f"  Your profile and chat history are saved, {user_name}.")
    print(f"  Run this program again to continue where you left off.\n")


if __name__ == "__main__":
    main()
