"""
Interactive Questionnaire — Answer the 10 Tier 1 questions yourself,
see your matched schemes, then optionally unlock Tier 2 for a category.

Run with:
    python -m bharat_access_hub.interactive
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bharat_access_hub.questionnaire import (
    build_profile, extend_profile,
    get_onboarding_questions, get_category_questions,
    suggest_tier2_categories, get_available_categories,
)
from bharat_access_hub.engine.eligibility import score_all_schemes


def clear():
    os.system("cls" if os.name == "nt" else "clear")


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
            print(f"    {i}. {opt['label']}{hi_part}")
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

    else:  # text
        while True:
            val = input(f"\n  Your answer: ").strip()
            if val:
                return val
            if not q.required:
                return ""
            print("  This field is required.")


def show_results(profile, label=""):
    """Show eligibility results for the current profile."""
    print(f"\n  {'='*55}")
    print(f"  YOUR SCHEME MATCHES{' - ' + label if label else ''}")
    print(f"  {'='*55}")

    results = score_all_schemes(profile, min_score=0.0)

    if not results:
        print("\n  No schemes found. Try completing more profile sections.")
        return

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
        if m.missing_criteria:
            missing_short = [x.split(" — ")[0].strip() for x in m.missing_criteria[:3]]
            print(f"      Missing: {', '.join(missing_short)}")

    print(f"\n  Total schemes checked: {len(results)}")
    high = sum(1 for r in results if r.eligibility_score >= 75)
    mid = sum(1 for r in results if 50 <= r.eligibility_score < 75)
    print(f"  High matches (>75%): {high}  |  Partial (50-75%): {mid}")


def run_interactive():
    clear()
    print("\n" + "="*55)
    print("   BHARAT ACCESS HUB - Smart Profile Builder")
    print("   Answer 10 questions to find schemes you qualify for")
    print("="*55)

    # ─── TIER 1 ───────────────────────────────────────────────
    questions = get_onboarding_questions()
    answers = {}

    for i, q in enumerate(questions, 1):
        answer = ask_question(q, i, len(questions))
        answers[q.key] = answer
        print(f"  -> Saved: {q.key} = {answer}")

    # Build profile
    profile = build_profile(answers)

    print(f"\n  Profile built successfully!")
    print(f"  Completion: {profile.tier1_completion_pct}%")
    print(f"  {profile}")

    # Show Tier 1 results
    show_results(profile, "After Tier 1 (10 questions)")

    # ─── TIER 2 (optional) ────────────────────────────────────
    while True:
        print(f"\n  {'-'*55}")
        print("  UNLOCK TIER 2 (10 more targeted questions)")
        print(f"  {'-'*55}")

        suggestions = suggest_tier2_categories(profile)
        already_done = profile.completed_tier2_categories

        available = [s for s in suggestions if s["category"] not in already_done]
        if not available:
            print("\n  You've completed all Tier 2 categories!")
            break

        for i, s in enumerate(available, 1):
            display = s["display_name"]
            # Strip emoji for Windows compatibility
            try:
                display.encode("ascii")
            except UnicodeEncodeError:
                display = s["category"].title()
            print(f"    {i}. {display} - {s['reason']}")
        print(f"    0. Skip / I'm done")

        try:
            choice = input(f"\n  Pick a category (0-{len(available)}): ").strip()
            idx = int(choice) - 1
            if idx == -1:
                break
            if 0 <= idx < len(available):
                cat = available[idx]["category"]
                cat_questions = get_category_questions(cat)
                cat_answers = {}

                print(f"\n  --- {cat.upper()} Tier 2 Questions ---")
                for j, q in enumerate(cat_questions, 1):
                    answer = ask_question(q, j, len(cat_questions))
                    cat_answers[q.key] = answer
                    print(f"  -> Saved: {q.key} = {answer}")

                profile = extend_profile(profile, cat, cat_answers)
                show_results(profile, f"After Tier 2 ({cat.title()})")
            else:
                print("  Invalid choice.")
        except (ValueError, KeyboardInterrupt):
            break

    # ─── FINAL SUMMARY ────────────────────────────────────────
    print(f"\n  {'='*55}")
    print("  FINAL PROFILE SUMMARY")
    print(f"  {'='*55}")
    print(f"  Name: {profile.name}")
    print(f"  Age: {profile.age} | Gender: {profile.gender}")
    print(f"  State: {profile.state} | Area: {profile.area_type}")
    print(f"  Category: {profile.category} | Education: {profile.education_level}")
    print(f"  Employment: {profile.employment_status}")
    print(f"  Annual Income: Rs.{profile.annual_income:,}")
    print(f"  Family Size: {profile.family_size}")
    print(f"  Tier 2 completed: {profile.completed_tier2_categories}")
    print()

    show_results(profile, "FINAL")
    print("\n  Thank you for using Bharat Access Hub!\n")


if __name__ == "__main__":
    run_interactive()
