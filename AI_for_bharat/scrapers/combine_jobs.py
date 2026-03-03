"""
Combine scraped jobs from OLX and JustDial into a single jobs.json.

Usage:
    python scrapers/combine_jobs.py

Output:
    bharat_access_hub/data/jobs.json
"""

import json
import os

SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

OLX_FILE = os.path.join(SCRIPT_DIR, "olx_jobs.json")
JD_FILE = os.path.join(SCRIPT_DIR, "justdial_jobs.json")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "bharat_access_hub", "data", "jobs.json")


def main():
    all_jobs = []

    if os.path.exists(OLX_FILE):
        with open(OLX_FILE, "r", encoding="utf-8") as f:
            olx = json.load(f)
            all_jobs.extend(olx)
            print(f"[Combine] Loaded {len(olx)} OLX jobs")
    else:
        print(f"[Combine] OLX file not found: {OLX_FILE}")

    if os.path.exists(JD_FILE):
        with open(JD_FILE, "r", encoding="utf-8") as f:
            jd = json.load(f)
            all_jobs.extend(jd)
            print(f"[Combine] Loaded {len(jd)} JustDial jobs")
    else:
        print(f"[Combine] JustDial file not found: {JD_FILE}")

    # Sort by posted date (newest first)
    all_jobs.sort(key=lambda j: j.get("posted_date", ""), reverse=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_jobs, f, indent=2, ensure_ascii=False)

    print(f"\n[Combine] Total: {len(all_jobs)} jobs → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
