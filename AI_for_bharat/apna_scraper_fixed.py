"""
Apna.co Jobs Scraper
====================
Scrapes job listings from apna.co for 10th-pass and 12th-pass candidates.
Uses requests + BeautifulSoup (no Selenium needed — page is server-side rendered).

Output JSON format is identical to olx_jobs_scraper.py so both files can be
merged directly into bharat_access_hub/data/jobs.json.

Usage:
    python apna_scraper.py
    python apna_scraper.py --pages 5 --output bharat_access_hub/data/jobs.json
    python apna_scraper.py --merge  bharat_access_hub/data/jobs.json
"""

import json
import re
import time
import random
import argparse
import logging
from datetime import datetime
from pathlib import Path
from collections import Counter

import requests
from bs4 import BeautifulSoup

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_URL = "https://apna.co"

SCRAPE_URLS = [
    ("https://apna.co/jobs/jobs-in-mumbai/10th_pass-jobs",     "10th Pass"),
    ("https://apna.co/jobs/jobs-in-delhi/10th_pass-jobs",      "10th Pass"),
    ("https://apna.co/jobs/jobs-in-bangalore/10th_pass-jobs",  "10th Pass"),
    ("https://apna.co/jobs/jobs-in-hyderabad/10th_pass-jobs",  "10th Pass"),
    ("https://apna.co/jobs/jobs-in-chennai/10th_pass-jobs",    "10th Pass"),
    ("https://apna.co/jobs/jobs-in-pune/10th_pass-jobs",       "10th Pass"),
    ("https://apna.co/jobs/jobs-in-kolkata/10th_pass-jobs",    "10th Pass"),
    ("https://apna.co/jobs/jobs-in-ahmedabad/10th_pass-jobs",  "10th Pass"),
    ("https://apna.co/jobs/jobs-in-jaipur/10th_pass-jobs",     "10th Pass"),
    ("https://apna.co/jobs/jobs-in-lucknow/10th_pass-jobs",    "10th Pass"),
    ("https://apna.co/jobs/jobs-in-mumbai/12th_pass-jobs",     "12th Pass"),
    ("https://apna.co/jobs/jobs-in-delhi/12th_pass-jobs",      "12th Pass"),
    ("https://apna.co/jobs/jobs-in-bangalore/12th_pass-jobs",  "12th Pass"),
    ("https://apna.co/jobs/jobs-in-hyderabad/12th_pass-jobs",  "12th Pass"),
    ("https://apna.co/jobs/jobs-in-chennai/12th_pass-jobs",    "12th Pass"),
    ("https://apna.co/jobs/jobs-in-pune/12th_pass-jobs",       "12th Pass"),
    ("https://apna.co/jobs/jobs-in-kolkata/12th_pass-jobs",    "12th Pass"),
    ("https://apna.co/jobs/jobs-in-ahmedabad/12th_pass-jobs",  "12th Pass"),
    ("https://apna.co/jobs/jobs-in-jaipur/12th_pass-jobs",     "12th Pass"),
    ("https://apna.co/jobs/jobs-in-lucknow/12th_pass-jobs",    "12th Pass"),
]
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://apna.co/",
}

# ── Category inference (same logic as OLX scraper) ────────────────────────────

def infer_category(title: str, tags: list[str]) -> str:
    text = (title + " " + " ".join(tags)).lower()

    if any(x in text for x in ["driver", "cab", "chauffeur"]):
        return "Driver"
    if any(x in text for x in ["cook", "chef", "kitchen", "food"]):
        return "Cook / Chef"
    if any(x in text for x in ["delivery", "courier", "logistics", "dispatch"]):
        return "Delivery"
    if any(x in text for x in ["security", "guard", "watchman", "bouncer"]):
        return "Security"
    if any(x in text for x in ["sales", "marketing", "telecaller", "bde", "business development"]):
        return "Sales & Marketing"
    if any(x in text for x in ["software", "developer", "programmer", "it ", "tech"]):
        return "IT & Software"
    if any(x in text for x in ["accountant", "account", "finance", "tally", "gst", "bookkeeping"]):
        return "Accounting & Finance"
    if any(x in text for x in ["teach", "tutor", "faculty", "trainer", "instructor", "school"]):
        return "Teaching & Education"
    if any(x in text for x in ["factory", "manufactur", "production", "operator", "assembly"]):
        return "Factory & Manufacturing"
    if any(x in text for x in ["maid", "domestic", "housekeep", "caretaker", "nanny", "helper"]):
        return "Domestic Help"
    if any(x in text for x in ["construct", "labour", "mason", "plumber", "electrician", "carpenter"]):
        return "Construction & Labour"
    if any(x in text for x in ["nurse", "doctor", "hospital", "medical", "pharmacy", "health"]):
        return "Healthcare"
    if any(x in text for x in ["hotel", "waiter", "hospitality", "restaurant", "housekeeping"]):
        return "Hospitality"
    if any(x in text for x in ["manager", "supervisor", "executive", "officer", "admin", "back office"]):
        return "Office & Admin"
    if any(x in text for x in ["spa", "salon", "beauty", "massage", "therapist"]):
        return "Beauty & Wellness"

    return "General"


# ── City normalisation ─────────────────────────────────────────────────────────
# Apna uses full city names; normalise to lowercase slugs matching OLX format.

def normalise_city(raw: str) -> str:
    """Map Apna city strings to lowercase slugs used in jobs.json."""
    raw = raw.strip().lower()
    mapping = {
        "mumbai/bombay":          "mumbai",
        "bengaluru/bangalore":    "bangalore",
        "kolkata/calcutta":       "kolkata",
        "kolkata/calcutta region":"kolkata",
        "new delhi":              "delhi",
        "delhi":                  "delhi",
        "hyderabad":              "hyderabad",
        "chennai":                "chennai",
        "pune":                   "pune",
        "ahmedabad":              "ahmedabad",
        "jaipur":                 "jaipur",
        "lucknow":                "lucknow",
    }
    # Exact match first
    if raw in mapping:
        return mapping[raw]
    # Partial match
    for key, val in mapping.items():
        if key in raw or raw in key:
            return val
    # Strip region suffixes and return whatever we have
    raw = re.sub(r"/.*$", "", raw).strip()
    return raw


# ── Salary normalisation ───────────────────────────────────────────────────────

def normalise_salary(raw: str) -> str:
    """Convert Apna monthly salary ranges to a clean string."""
    if not raw:
        return "Not specified"
    # Apna shows "₹50,000 - ₹149,999 monthly" or "₹50,000 - ₹149,999 monthly*"
    cleaned = raw.replace("*", "").strip()
    # Convert to yearly if needed — keep as-is for now, just clean up
    return cleaned if cleaned else "Not specified"


# ── HTML parsing ───────────────────────────────────────────────────────────────

def parse_apna_page(html: str, qualification: str) -> list[dict]:
    """
    Parse one Apna jobs listing page and return a list of job dicts.

    Apna renders job cards as <a href="/job/<city>/<title-id>"> elements.
    Inside each card:
      - h2 or h3 → job title
      - img alt text or surrounding text → company name (best-effort)
      - icon + text pairs → location, salary, work mode, experience
    """
    soup = BeautifulSoup(html, "lxml")
    jobs = []

    # Each job card is an <a> tag whose href starts with /job/
    cards = soup.find_all("a", href=re.compile(r"^/job/"))

    for card in cards:
        href = card.get("href", "")
        url  = BASE_URL + href

        # ── Title ─────────────────────────────────────────────────────────────
        title_tag = card.find(["h2", "h3"])
        title = title_tag.get_text(strip=True) if title_tag else ""
        if not title:
            continue

        # ── Company ───────────────────────────────────────────────────────────
        # Company logo img has alt text set to the company name on most cards
        logo_img = card.find("img", class_=re.compile(r"company.logo", re.I))
        if not logo_img:
            # Fallback: second img in card (first is usually company logo)
            imgs = card.find_all("img")
            logo_img = imgs[1] if len(imgs) > 1 else None
        company = ""
        if logo_img:
            alt = logo_img.get("alt", "")
            if alt and "logo" not in alt.lower() and alt != "company-logo":
                company = alt.strip()

        # ── Structured fields via icon+text pattern ───────────────────────────
        # Apna uses <img src="...Location_icon..."> followed by a text node
        # We extract all img src values and the text that follows each one.
        fields = {}
        for img in card.find_all("img"):
            src = img.get("src", "")
            # Get the next sibling text
            nxt = img.find_next_sibling(string=True)
            if not nxt:
                # Sometimes wrapped in a span/div — get parent's text minus img
                parent = img.parent
                if parent:
                    nxt = parent.get_text(separator=" ", strip=True)
            if not nxt:
                continue
            nxt = nxt.strip()
            if not nxt:
                continue

            src_lower = src.lower()
            if "location" in src_lower or "Location_icon" in src:
                fields["location"] = nxt
            elif "salary" in src_lower or "Salary_icon" in src:
                fields["salary"] = nxt
            elif "work_from" in src_lower or "work%20from" in src_lower:
                fields["work_mode"] = nxt
            elif "full_time" in src_lower or "full%20time" in src_lower:
                fields["work_type"] = nxt
            elif "part_time" in src_lower or "part%20time" in src_lower:
                fields["work_type"] = fields.get("work_type", "") + " Part Time"
            elif "experience" in src_lower:
                fields["experience"] = nxt
            elif "field_job" in src_lower or "field%20job" in src_lower:
                fields["work_mode"] = "Field Job"
            elif "english" in src_lower or "advanced%20english" in src_lower:
                fields["english"] = nxt

        raw_location = fields.get("location", "")
        raw_salary   = fields.get("salary", "")
        work_type    = fields.get("work_type", "").strip()
        experience   = fields.get("experience", "")

        city = normalise_city(raw_location)

        # Extract region from raw_location (strip the city prefix)
        # e.g. "Andheri West, Mumbai/Bombay" → "Andheri West, Mumbai"
        location_clean = raw_location.replace("/Bombay", "").replace(
            "/Bangalore", "").replace("/Calcutta", "").replace(
            "/Calcutta Region", "").strip().rstrip(",").strip()

        # Build tags list for category inference
        tags = [work_type, experience, fields.get("work_mode", "")]

        jobs.append({
            "title":         title,
            "company":       company,
            "salary":        normalise_salary(raw_salary),
            "location":      location_clean or city,
            "city":          city,
            "date_posted":   "Recent",         # Apna doesn't show listing dates
            "experience":    experience,
            "work_type":     work_type,
            "qualification": qualification,    # "10th Pass" or "12th Pass"
            "category":      infer_category(title, tags),
            "url":           url,
            "source":        "Apna",
            "scraped_at":    datetime.now().isoformat(),
        })

    return jobs


# ── Page fetcher ───────────────────────────────────────────────────────────────

def fetch_page(url: str, session: requests.Session) -> str | None:
    """Fetch a URL and return HTML, or None on failure."""
    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            return resp.text
        log.warning(f"HTTP {resp.status_code} for {url}")
        return None
    except requests.RequestException as e:
        log.error(f"Request failed for {url}: {e}")
        return None


# ── Main scraper ───────────────────────────────────────────────────────────────

def scrape_apna(max_pages: int = 3, output_file: str = "bharat_access_hub/data/jobs.json") -> list[dict]:
    """
    Scrape both 10th-pass and 12th-pass job pages from Apna.co.

    Apna paginates with ?page=2, ?page=3 etc.
    """
    session   = requests.Session()
    all_jobs  = []
    seen_urls = set()

    for base_url, qualification in SCRAPE_URLS:
    # Extract city name from URL for logging
        city_match = re.search(r'jobs-in-(\w+)', base_url)
        city_label = city_match.group(1).capitalize() if city_match else "National"
        log.info(f"\nScraping: {qualification} / {city_label} — {base_url}")
        
        for page in range(1, max_pages + 1):
            url = base_url if page == 1 else f"{base_url}?page={page}"
            log.info(f"  Page {page}: {url}")

            html = fetch_page(url, session)
            if not html:
                log.warning(f"  Skipping page {page} — fetch failed")
                break

            jobs = parse_apna_page(html, qualification)

            if not jobs:
                log.info(f"  No jobs found on page {page}, stopping.")
                break

            new = [j for j in jobs if j["url"] not in seen_urls]
            seen_urls.update(j["url"] for j in new)
            all_jobs.extend(new)

            log.info(f"  +{len(new)} new jobs (total so far: {len(all_jobs)})")

            if len(new) == 0:
                log.info("  No new jobs on this page, stopping pagination.")
                break

            # Polite delay between pages
            time.sleep(random.uniform(1.5, 3.0))

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "source":         "apna.co",
            "qualifications": ["10th Pass", "12th Pass"],
            "cities": list({re.search(r'jobs-in-(\w+)', u).group(1) for u, _ in SCRAPE_URLS if re.search(r'jobs-in-(\w+)', u)}),
            "scraped_at":     datetime.now().isoformat(),
        },
        "jobs": all_jobs,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    log.info(f"\nDone! {len(all_jobs)} Apna jobs saved to '{output_path}'")

    cats = Counter(j["category"] for j in all_jobs)
    log.info("\nCategory breakdown:")
    for cat, count in cats.most_common():
        log.info(f"  {cat}: {count}")

    quals = Counter(j["qualification"] for j in all_jobs)
    log.info("\nQualification breakdown:")
    for q, count in quals.most_common():
        log.info(f"  {q}: {count}")

    return all_jobs


# ── Merge into existing jobs.json ─────────────────────────────────────────────

def merge_into_jobs_json(apna_file: str, jobs_file: str):
    """
    Merge Apna jobs into the main jobs.json (from OLX scraper).
    Deduplicates by URL.
    """
    apna_path = Path(apna_file)
    jobs_path = Path(jobs_file)

    if not apna_path.exists():
        log.error(f"Apna file not found: {apna_file}")
        return
    if not jobs_path.exists():
        log.error(f"Jobs file not found: {jobs_file}")
        return

    with open(apna_path, encoding="utf-8") as f:
        apna_data = json.load(f)
    with open(jobs_path, encoding="utf-8") as f:
        jobs_data = json.load(f)

    apna_jobs = apna_data.get("jobs", apna_data) if isinstance(apna_data, dict) else apna_data
    existing  = jobs_data.get("jobs", jobs_data) if isinstance(jobs_data, dict) else jobs_data
    metadata  = jobs_data.get("metadata", {}) if isinstance(jobs_data, dict) else {}

    existing_urls = {j["url"] for j in existing}
    new_jobs = [j for j in apna_jobs if j["url"] not in existing_urls]

    merged = existing + new_jobs
    metadata["total_listings"] = len(merged)
    metadata["sources"]        = list(set(metadata.get("sources", ["olx.in"]) + ["apna.co"]))
    metadata["merged_at"]      = datetime.now().isoformat()

    with open(jobs_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "jobs": merged}, f, ensure_ascii=False, indent=2)

    log.info(f"Merged {len(new_jobs)} new Apna jobs into '{jobs_file}'")
    log.info(f"Total jobs in {jobs_file}: {len(merged)}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scrape job listings from Apna.co")
    parser.add_argument("--pages",  "-p", type=int, default=3,
                        help="Pages to scrape per qualification (default: 3)")
    parser.add_argument("--output", "-o", default="bharat_access_hub/data/jobs.json",
                        help="Output file path (default: jobs.json)")
    parser.add_argument("--merge",  "-m", metavar="JOBS_JSON", default=None,
                        help="After scraping, merge results into this existing jobs.json")
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  Apna.co Jobs Scraper")
    print("=" * 55)
    print(f"  Qualifications : 10th Pass, 12th Pass")
    print(f"  Pages          : {args.pages} per qualification")
    print(f"  Output         : {args.output}")
    if args.merge:
        print(f"  Merge into     : {args.merge}")
    print("=" * 55 + "\n")

    scrape_apna(max_pages=args.pages, output_file=args.output)

    if args.merge:
        print(f"\nMerging into {args.merge}...")
        merge_into_jobs_json(args.output, args.merge)


if __name__ == "__main__":
    main()
