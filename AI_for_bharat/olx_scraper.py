"""
OLX India Jobs Scraper
======================
Scrapes job listings from olx.in/india/jobs and saves to JSON.
Filters out non-job listings and listings older than 3 months.

Usage:
    python olx_jobs_scraper.py
    python olx_jobs_scraper.py --pages 5 --output bharat_access_hub/data/jobs.json
"""

import json
import time
import random
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_URL = "https://www.olx.in"
JOBS_URL = "https://www.olx.in/india/jobs"

CITIES = [
    "mumbai", "delhi", "bangalore", "hyderabad", "chennai",
    "pune", "kolkata", "ahmedabad", "jaipur", "lucknow",
]

# URL keywords that indicate a real job listing
NON_JOB_PATHS = [
    "/cars-", "/lands-", "/mobiles-", "/electronics-", 
    "/furniture-", "/bikes-", "/spare-parts-", "/pets-",
    "/fashion-", "/books-", "/sports-", "/motorcycles-",
]


# ── Driver setup ───────────────────────────────────────────────────────────────

def make_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=options
    )
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver


# ── Date parsing ───────────────────────────────────────────────────────────────

def parse_date(date_str):
    """Parse OLX date strings: 'Yesterday', 'Feb 14', 'Today', '4 days ago'."""
    if not date_str or date_str == "N/A":
        return None

    now = datetime.now()
    s = date_str.strip().lower()

    if s == "today":
        return now
    if s == "yesterday":
        return now - timedelta(days=1)
    if "days ago" in s:
        try:
            return now - timedelta(days=int(s.split()[0]))
        except ValueError:
            pass
    if "weeks ago" in s:
        try:
            return now - timedelta(weeks=int(s.split()[0]))
        except ValueError:
            pass

    for fmt in ("%b %d", "%B %d"):
        try:
            parsed = datetime.strptime(date_str.strip(), fmt).replace(year=now.year)
            if parsed > now:
                parsed = parsed.replace(year=now.year - 1)
            return parsed
        except ValueError:
            continue

    return None


def is_within_months(date_str, months=3):
    """Return True if the listing is within the last N months."""
    parsed = parse_date(date_str)
    if parsed is None:
        return True
    return parsed >= datetime.now() - timedelta(days=months * 30)


# ── Category inference ─────────────────────────────────────────────────────────

def infer_category(url, title):
    """Infer job category from URL and title."""
    text = (url + " " + title).lower()

    if any(x in text for x in ["driver", "uber", "ola", "cab"]):
        return "Driver"
    if any(x in text for x in ["cook", "chef", "kitchen"]):
        return "Cook / Chef"
    if any(x in text for x in ["delivery", "courier", "logistics"]):
        return "Delivery"
    if any(x in text for x in ["security", "guard", "watchman"]):
        return "Security"
    if any(x in text for x in ["sales", "marketing", "telecaller", "bde"]):
        return "Sales & Marketing"
    if any(x in text for x in ["it-", "software", "developer", "programmer"]):
        return "IT & Software"
    if any(x in text for x in ["account", "finance", "tally", "gst"]):
        return "Accounting & Finance"
    if any(x in text for x in ["teach", "tutor", "faculty", "trainer", "school"]):
        return "Teaching & Education"
    if any(x in text for x in ["factory", "manufactur", "production", "operator"]):
        return "Factory & Manufacturing"
    if any(x in text for x in ["maid", "domestic", "housekeep", "caretaker", "nanny"]):
        return "Domestic Help"
    if any(x in text for x in ["construct", "labour", "mason", "plumber", "electrician"]):
        return "Construction & Labour"
    if any(x in text for x in ["nurse", "doctor", "hospital", "medical"]):
        return "Healthcare"
    if any(x in text for x in ["hotel", "waiter", "hospitality", "restaurant"]):
        return "Hospitality"
    if any(x in text for x in ["manager", "supervisor", "executive", "officer", "assistant"]):
        return "Office & Admin"

    return "General"


# ── Parsing ────────────────────────────────────────────────────────────────────

def parse_jobs(html, city=""):
    """Extract job listings from page HTML."""
    soup = BeautifulSoup(html, "lxml")
    cards = soup.select('[data-aut-id="itemBox3"]')
    jobs = []

    for card in cards:
        title_tag = card.select_one('[data-aut-id="itemTitle"]')
        price_tag = card.select_one('[data-aut-id="itemPrice"]')
        location_tag = card.select_one('[data-aut-id="item-location"]')
        date_tag = card.select_one('[data-aut-id="itemDate"]')
        link_tag = card.select_one("a[href]")

        title = title_tag.text.strip() if title_tag else "N/A"
        salary = price_tag.text.strip() if price_tag else "Not specified"
        location = location_tag.text.strip() if location_tag else city or "N/A"
        date_str = date_tag.text.strip() if date_tag else "N/A"
        href = link_tag["href"] if link_tag else ""
        url = href if href.startswith("http") else BASE_URL + href

        # Skip if no title
        if title == "N/A":
            continue

        if any(x in url.lower() for x in NON_JOB_PATHS):
        	continue

        # Filter by date
        if not is_within_months(date_str, months=3):
            log.info(f"  Skipping old listing: {title} ({date_str})")
            continue

        jobs.append({
            "title": title,
            "salary": salary,
            "location": location,
            "city": city,
            "date_posted": date_str,
            "category": infer_category(url, title),
            "url": url,
            "source": "OLX",
            "scraped_at": datetime.now().isoformat(),
        })

    return jobs


# ── Scraper ────────────────────────────────────────────────────────────────────

def scrape_city(driver, city, max_pages=2):
    """Scrape jobs for a specific city."""
    city_url = f"https://www.olx.in/{city}/jobs_c4"
    all_jobs = []
    seen_urls = set()

    for page in range(1, max_pages + 1):
        url = city_url if page == 1 else f"{city_url}?page={page}"
        log.info(f"  [{city}] Page {page}: {url}")

        try:
            driver.get(url)
            time.sleep(random.uniform(3, 5))
        except Exception as e:
            log.error(f"  Failed to load {url}: {e}")
            break

        jobs = parse_jobs(driver.page_source, city=city)
        new = [j for j in jobs if j["url"] not in seen_urls]
        seen_urls.update(j["url"] for j in new)
        all_jobs.extend(new)

        log.info(f"  [{city}] Page {page}: +{len(new)} jobs (total: {len(all_jobs)})")

        if len(new) == 0:
            log.info(f"  [{city}] No new jobs, stopping.")
            break

    return all_jobs


def scrape_all(cities=None, max_pages=2, output_file="bharat_access_hub/data/jobs.json"):
    """Main scraper — scrapes all cities and saves to JSON."""
    if cities is None:
        cities = CITIES

    driver = make_driver()
    all_jobs = []
    seen_urls = set()

    try:
        log.info("Scraping national jobs feed...")
        driver.get(JOBS_URL)
        time.sleep(random.uniform(3, 5))
        national_jobs = parse_jobs(driver.page_source, city="")
        new = [j for j in national_jobs if j["url"] not in seen_urls]
        seen_urls.update(j["url"] for j in new)
        all_jobs.extend(new)
        log.info(f"National feed: {len(new)} jobs")

        for city in cities:
            log.info(f"\nScraping city: {city}")
            city_jobs = scrape_city(driver, city, max_pages=max_pages)
            new = [j for j in city_jobs if j["url"] not in seen_urls]
            seen_urls.update(j["url"] for j in new)
            all_jobs.extend(new)
            log.info(f"City {city}: +{len(new)} unique jobs")
            time.sleep(random.uniform(2, 4))

    finally:
        driver.quit()

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "source": "olx.in",
            "cities": cities,
            "total_listings": len(all_jobs),
            "scraped_at": datetime.now().isoformat(),
            "max_age_months": 3,
        },
        "jobs": all_jobs,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    log.info(f"\nDone! {len(all_jobs)} jobs saved to '{output_path}'")

    cats = Counter(j["category"] for j in all_jobs)
    log.info("\nCategory breakdown:")
    for cat, count in cats.most_common():
        log.info(f"  {cat}: {count}")

    return all_jobs


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scrape job listings from OLX India")
    parser.add_argument("--pages", "-p", type=int, default=2)
    parser.add_argument("--cities", "-c", nargs="+", default=None)
    parser.add_argument("--output", "-o", default="bharat_access_hub/data/jobs.json")
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  OLX India Jobs Scraper")
    print("=" * 55)
    print(f"  Cities : {args.cities or '10 major cities'}")
    print(f"  Pages  : {args.pages} per city")
    print(f"  Output : {args.output}")
    print(f"  Filter : last 3 months only")
    print("=" * 55 + "\n")

    scrape_all(cities=args.cities, max_pages=args.pages, output_file=args.output)


if __name__ == "__main__":
    main()
