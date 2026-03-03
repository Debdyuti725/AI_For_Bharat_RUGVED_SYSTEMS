"""
JustDial Job/Service Scraper — Scrapes skilled trade & labor listings.

Usage:
    python scrapers/scrape_justdial.py

Output:
    scrapers/justdial_jobs.json
"""

import json
import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import random

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "justdial_jobs.json")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-IN,en;q=0.9",
}

# JustDial search URLs for service providers
JUSTDIAL_SEARCHES = [
    "https://www.justdial.com/Delhi/Labour-Contractors",
    "https://www.justdial.com/Mumbai/Daily-Wage-Workers",
    "https://www.justdial.com/Bangalore/Plumbers",
    "https://www.justdial.com/Chennai/Electricians",
]


def scrape_justdial():
    """Try scraping JustDial. Returns list of job dicts."""
    jobs = []
    for url in JUSTDIAL_SEARCHES:
        try:
            print(f"[JustDial] Scraping: {url}")
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                print(f"[JustDial] Got status {resp.status_code}, skipping...")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            listings = soup.select(".resultbox_info") or soup.select(".store-details")
            print(f"[JustDial] Found {len(listings)} listings")

            for item in listings[:15]:
                name_el = item.select_one(".resultbox_title_anchor") or item.select_one("h2")
                addr_el = item.select_one(".resultbox_address") or item.select_one(".address")
                rating_el = item.select_one(".green-box") or item.select_one(".rating")

                if name_el:
                    jobs.append({
                        "title": name_el.get_text(strip=True),
                        "location": addr_el.get_text(strip=True) if addr_el else "",
                        "rating": rating_el.get_text(strip=True) if rating_el else "",
                        "source": "JustDial",
                    })
        except Exception as e:
            print(f"[JustDial] Error: {e}")

    return jobs


def generate_sample_justdial_jobs():
    """Generate realistic JustDial job listings for prototype."""
    cities = {
        "maharashtra": ["Mumbai", "Pune", "Nagpur", "Aurangabad"],
        "karnataka": ["Bangalore", "Mysore", "Belgaum"],
        "tamil_nadu": ["Chennai", "Coimbatore", "Trichy"],
        "delhi": ["New Delhi", "Gurgaon", "Faridabad"],
        "uttar_pradesh": ["Lucknow", "Noida", "Ghaziabad"],
        "rajasthan": ["Jaipur", "Udaipur"],
        "west_bengal": ["Kolkata", "Howrah"],
        "telangana": ["Hyderabad", "Secunderabad"],
        "gujarat": ["Ahmedabad", "Surat"],
        "kerala": ["Kochi", "Trivandrum"],
    }

    job_templates = [
        {"title": "Plumber Required – Urgent", "category": "plumber", "salary_min": 500, "salary_max": 800, "desc": "Need experienced plumber for bathroom renovation. Pipe fitting, tap installation, and drainage work.", "req": ["Pipe fitting experience", "Own tools", "Available immediately"], "daily": True},
        {"title": "Electrician for Shop Wiring", "category": "electrician", "salary_min": 600, "salary_max": 1000, "desc": "Electrician needed for new shop wiring and light fitting. Must have experience with commercial setups.", "req": ["Commercial wiring experience", "Safety knowledge", "Own tools"], "daily": True},
        {"title": "House Painting Work", "category": "painter", "salary_min": 500, "salary_max": 800, "desc": "Need painters for 2BHK flat painting. Asian Paints / Berger experience. Wall putty and primer work included.", "req": ["Interior painting experience", "Knowledge of paint brands", "Can start this week"], "daily": True},
        {"title": "AC Service & Installation", "category": "technician", "salary_min": 15000, "salary_max": 25000, "desc": "Hiring AC service technician. Must know Split AC installation, gas charging, and PCB repair.", "req": ["AC servicing experience", "Split AC knowledge", "Gas charging skills"]},
        {"title": "Carpenter – Kitchen Cabinet Work", "category": "carpenter", "salary_min": 600, "salary_max": 1000, "desc": "Carpenter needed for modular kitchen cabinet installation. Should know laminate and hardware fitting.", "req": ["Modular kitchen experience", "Measurement skills", "Own tools"], "daily": True},
        {"title": "Welder for Gate Fabrication", "category": "welder", "salary_min": 500, "salary_max": 900, "desc": "Skilled welder needed for iron gate and grille fabrication. Arc welding and cutting torch experience required.", "req": ["Arc welding skills", "Fabrication experience", "Safety gear"], "daily": True},
        {"title": "CCTV Camera Installation", "category": "technician", "salary_min": 12000, "salary_max": 18000, "desc": "Technician for CCTV camera installation and networking. IP camera and DVR configuration knowledge needed.", "req": ["CCTV installation experience", "Networking basics", "Height comfortable"]},
        {"title": "Tile & Marble Fitting Work", "category": "mason", "salary_min": 600, "salary_max": 1000, "desc": "Mason needed for floor tile and marble fitting in new apartment. Vitrified and granite experience preferred.", "req": ["Tile cutting skills", "Level and measurement", "Clean finishing"], "daily": True},
        {"title": "Mobile Phone Repair Technician", "category": "technician", "salary_min": 10000, "salary_max": 16000, "desc": "Mobile repair technician for shop. Should know screen replacement, battery change, and software issues.", "req": ["Mobile repair skills", "Microsoldering basic", "Customer handling"]},
        {"title": "RO Water Purifier Service", "category": "technician", "salary_min": 10000, "salary_max": 15000, "desc": "Technician for RO service and installation. Filter change, membrane replacement knowledge needed.", "req": ["RO service experience", "Basic plumbing", "Customer area visits"]},
        {"title": "Mason for House Construction", "category": "mason", "salary_min": 700, "salary_max": 1200, "desc": "Experienced mason for residential construction. Brick laying, plastering, and foundation work.", "req": ["Construction experience", "Can read basic plans", "Team management"], "daily": True},
        {"title": "Pest Control Technician", "category": "pest_control", "salary_min": 10000, "salary_max": 14000, "desc": "Pest control technician for residential and commercial service. Training provided for right candidate.", "req": ["Basic science knowledge", "Customer friendly", "Can handle chemicals"]},
        {"title": "Furniture Assembly & Repair", "category": "carpenter", "salary_min": 400, "salary_max": 700, "desc": "Need someone for flatpack furniture assembly and minor wood repairs. IKEA/Godrej furniture experience preferred.", "req": ["Assembly skills", "Own basic tools", "Careful with furniture"], "daily": True},
        {"title": "Glass & Aluminium Work", "category": "glazier", "salary_min": 14000, "salary_max": 22000, "desc": "Experienced glass fitter for aluminium window and partition work. Should know sliding and UPVC fitting.", "req": ["Glass fitting experience", "Aluminium work", "Measurement precision"]},
        {"title": "Washing Machine Repair", "category": "technician", "salary_min": 11000, "salary_max": 16000, "desc": "Service technician for washing machine repair. All brands – Samsung, LG, Whirlpool. Home visits required.", "req": ["Appliance repair knowledge", "Two-wheeler for travel", "Brand knowledge"]},
    ]

    jobs = []
    job_id = 1

    for state, city_list in cities.items():
        selected = random.sample(job_templates, min(4, len(job_templates)))
        for template in selected:
            city = random.choice(city_list)
            days_ago = random.randint(0, 10)
            posted = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")

            is_daily = template.get("daily", False)
            salary_min = template["salary_min"]
            salary_max = template["salary_max"]

            jobs.append({
                "job_id": f"jd_{job_id:03d}",
                "title": template["title"],
                "category": template["category"],
                "location": city,
                "state": state,
                "salary_min": salary_min,
                "salary_max": salary_max,
                "salary_type": "daily" if is_daily else "monthly",
                "salary_display": f"₹{salary_min}-{salary_max}/{'day' if is_daily else 'month'}",
                "description": template["desc"],
                "requirements": template["req"],
                "contact": f"+91 {random.randint(70000, 99999)}{random.randint(10000, 99999)}",
                "source": "JustDial",
                "source_url": f"https://www.justdial.com/listing/{job_id}",
                "posted_date": posted,
                "job_type": "daily_wage" if is_daily else "full_time",
            })
            job_id += 1

    return jobs


def main():
    print("=" * 60)
    print("JustDial Job Scraper — Bharat Access Hub")
    print("=" * 60)

    real_jobs = scrape_justdial()

    if real_jobs:
        print(f"\n[JustDial] Successfully scraped {len(real_jobs)} real listings!")
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(real_jobs, f, indent=2, ensure_ascii=False)
    else:
        print("\n[JustDial] Real scraping returned no results (site may block scraping).")
        print("[JustDial] Generating realistic sample data for prototype...")
        sample_jobs = generate_sample_justdial_jobs()
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(sample_jobs, f, indent=2, ensure_ascii=False)
        print(f"[JustDial] Generated {len(sample_jobs)} sample job listings.")

    print(f"[JustDial] Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
