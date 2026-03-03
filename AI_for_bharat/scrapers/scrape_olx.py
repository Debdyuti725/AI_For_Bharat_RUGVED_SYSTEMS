"""
OLX India Job Scraper — Scrapes daily wage / low-income job listings.

Usage:
    python scrapers/scrape_olx.py

Output:
    scrapers/olx_jobs.json
"""

import json
import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import random

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "olx_jobs.json")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-IN,en;q=0.9,hi;q=0.8",
}

# OLX Jobs URLs to scrape
OLX_URLS = [
    "https://www.olx.in/jobs_c976",
    "https://www.olx.in/jobs_c976?filter=category_eq_cp1217",  # Driver
    "https://www.olx.in/jobs_c976?filter=category_eq_cp1218",  # Delivery
    "https://www.olx.in/jobs_c976?filter=category_eq_cp1219",  # Labour
]


def scrape_olx():
    """Try scraping OLX. Returns list of job dicts."""
    jobs = []
    for url in OLX_URLS:
        try:
            print(f"[OLX] Scraping: {url}")
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                print(f"[OLX] Got status {resp.status_code}, skipping...")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            # OLX uses dynamic rendering, so HTML parsing may get limited results
            listings = soup.select('[data-aut-id="itemBox"]') or soup.select(".EIR5N")
            print(f"[OLX] Found {len(listings)} listings")

            for item in listings[:20]:
                title_el = item.select_one('[data-aut-id="itemTitle"]') or item.select_one("span")
                price_el = item.select_one('[data-aut-id="itemPrice"]') or item.select_one(".price")
                loc_el = item.select_one('[data-aut-id="item-location"]') or item.select_one(".location")
                link_el = item.select_one("a")

                if title_el:
                    jobs.append({
                        "title": title_el.get_text(strip=True),
                        "salary_text": price_el.get_text(strip=True) if price_el else "",
                        "location": loc_el.get_text(strip=True) if loc_el else "",
                        "url": "https://www.olx.in" + link_el["href"] if link_el and link_el.get("href") else "",
                        "source": "OLX",
                    })
        except Exception as e:
            print(f"[OLX] Error: {e}")

    return jobs


def generate_sample_olx_jobs():
    """Generate realistic sample OLX job listings for prototype."""
    cities = {
        "maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Thane"],
        "karnataka": ["Bangalore", "Mysore", "Hubli", "Mangalore"],
        "tamil_nadu": ["Chennai", "Coimbatore", "Madurai", "Salem"],
        "delhi": ["New Delhi", "Dwarka", "Rohini", "Noida"],
        "uttar_pradesh": ["Lucknow", "Kanpur", "Varanasi", "Agra"],
        "rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota"],
        "west_bengal": ["Kolkata", "Howrah", "Durgapur", "Siliguri"],
        "telangana": ["Hyderabad", "Warangal", "Secunderabad"],
        "gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot"],
        "kerala": ["Kochi", "Thiruvananthapuram", "Kozhikode"],
    }

    job_templates = [
        {"title": "Driver Needed – Personal Car", "category": "driver", "salary_min": 12000, "salary_max": 18000, "desc": "Looking for an experienced driver for daily commute. Must have valid license and know local routes.", "req": ["Valid driving license", "2+ years experience", "Know local roads"]},
        {"title": "Delivery Boy – E-commerce", "category": "delivery", "salary_min": 10000, "salary_max": 15000, "desc": "Delivery executive needed for parcel delivery. Own bike preferred. Flexible hours.", "req": ["Own bike preferred", "Smartphone required", "Know the city area"]},
        {"title": "Cook/Chef for Restaurant", "category": "cook", "salary_min": 12000, "salary_max": 20000, "desc": "Experienced cook needed for small restaurant. North Indian and South Indian cuisine knowledge required.", "req": ["Cooking experience", "Hygiene awareness", "Can work in shifts"]},
        {"title": "House Maid / Domestic Helper", "category": "domestic", "salary_min": 8000, "salary_max": 12000, "desc": "Looking for reliable domestic help for cleaning and cooking. Morning shift preferred.", "req": ["Reliable and punctual", "Cooking knowledge", "References preferred"]},
        {"title": "Security Guard – Night Shift", "category": "security", "salary_min": 10000, "salary_max": 14000, "desc": "Security guard needed for residential society. Night shift (8pm-8am). Uniform provided.", "req": ["Physically fit", "No criminal record", "Can work night shifts"]},
        {"title": "Construction Worker / Labour", "category": "construction", "salary_min": 400, "salary_max": 700, "desc": "Daily wage construction workers needed for building project. Experience in masonry or painting preferred.", "req": ["Physical fitness", "Basic construction skills", "Own tools preferred"], "daily": True},
        {"title": "Electrician – Residential Work", "category": "electrician", "salary_min": 15000, "salary_max": 25000, "desc": "Experienced electrician for residential wiring and maintenance. ITI certification preferred.", "req": ["ITI/Diploma in Electrical", "Own tools", "Experience in residential work"]},
        {"title": "Plumber for Apartment Complex", "category": "plumber", "salary_min": 12000, "salary_max": 20000, "desc": "Skilled plumber needed for apartment complex maintenance. Must know modern fittings.", "req": ["Plumbing experience", "Knowledge of modern fittings", "Own tools"]},
        {"title": "Auto Rickshaw Driver", "category": "driver", "salary_min": 500, "salary_max": 800, "desc": "Auto rickshaw driver needed on sharing basis. Vehicle provided. Daily earnings basis.", "req": ["Auto driving license", "Know city routes", "Good behavior with passengers"], "daily": True},
        {"title": "Watchman / Chowkidar", "category": "security", "salary_min": 8000, "salary_max": 12000, "desc": "Watchman needed for factory premises. 12-hour shifts, rotational. Accommodation available.", "req": ["Trustworthy", "Can do 12-hour shifts", "Basic Hindi/local language"]},
        {"title": "Tailor – Garment Stitching", "category": "tailor", "salary_min": 10000, "salary_max": 18000, "desc": "Experienced tailor needed for garment stitching shop. Should know machine stitching and alterations.", "req": ["Tailoring experience", "Machine stitching skills", "Alteration knowledge"]},
        {"title": "Painter – Home & Office", "category": "painter", "salary_min": 500, "salary_max": 900, "desc": "Skilled painter needed for home and office painting work. Daily wage basis. Experience with distemper and emulsion.", "req": ["Painting experience", "Own brushes/rollers", "Height comfortable"], "daily": True},
        {"title": "Gardener / Mali", "category": "gardener", "salary_min": 8000, "salary_max": 12000, "desc": "Gardener required for maintaining residential garden. Knowledge of local plants and lawn maintenance.", "req": ["Gardening knowledge", "Own tools", "Regular and punctual"]},
        {"title": "Warehouse Helper / Loader", "category": "warehouse", "salary_min": 10000, "salary_max": 14000, "desc": "Warehouse helper needed for loading/unloading goods. Must be physically strong. Shift timings flexible.", "req": ["Physical strength", "Can lift heavy items", "Available for shifts"]},
        {"title": "Office Boy / Peon", "category": "office", "salary_min": 8000, "salary_max": 12000, "desc": "Office boy needed for file delivery, tea service, and general office work. Basic education required.", "req": ["Basic reading/writing", "Polite behavior", "Punctual"]},
        {"title": "Bike Mechanic", "category": "mechanic", "salary_min": 12000, "salary_max": 18000, "desc": "Bike mechanic with experience in Hero, Honda, and Bajaj bikes. Workshop experience preferred.", "req": ["Bike repair experience", "Own tools preferred", "Brand knowledge"]},
        {"title": "Truck Driver – Long Distance", "category": "driver", "salary_min": 18000, "salary_max": 30000, "desc": "Experienced truck driver for interstate transport. Must have HMV license and highway driving experience.", "req": ["HMV license", "Highway driving experience", "Can travel long distances"]},
        {"title": "Carpenter – Furniture Making", "category": "carpenter", "salary_min": 15000, "salary_max": 22000, "desc": "Skilled carpenter needed for custom furniture orders. Should know modern design techniques.", "req": ["Carpentry skills", "Own tools", "Modern design knowledge"]},
        {"title": "Helper for Catering / Events", "category": "catering", "salary_min": 400, "salary_max": 600, "desc": "Helpers needed for catering and event setup. Occasional basis for weddings and parties.", "req": ["Willing to work late hours", "Physical fitness", "Team player"], "daily": True},
        {"title": "AC Repair Technician", "category": "technician", "salary_min": 14000, "salary_max": 22000, "desc": "AC technician needed for repair and maintenance. Knowledge of split and window AC required.", "req": ["AC repair experience", "Own tools", "Customer friendly"]},
        {"title": "Cleaner / Housekeeping Staff", "category": "cleaner", "salary_min": 8000, "salary_max": 11000, "desc": "Housekeeping staff required for office building. Morning shift. Uniform and meals provided.", "req": ["Cleaning experience", "Punctual", "Can follow instructions"]},
        {"title": "Welding Worker", "category": "welder", "salary_min": 13000, "salary_max": 20000, "desc": "Experienced welder needed for fabrication shop. Arc and gas welding knowledge required.", "req": ["Welding experience", "Safety awareness", "Own protective gear preferred"]},
        {"title": "Laundry / Ironing Worker", "category": "laundry", "salary_min": 7000, "salary_max": 10000, "desc": "Worker needed for laundry shop. Ironing and dry cleaning experience preferred.", "req": ["Ironing skills", "Careful with clothes", "Can work standing"]},
        {"title": "Farm Worker / Agricultural Labour", "category": "farming", "salary_min": 300, "salary_max": 500, "desc": "Farm workers needed for seasonal harvest work. Rice/wheat field experience preferred.", "req": ["Farming experience", "Can work in sun", "Physical fitness"], "daily": True},
        {"title": "Shop Assistant / Salesman", "category": "retail", "salary_min": 8000, "salary_max": 13000, "desc": "Shop assistant needed for grocery/general store. Must be well-mannered and can handle cash.", "req": ["Basic math skills", "Customer handling", "Honest and reliable"]},
    ]

    jobs = []
    job_id = 1

    for state, city_list in cities.items():
        # Pick 3-5 random job templates per state
        selected = random.sample(job_templates, min(5, len(job_templates)))
        for template in selected:
            city = random.choice(city_list)
            days_ago = random.randint(0, 14)
            posted = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")

            is_daily = template.get("daily", False)
            salary_min = template["salary_min"]
            salary_max = template["salary_max"]

            jobs.append({
                "job_id": f"olx_{job_id:03d}",
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
                "source": "OLX",
                "source_url": f"https://www.olx.in/item/job-{job_id}",
                "posted_date": posted,
                "job_type": "daily_wage" if is_daily else "full_time",
            })
            job_id += 1

    return jobs


def main():
    print("=" * 60)
    print("OLX Job Scraper — Bharat Access Hub")
    print("=" * 60)

    # Try real scraping first
    real_jobs = scrape_olx()

    if real_jobs:
        print(f"\n[OLX] Successfully scraped {len(real_jobs)} real jobs!")
        # Save raw scraped data
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(real_jobs, f, indent=2, ensure_ascii=False)
    else:
        print("\n[OLX] Real scraping returned no results (site may block scraping).")
        print("[OLX] Generating realistic sample data for prototype...")
        sample_jobs = generate_sample_olx_jobs()
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(sample_jobs, f, indent=2, ensure_ascii=False)
        print(f"[OLX] Generated {len(sample_jobs)} sample job listings.")

    print(f"[OLX] Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
