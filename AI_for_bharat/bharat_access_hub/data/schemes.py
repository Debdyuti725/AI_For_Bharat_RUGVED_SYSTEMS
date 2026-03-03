"""
Scheme Database — 15 real Indian government schemes as structured Python dicts.

Each scheme has:
  - scheme_id, name, category
  - benefit_amount (INR), benefit_type
  - eligibility_criteria (dict with typed conditions)
  - required_documents, application_url, deadline
"""

from typing import List, Dict, Any


SCHEMES: List[Dict[str, Any]] = [

    # ─── AGRICULTURE ──────────────────────────────────────────────────────────

    {
        "scheme_id": "PM-KISAN",
        "name": "Pradhan Mantri Kisan Samman Nidhi (PM-KISAN)",
        "category": "agriculture",
        "description": (
            "Provides ₹6,000 per year in three equal installments of ₹2,000 "
            "directly to farmers' bank accounts as income support."
        ),
        "benefit_amount": 6000,
        "benefit_type": "Direct Benefit Transfer (DBT)",
        "application_url": "https://pmkisan.gov.in/",
        "deadline": None,  # Ongoing
        "eligibility_criteria": {
            "employment_status": ["farmer"],
            "owns_land": True,
            "land_area_max_acres": 5.0,          # No upper cap officially, but targeting small/marginal
            "income_max": None,                   # No income ceiling
            "states": ["all"],
            "area_type": ["rural"],
        },
        "required_documents": [
            "Aadhaar Card", "Land Records / Khatoni", "Bank Account Details",
            "Mobile Number linked to Aadhaar",
        ],
        "weights_override": None,
    },

    {
        "scheme_id": "PMFBY",
        "name": "Pradhan Mantri Fasal Bima Yojana (PMFBY)",
        "category": "agriculture",
        "description": (
            "Comprehensive crop insurance scheme. Farmers pay only 2% premium for Kharif, "
            "1.5% for Rabi crops. Government subsidises rest."
        ),
        "benefit_amount": 200000,   # Up to ₹2 lakh claim
        "benefit_type": "Crop Insurance",
        "application_url": "https://pmfby.gov.in/",
        "deadline": None,
        "eligibility_criteria": {
            "employment_status": ["farmer"],
            "owns_land": True,
            "states": ["all"],
            "area_type": ["rural", "urban"],
        },
        "required_documents": [
            "Aadhaar Card", "Land Records", "Bank Account", "Sowing Certificate",
        ],
        "weights_override": None,
    },

    {
        "scheme_id": "KCC",
        "name": "Kisan Credit Card (KCC)",
        "category": "agriculture",
        "description": (
            "Provides farmers with timely and adequate credit for agriculture needs "
            "at subsidised interest rates (4% effective)."
        ),
        "benefit_amount": 300000,   # Up to ₹3 lakh at 4%
        "benefit_type": "Subsidised Credit",
        "application_url": "https://www.nabard.org/auth/writereaddata/tender/1806184721KCC%20Revised%20Scheme.pdf",
        "deadline": None,
        "eligibility_criteria": {
            "employment_status": ["farmer"],
            "owns_land": True,
            "states": ["all"],
            "area_type": ["rural", "urban"],
        },
        "required_documents": [
            "Aadhaar Card", "Land Records", "Passport Photo", "Bank Account",
        ],
        "weights_override": None,
    },

    # ─── EDUCATION ────────────────────────────────────────────────────────────

    {
        "scheme_id": "NSP-POST-MATRIC",
        "name": "National Scholarship Portal — Post-Matric Scholarship (SC/ST/OBC)",
        "category": "education",
        "description": (
            "Merit-cum-means based scholarship for SC, ST, and OBC students "
            "pursuing post-matric (11th onwards) education."
        ),
        "benefit_amount": 12000,    # Up to ₹12,000/year
        "benefit_type": "Scholarship (Annual)",
        "application_url": "https://scholarships.gov.in/",
        "deadline": "2025-11-30",
        "eligibility_criteria": {
            "category": ["sc", "st", "obc"],
            "income_max": 250000,               # ₹2.5 lakh for OBC, ₹2L for SC/ST
            "education_level": ["12th", "graduate", "postgraduate", "doctorate"],
            "currently_enrolled": True,
            "states": ["all"],
        },
        "required_documents": [
            "Aadhaar Card", "Caste Certificate", "Income Certificate",
            "Previous Year Marksheet", "Institution Verification",
        ],
        "weights_override": None,
    },

    {
        "scheme_id": "PM-SCHOLARSHIP",
        "name": "PM Scholarship Scheme for Central Armed Police Forces",
        "category": "education",
        "description": (
            "Scholarships for wards of ex/serving personnel of CAPF. "
            "₹3,000/month for boys, ₹3,250/month for girls."
        ),
        "benefit_amount": 36000,    # ₹3,000 × 12 months
        "benefit_type": "Monthly Scholarship",
        "application_url": "https://ksb.gov.in/pm-scholarship.htm",
        "deadline": "2025-10-31",
        "eligibility_criteria": {
            "education_level": ["12th", "graduate"],
            "currently_enrolled": True,
            "states": ["all"],
            "academic_percentage_min": 60.0,
        },
        "required_documents": [
            "Aadhaar Card", "Service Certificate of Parent", "Marksheet",
            "Bank Account",
        ],
        "weights_override": None,
    },

    {
        "scheme_id": "PMKVY",
        "name": "Pradhan Mantri Kaushal Vikas Yojana (PMKVY)",
        "category": "employment",
        "description": (
            "Free skill development training and certification. "
            "Monetary reward on passing assessment + placement support."
        ),
        "benefit_amount": 8000,     # Average reward on certification
        "benefit_type": "Free Training + Certification + Cash Reward",
        "application_url": "https://www.pmkvyofficial.org/",
        "deadline": None,
        "eligibility_criteria": {
            "age_min": 14,
            "age_max": 45,
            "employment_status": ["unemployed", "student", "self_employed", "farmer"],
            "states": ["all"],
        },
        "required_documents": [
            "Aadhaar Card", "Bank Account", "Passport Photo",
        ],
        "weights_override": None,
    },

    # ─── HOUSING ──────────────────────────────────────────────────────────────

    {
        "scheme_id": "PMAY-G",
        "name": "Pradhan Mantri Awaas Yojana — Gramin (PMAY-G)",
        "category": "housing",
        "description": (
            "Financial assistance of ₹1.2–1.3 lakh (plain areas) or ₹1.3–1.5 lakh "
            "(hilly / NE / difficult areas) to construct a pucca house."
        ),
        "benefit_amount": 130000,
        "benefit_type": "Grant for House Construction",
        "application_url": "https://pmayg.nic.in/",
        "deadline": None,
        "eligibility_criteria": {
            "area_type": ["rural"],
            "has_pucca_house": False,
            "income_max": None,                 # No ceiling; targeted via SECC
            "bpl_card": True,
            "states": ["all"],
        },
        "required_documents": [
            "Aadhaar Card", "SECC / BPL List inclusion proof",
            "Land Record", "Bank Account",
        ],
        "weights_override": None,
    },

    {
        "scheme_id": "PMAY-U",
        "name": "Pradhan Mantri Awaas Yojana — Urban (PMAY-U)",
        "category": "housing",
        "description": (
            "Credit Linked Subsidy Scheme (CLSS). Interest subsidy on home loans: "
            "EWS/LIG: 6.5% on ₹6L; MIG-I: 4% on ₹9L; MIG-II: 3% on ₹12L."
        ),
        "benefit_amount": 267000,   # Max subsidy value
        "benefit_type": "Interest Subsidy on Home Loan",
        "application_url": "https://pmaymis.gov.in/",
        "deadline": "2025-03-31",
        "eligibility_criteria": {
            "area_type": ["urban"],
            "owns_home": False,
            "income_max": 1800000,  # MIG-II ceiling ₹18 lakh
            "states": ["all"],
        },
        "required_documents": [
            "Aadhaar Card", "Income Certificate", "Bank Account",
            "Property Documents", "Passport Photo",
        ],
        "weights_override": None,
    },

    # ─── EMPLOYMENT / LIVELIHOOD ──────────────────────────────────────────────

    {
        "scheme_id": "MGNREGS",
        "name": "Mahatma Gandhi National Rural Employment Guarantee Scheme (MGNREGS)",
        "category": "employment",
        "description": (
            "Guarantees 100 days of wage employment per year to rural households. "
            "Current wage: ₹220–300/day depending on state."
        ),
        "benefit_amount": 26400,    # 100 days × ₹264 avg
        "benefit_type": "Wage Employment Guarantee",
        "application_url": "https://nrega.nic.in/",
        "deadline": None,
        "eligibility_criteria": {
            "area_type": ["rural"],
            "age_min": 18,
            "employment_status": ["unemployed", "farmer", "self_employed"],
            "states": ["all"],
        },
        "required_documents": [
            "Aadhaar Card", "Job Card (from Gram Panchayat)", "Bank / Post Office Account",
        ],
        "weights_override": None,
    },

    {
        "scheme_id": "PM-MUDRA",
        "name": "Pradhan Mantri MUDRA Yojana (PMMY)",
        "category": "employment",
        "description": (
            "Collateral-free micro loans to non-corporate, non-farm small/micro enterprises. "
            "3 categories: Shishu (up to ₹50K), Kishor (₹50K–5L), Tarun (₹5L–10L)."
        ),
        "benefit_amount": 1000000,  # Up to ₹10 lakh
        "benefit_type": "Collateral-Free Business Loan",
        "application_url": "https://www.mudra.org.in/",
        "deadline": None,
        "eligibility_criteria": {
            "employment_status": ["self_employed", "unemployed", "farmer"],
            "age_min": 18,
            "states": ["all"],
        },
        "required_documents": [
            "Aadhaar Card", "PAN Card", "Business Plan / Proof",
            "Bank Account", "Passport Photo",
        ],
        "weights_override": None,
    },

    {
        "scheme_id": "STAND-UP-INDIA",
        "name": "Stand Up India Scheme",
        "category": "employment",
        "description": (
            "Bank loans between ₹10 lakh and ₹1 crore for at least one SC/ST borrower "
            "and one woman borrower per bank branch for greenfield enterprise."
        ),
        "benefit_amount": 10000000,  # Up to ₹1 crore
        "benefit_type": "Business Loan",
        "application_url": "https://www.standupmitra.in/",
        "deadline": None,
        "eligibility_criteria": {
            "category": ["sc", "st"],
            "age_min": 18,
            "states": ["all"],
        },
        "required_documents": [
            "Aadhaar Card", "Caste Certificate", "Business Plan", "Bank Account",
        ],
        "weights_override": None,
    },

    {
        "scheme_id": "PM-SVANIDHI",
        "name": "PM Street Vendor's AtmaNirbhar Nidhi (PM SVANidhi)",
        "category": "employment",
        "description": (
            "Working capital loans of ₹10,000–₹50,000 to street vendors. "
            "Digital transactions incentivised with cashback."
        ),
        "benefit_amount": 50000,
        "benefit_type": "Working Capital Loan",
        "application_url": "https://pmsvanidhi.mohua.gov.in/",
        "deadline": None,
        "eligibility_criteria": {
            "area_type": ["urban"],
            "age_min": 18,
            "employment_status": ["self_employed", "unemployed"],
            "states": ["all"],
        },
        "required_documents": [
            "Aadhaar Card", "Vendor Certificate / Recommendation Letter",
            "Bank Account",
        ],
        "weights_override": None,
    },

    # ─── HEALTH ───────────────────────────────────────────────────────────────

    {
        "scheme_id": "AYUSHMAN-BHARAT",
        "name": "Ayushman Bharat — Pradhan Mantri Jan Arogya Yojana (PM-JAY)",
        "category": "health",
        "description": (
            "World's largest health insurance scheme. ₹5 lakh health coverage per family "
            "per year for secondary and tertiary hospitalisation."
        ),
        "benefit_amount": 500000,
        "benefit_type": "Health Insurance Coverage",
        "application_url": "https://pmjay.gov.in/",
        "deadline": None,
        "eligibility_criteria": {
            "bpl_card": True,
            "income_max": 100000,   # Approximate BPL income ceiling
            "states": ["all"],
        },
        "required_documents": [
            "Aadhaar Card", "Ration Card (BPL/SECC)", "Passport Photo",
        ],
        "weights_override": None,
    },

    {
        "scheme_id": "BETI-BACHAO",
        "name": "Beti Bachao Beti Padhao (BBBP)",
        "category": "education",
        "description": (
            "Financial incentives for girl child education and survival. "
            "Sukanya Samriddhi Yojana: open account for girl below 10 years."
        ),
        "benefit_amount": 0,        # Indirect – SSY account reaches maturity
        "benefit_type": "Girl Child Savings Scheme (Sukanya Samriddhi)",
        "application_url": "https://wcd.nic.in/bbbp-schemes",
        "deadline": None,
        "eligibility_criteria": {
            "gender": ["female"],
            "age_max": 10,          # For Sukanya Samriddhi
            "states": ["all"],
        },
        "required_documents": [
            "Aadhaar Card of Girl", "Birth Certificate", "Parent's Aadhaar",
            "Bank Account",
        ],
        "weights_override": None,
    },

    {
        "scheme_id": "RASHTRIYA-AROGYA-NIDHI",
        "name": "Rashtriya Arogya Nidhi (RAN)",
        "category": "health",
        "description": (
            "One-time financial assistance up to ₹15 lakh for BPL patients "
            "suffering from life-threatening illnesses requiring treatment at Government hospitals."
        ),
        "benefit_amount": 1500000,
        "benefit_type": "One-Time Medical Grant",
        "application_url": "https://mohfw.gov.in/schemes-and-programmes",
        "deadline": None,
        "eligibility_criteria": {
            "bpl_card": True,
            "chronic_illness": True,
            "income_max": 100000,
            "states": ["all"],
        },
        "required_documents": [
            "Aadhaar Card", "BPL Card", "Medical Certificate from Govt Hospital",
            "Bank Account",
        ],
        "weights_override": None,
    },

    # ─── EDUCATION — LOANS & SCHOLARSHIPS ─────────────────────────────────────

    {
        "scheme_id": "VIDYALAKSHMI",
        "name": "Vidya Lakshmi Education Loan Portal",
        "category": "education",
        "description": (
            "Single window for students to access education loans from multiple banks. "
            "Covers tuition fees, hostel, books, and equipment. Loans up to ₹20 lakh "
            "for studies in India, ₹30 lakh for abroad."
        ),
        "benefit_amount": 2000000,
        "benefit_type": "Education Loan",
        "application_url": "https://www.vidyalakshmi.co.in/",
        "deadline": None,
        "eligibility_criteria": {
            "education_level": ["12th", "graduate", "postgraduate"],
            "currently_enrolled": True,
            "age_min": 16,
            "age_max": 35,
            "states": ["all"],
        },
        "required_documents": [
            "Aadhaar Card", "Admission Letter", "Fee Structure",
            "Marksheets", "Bank Account", "Income Certificate",
        ],
        "how_to_apply": "Register on vidyalakshmi.co.in, fill common form, apply to multiple banks at once.",
        "weights_override": None,
    },

    {
        "scheme_id": "CENTRAL-SECTOR-SCHOLARSHIP",
        "name": "Central Sector Scheme of Scholarship (CSSS)",
        "category": "education",
        "description": (
            "Merit-based scholarship for college and university students. "
            "₹12,000/year for UG (first 3 years), ₹20,000/year for PG. "
            "Top 20 percentile of Class 12 board exam. Family income < ₹8 lakh."
        ),
        "benefit_amount": 20000,
        "benefit_type": "Annual Scholarship",
        "application_url": "https://scholarships.gov.in/",
        "deadline": "2025-11-30",
        "eligibility_criteria": {
            "education_level": ["12th", "graduate", "postgraduate"],
            "currently_enrolled": True,
            "income_max": 800000,
            "states": ["all"],
        },
        "required_documents": [
            "Aadhaar Card", "Class 12 Marksheet", "Income Certificate",
            "Bank Account", "College Admission Proof",
        ],
        "weights_override": None,
    },

    {
        "scheme_id": "PM-YASASVI",
        "name": "PM Young Achievers Scholarship (PM-YASASVI)",
        "category": "education",
        "description": (
            "Scholarship for OBC, EBC, and DNT students studying in top schools. "
            "₹75,000/year for Class 9-10, ₹1,25,000/year for Class 11-12."
        ),
        "benefit_amount": 125000,
        "benefit_type": "Annual Scholarship",
        "application_url": "https://yet.nta.ac.in/",
        "deadline": "2025-09-30",
        "eligibility_criteria": {
            "category": ["obc"],
            "income_max": 250000,
            "age_min": 14,
            "age_max": 18,
            "states": ["all"],
        },
        "required_documents": [
            "Aadhaar Card", "Caste Certificate", "Income Certificate",
            "School ID", "Marksheet",
        ],
        "weights_override": None,
    },

    {
        "scheme_id": "INTEREST-SUBSIDY-LOAN",
        "name": "Interest Subsidy on Education Loans (CSIS)",
        "category": "education",
        "description": (
            "Full interest subsidy during moratorium period (course + 1 year) "
            "on education loans up to ₹10 lakh for economically weaker students. "
            "Family income must be below ₹4.5 lakh/year."
        ),
        "benefit_amount": 100000,
        "benefit_type": "Interest Subsidy",
        "application_url": "https://www.education.gov.in/",
        "deadline": None,
        "eligibility_criteria": {
            "education_level": ["graduate", "postgraduate", "doctorate"],
            "currently_enrolled": True,
            "income_max": 450000,
            "states": ["all"],
        },
        "required_documents": [
            "Aadhaar Card", "Admission Letter", "Loan Sanction Letter",
            "Income Certificate", "Bank Account",
        ],
        "weights_override": None,
    },

    # ─── WOMEN WELFARE ────────────────────────────────────────────────────────

    {
        "scheme_id": "MAHILA-SAMMAN",
        "name": "Mahila Samman Savings Certificate",
        "category": "financial_aid",
        "description": (
            "Special savings scheme for women and girls. Deposit up to ₹2 lakh "
            "for 2 years at 7.5% interest rate. Partial withdrawal allowed after 1 year."
        ),
        "benefit_amount": 30000,
        "benefit_type": "Savings Interest",
        "application_url": "https://www.indiapost.gov.in/",
        "deadline": None,
        "eligibility_criteria": {
            "gender": ["female"],
            "states": ["all"],
        },
        "required_documents": [
            "Aadhaar Card", "PAN Card", "Passport Photo", "Bank Account",
        ],
        "weights_override": None,
    },

    {
        "scheme_id": "FREE-SEWING-MACHINE",
        "name": "Free Sewing Machine Scheme",
        "category": "employment",
        "description": (
            "Free sewing machines to poor and working women to help them become "
            "self-employed. Women aged 20-40 with family income below ₹12,000/month."
        ),
        "benefit_amount": 5000,
        "benefit_type": "Free Equipment",
        "application_url": "https://www.india.gov.in/",
        "deadline": None,
        "eligibility_criteria": {
            "gender": ["female"],
            "age_min": 20,
            "age_max": 40,
            "income_max": 144000,
            "states": ["all"],
        },
        "required_documents": [
            "Aadhaar Card", "Income Certificate", "Age Proof", "Passport Photo",
        ],
        "weights_override": None,
    },

    {
        "scheme_id": "UJJWALA",
        "name": "Pradhan Mantri Ujjwala Yojana (PMUY)",
        "category": "financial_aid",
        "description": (
            "Free LPG connections to women from BPL households. "
            "₹1,600 subsidy for new LPG connection and first refill."
        ),
        "benefit_amount": 1600,
        "benefit_type": "Free LPG Connection",
        "application_url": "https://www.pmuy.gov.in/",
        "deadline": None,
        "eligibility_criteria": {
            "bpl_card": True,
            "gender": ["female"],
            "states": ["all"],
        },
        "required_documents": [
            "Aadhaar Card", "BPL Card", "Passport Photo", "Bank Account",
        ],
        "weights_override": None,
    },

    # ─── SENIOR CITIZENS ──────────────────────────────────────────────────────

    {
        "scheme_id": "SENIOR-PENSION",
        "name": "Indira Gandhi National Old Age Pension Scheme (IGNOAPS)",
        "category": "financial_aid",
        "description": (
            "Monthly pension of ₹200 (age 60-79) or ₹500 (age 80+) for BPL elderly. "
            "States may add their own contribution on top."
        ),
        "benefit_amount": 6000,
        "benefit_type": "Monthly Pension",
        "application_url": "https://nsap.nic.in/",
        "deadline": None,
        "eligibility_criteria": {
            "age_min": 60,
            "bpl_card": True,
            "states": ["all"],
        },
        "required_documents": [
            "Aadhaar Card", "Age Proof", "BPL Card", "Bank Account",
        ],
        "weights_override": None,
    },

    {
        "scheme_id": "PMVVY",
        "name": "Pradhan Mantri Vaya Vandana Yojana (PMVVY)",
        "category": "financial_aid",
        "description": (
            "Guaranteed pension for senior citizens (60+). "
            "Invest ₹1.5 lakh - ₹15 lakh, get 7.4% annual return. "
            "Monthly/quarterly/annual pension options for 10 years."
        ),
        "benefit_amount": 111000,
        "benefit_type": "Pension/Investment",
        "application_url": "https://licindia.in/",
        "deadline": None,
        "eligibility_criteria": {
            "age_min": 60,
            "states": ["all"],
        },
        "required_documents": [
            "Aadhaar Card", "PAN Card", "Age Proof", "Bank Account",
        ],
        "weights_override": None,
    },

    # ─── DISABILITY ───────────────────────────────────────────────────────────

    {
        "scheme_id": "DISABILITY-PENSION",
        "name": "Indira Gandhi National Disability Pension Scheme (IGNDPS)",
        "category": "health",
        "description": (
            "Monthly pension of ₹300 for persons with severe/multiple disabilities "
            "(80%+) who are BPL. Age 18-79. ₹500/month for age 80+."
        ),
        "benefit_amount": 3600,
        "benefit_type": "Monthly Pension",
        "application_url": "https://nsap.nic.in/",
        "deadline": None,
        "eligibility_criteria": {
            "has_disability": True,
            "bpl_card": True,
            "age_min": 18,
            "states": ["all"],
        },
        "required_documents": [
            "Aadhaar Card", "Disability Certificate", "BPL Card", "Bank Account",
        ],
        "weights_override": None,
    },

    {
        "scheme_id": "ADIP",
        "name": "Assistance to Disabled Persons (ADIP Scheme)",
        "category": "health",
        "description": (
            "Free assistive devices like hearing aids, wheelchairs, artificial limbs, "
            "braille kits. Income limit: ₹20,000/month."
        ),
        "benefit_amount": 10000,
        "benefit_type": "Free Assistive Devices",
        "application_url": "https://www.nhfdc.nic.in/",
        "deadline": None,
        "eligibility_criteria": {
            "has_disability": True,
            "income_max": 240000,
            "states": ["all"],
        },
        "required_documents": [
            "Aadhaar Card", "Disability Certificate (40%+)", "Income Certificate",
            "Prescription from Govt Doctor",
        ],
        "weights_override": None,
    },

    # ─── STARTUP / ENTREPRENEURSHIP ───────────────────────────────────────────

    {
        "scheme_id": "STARTUP-INDIA",
        "name": "Startup India Seed Fund Scheme",
        "category": "employment",
        "description": (
            "Financial assistance up to ₹50 lakh for startups for proof of concept, "
            "prototype development, product trials, and market entry. "
            "Startups must be DPIIT recognized."
        ),
        "benefit_amount": 5000000,
        "benefit_type": "Seed Funding Grant",
        "application_url": "https://seedfund.startupindia.gov.in/",
        "deadline": None,
        "eligibility_criteria": {
            "employment_status": ["self_employed"],
            "age_min": 18,
            "states": ["all"],
        },
        "required_documents": [
            "DPIIT Recognition Certificate", "Business Plan",
            "PAN Card", "Bank Account", "Company Registration",
        ],
        "weights_override": None,
    },

    {
        "scheme_id": "PMEGP",
        "name": "PM Employment Generation Programme (PMEGP)",
        "category": "employment",
        "description": (
            "Credit-linked subsidy for setting up micro enterprises. "
            "Manufacturing: project up to ₹50 lakh. Service: up to ₹20 lakh. "
            "15-35% subsidy on project cost depending on category and area."
        ),
        "benefit_amount": 1000000,
        "benefit_type": "Subsidy on Business Loan",
        "application_url": "https://www.kviconline.gov.in/pmegpeportal/",
        "deadline": None,
        "eligibility_criteria": {
            "age_min": 18,
            "education_level": ["8th", "10th", "12th", "graduate"],
            "employment_status": ["unemployed", "self_employed"],
            "states": ["all"],
        },
        "required_documents": [
            "Aadhaar Card", "Project Report", "Caste Certificate (if applicable)",
            "Education Certificate", "Bank Account",
        ],
        "weights_override": None,
    },

    # ─── FINANCIAL AID / SOCIAL SECURITY ──────────────────────────────────────

    {
        "scheme_id": "PM-JEEVAN-JYOTI",
        "name": "Pradhan Mantri Jeevan Jyoti Bima Yojana (PMJJBY)",
        "category": "financial_aid",
        "description": (
            "Life insurance of ₹2 lakh at just ₹436/year premium. "
            "Auto-debit from bank account. Age 18-50."
        ),
        "benefit_amount": 200000,
        "benefit_type": "Life Insurance",
        "application_url": "https://www.jansuraksha.gov.in/",
        "deadline": None,
        "eligibility_criteria": {
            "age_min": 18,
            "age_max": 50,
            "states": ["all"],
        },
        "required_documents": [
            "Aadhaar Card", "Bank Account", "Nominee Details",
        ],
        "weights_override": None,
    },

    {
        "scheme_id": "PM-SURAKSHA",
        "name": "Pradhan Mantri Suraksha Bima Yojana (PMSBY)",
        "category": "financial_aid",
        "description": (
            "Accidental death and disability insurance of ₹2 lakh at ₹20/year. "
            "Covers accidental death, full disability (₹2L), partial disability (₹1L)."
        ),
        "benefit_amount": 200000,
        "benefit_type": "Accident Insurance",
        "application_url": "https://www.jansuraksha.gov.in/",
        "deadline": None,
        "eligibility_criteria": {
            "age_min": 18,
            "age_max": 70,
            "states": ["all"],
        },
        "required_documents": [
            "Aadhaar Card", "Bank Account", "Nominee Details",
        ],
        "weights_override": None,
    },

    {
        "scheme_id": "APY",
        "name": "Atal Pension Yojana (APY)",
        "category": "financial_aid",
        "description": (
            "Guaranteed minimum pension of ₹1,000-₹5,000/month after age 60. "
            "Monthly contribution starts from ₹42. Government co-contributes 50%."
        ),
        "benefit_amount": 60000,
        "benefit_type": "Pension Scheme",
        "application_url": "https://www.npscra.nsdl.co.in/",
        "deadline": None,
        "eligibility_criteria": {
            "age_min": 18,
            "age_max": 40,
            "states": ["all"],
        },
        "required_documents": [
            "Aadhaar Card", "Bank Account", "Mobile Number",
        ],
        "weights_override": None,
    },

    {
        "scheme_id": "SUKANYA-SAMRIDDHI",
        "name": "Sukanya Samriddhi Yojana (SSY)",
        "category": "financial_aid",
        "description": (
            "High-interest savings scheme for girl child. Currently 8.2% interest. "
            "Deposit ₹250-₹1.5 lakh/year. Tax benefits under 80C. "
            "Matures when girl turns 21."
        ),
        "benefit_amount": 0,
        "benefit_type": "Savings Scheme",
        "application_url": "https://www.indiapost.gov.in/",
        "deadline": None,
        "eligibility_criteria": {
            "gender": ["female"],
            "age_max": 10,
            "states": ["all"],
        },
        "required_documents": [
            "Girl's Birth Certificate", "Parent Aadhaar", "PAN Card",
            "Address Proof",
        ],
        "weights_override": None,
    },

    # ─── RURAL / INFRASTRUCTURE ───────────────────────────────────────────────

    {
        "scheme_id": "SWACHH-BHARAT",
        "name": "Swachh Bharat Mission — Gramin (SBM-G)",
        "category": "housing",
        "description": (
            "₹12,000 incentive for constructing individual household toilet. "
            "Additional ₹15,000 for solid/liquid waste management. "
            "For BPL/SC/ST/marginalised households in rural areas."
        ),
        "benefit_amount": 12000,
        "benefit_type": "Toilet Construction Grant",
        "application_url": "https://swachhbharatmission.gov.in/sbmcms/index.htm",
        "deadline": None,
        "eligibility_criteria": {
            "area_type": ["rural"],
            "bpl_card": True,
            "states": ["all"],
        },
        "required_documents": [
            "Aadhaar Card", "BPL Card", "Bank Account", "Photo of House",
        ],
        "weights_override": None,
    },

    {
        "scheme_id": "SAUBHAGYA",
        "name": "Saubhagya — PM Sahaj Bijli Har Ghar Yojana",
        "category": "housing",
        "description": (
            "Free electricity connection to all un-electrified households. "
            "Free for BPL households, ₹500 for others (payable in 10 installments)."
        ),
        "benefit_amount": 3000,
        "benefit_type": "Free Electricity Connection",
        "application_url": "https://saubhagya.gov.in/",
        "deadline": None,
        "eligibility_criteria": {
            "area_type": ["rural"],
            "states": ["all"],
        },
        "required_documents": [
            "Aadhaar Card", "Address Proof", "BPL Card (for free connection)",
        ],
        "weights_override": None,
    },

    # ─── FOOD SECURITY ────────────────────────────────────────────────────────

    {
        "scheme_id": "NFSA",
        "name": "National Food Security Act (NFSA) / Ration Card",
        "category": "financial_aid",
        "description": (
            "Subsidised food grains: 5 kg/person/month at ₹1-3/kg. "
            "Rice at ₹3/kg, wheat at ₹2/kg, coarse grains at ₹1/kg. "
            "Covers 75% rural and 50% urban population."
        ),
        "benefit_amount": 5000,
        "benefit_type": "Subsidised Food",
        "application_url": "https://nfsa.gov.in/",
        "deadline": None,
        "eligibility_criteria": {
            "income_max": 300000,
            "states": ["all"],
        },
        "required_documents": [
            "Aadhaar Card", "Address Proof", "Income Certificate",
            "Family Photo",
        ],
        "weights_override": None,
    },

    {
        "scheme_id": "PM-GARIB-KALYAN",
        "name": "PM Garib Kalyan Anna Yojana (PMGKAY)",
        "category": "financial_aid",
        "description": (
            "5 kg free food grains per person per month to all NFSA beneficiaries. "
            "In addition to regular ration entitlement. Extended multiple times."
        ),
        "benefit_amount": 3000,
        "benefit_type": "Free Food Grains",
        "application_url": "https://nfsa.gov.in/",
        "deadline": None,
        "eligibility_criteria": {
            "bpl_card": True,
            "states": ["all"],
        },
        "required_documents": [
            "Ration Card", "Aadhaar Card",
        ],
        "weights_override": None,
    },

    # ─── DIGITAL & SKILL DEVELOPMENT ─────────────────────────────────────────

    {
        "scheme_id": "DIGITAL-INDIA",
        "name": "Digital India — Free Digital Literacy (PMGDISHA)",
        "category": "education",
        "description": (
            "Free computer and digital literacy training for rural households. "
            "One person per household. 20 hours training covering computer basics, "
            "internet, email, digital payments, and government services."
        ),
        "benefit_amount": 0,
        "benefit_type": "Free Training",
        "application_url": "https://www.pmgdisha.in/",
        "deadline": None,
        "eligibility_criteria": {
            "area_type": ["rural"],
            "age_min": 14,
            "age_max": 60,
            "states": ["all"],
        },
        "required_documents": [
            "Aadhaar Card", "Mobile Number",
        ],
        "weights_override": None,
    },

    {
        "scheme_id": "DDU-GKY",
        "name": "Deen Dayal Upadhyaya Grameen Kaushalya Yojana (DDU-GKY)",
        "category": "employment",
        "description": (
            "Placement-linked skill training for rural youth. "
            "Free training, food, lodging, and transport. "
            "Guaranteed placement after training with minimum ₹6,000/month salary."
        ),
        "benefit_amount": 72000,
        "benefit_type": "Free Training + Placement",
        "application_url": "https://ddugky.gov.in/",
        "deadline": None,
        "eligibility_criteria": {
            "area_type": ["rural"],
            "age_min": 15,
            "age_max": 35,
            "employment_status": ["unemployed", "farmer"],
            "states": ["all"],
        },
        "required_documents": [
            "Aadhaar Card", "Bank Account", "Age Proof", "Education Certificate",
        ],
        "weights_override": None,
    },

    # ─── STATE-LEVEL EXAMPLES ─────────────────────────────────────────────────

    {
        "scheme_id": "LADLI-BEHNA",
        "name": "Ladli Behna Yojana (Madhya Pradesh)",
        "category": "financial_aid",
        "description": (
            "₹1,250/month (₹15,000/year) to women aged 23-60 in MP. "
            "Family income below ₹2.5 lakh, land below 5 acres."
        ),
        "benefit_amount": 15000,
        "benefit_type": "Monthly Cash Transfer",
        "application_url": "https://ladlibehna.mp.gov.in/",
        "deadline": None,
        "eligibility_criteria": {
            "gender": ["female"],
            "age_min": 23,
            "age_max": 60,
            "income_max": 250000,
            "states": ["madhya_pradesh"],
        },
        "required_documents": [
            "Aadhaar Card", "Samagra ID", "Bank Account",
        ],
        "weights_override": None,
    },

    {
        "scheme_id": "KALIA",
        "name": "KALIA Scheme (Odisha)",
        "category": "agriculture",
        "description": (
            "₹10,000/year to small and marginal farmers for crop cultivation. "
            "₹12,500 one-time for landless agricultural labourers. "
            "Life insurance cover of ₹2 lakh."
        ),
        "benefit_amount": 10000,
        "benefit_type": "Income Support",
        "application_url": "https://kalia.odisha.gov.in/",
        "deadline": None,
        "eligibility_criteria": {
            "employment_status": ["farmer"],
            "states": ["odisha"],
        },
        "required_documents": [
            "Aadhaar Card", "Land Records", "Bank Account",
        ],
        "weights_override": None,
    },

    # ─── MORE EDUCATION SCHEMES ───────────────────────────────────────────────

    {
        "scheme_id": "ISHAN-UDAY",
        "name": "Ishan Uday Special Scholarship for NE Region",
        "category": "education",
        "description": "₹5,400/month for general degree, ₹7,800/month for technical/medical/professional courses for students from North East region.",
        "benefit_amount": 93600, "benefit_type": "Monthly Scholarship",
        "application_url": "https://scholarships.gov.in/",
        "deadline": "2025-10-31",
        "eligibility_criteria": {"income_max": 450000, "states": ["assam", "meghalaya", "manipur", "mizoram", "tripura", "nagaland", "arunachal_pradesh", "sikkim"]},
        "required_documents": ["Aadhaar Card", "Domicile Certificate", "Income Certificate", "Admission Proof"],
        "weights_override": None,
    },

    {
        "scheme_id": "PRAGATI-SCHOLARSHIP",
        "name": "PRAGATI Scholarship for Girl Students (AICTE)",
        "category": "education",
        "description": "₹50,000/year for girl students in technical education (degree/diploma). Up to 2 girls per family. Family income < ₹8 lakh.",
        "benefit_amount": 50000, "benefit_type": "Annual Scholarship",
        "application_url": "https://www.aicte-india.org/schemes/students-development-schemes/PRAGATI",
        "deadline": "2025-12-31",
        "eligibility_criteria": {"gender": ["female"], "income_max": 800000, "education_level": ["graduate"], "states": ["all"]},
        "required_documents": ["Aadhaar Card", "Income Certificate", "Admission Letter", "Bank Account"],
        "weights_override": None,
    },

    {
        "scheme_id": "SAKSHAM-SCHOLARSHIP",
        "name": "SAKSHAM Scholarship for Differently Abled (AICTE)",
        "category": "education",
        "description": "₹50,000/year for differently abled students in technical education. Disability must be 40%+. Family income < ₹8 lakh.",
        "benefit_amount": 50000, "benefit_type": "Annual Scholarship",
        "application_url": "https://www.aicte-india.org/schemes/students-development-schemes/SAKSHAM",
        "deadline": "2025-12-31",
        "eligibility_criteria": {"has_disability": True, "income_max": 800000, "education_level": ["graduate"], "states": ["all"]},
        "required_documents": ["Aadhaar Card", "Disability Certificate", "Income Certificate", "Admission Proof"],
        "weights_override": None,
    },

    {
        "scheme_id": "MINORITY-SCHOLARSHIP-PRE",
        "name": "Pre-Matric Scholarship for Minorities",
        "category": "education",
        "description": "₹5,700/year for minority students (Muslim, Christian, Sikh, Buddhist, Jain, Parsi) studying Class 1-10.",
        "benefit_amount": 5700, "benefit_type": "Annual Scholarship",
        "application_url": "https://scholarships.gov.in/",
        "deadline": "2025-11-30",
        "eligibility_criteria": {"income_max": 100000, "states": ["all"]},
        "required_documents": ["Aadhaar Card", "Minority Certificate", "Income Certificate", "School Certificate"],
        "weights_override": None,
    },

    {
        "scheme_id": "MINORITY-SCHOLARSHIP-POST",
        "name": "Post-Matric Scholarship for Minorities",
        "category": "education",
        "description": "Up to ₹25,000/year for minority students pursuing 11th, 12th, UG, PG, or professional courses.",
        "benefit_amount": 25000, "benefit_type": "Annual Scholarship",
        "application_url": "https://scholarships.gov.in/",
        "deadline": "2025-11-30",
        "eligibility_criteria": {"income_max": 200000, "education_level": ["12th", "graduate", "postgraduate"], "states": ["all"]},
        "required_documents": ["Aadhaar Card", "Minority Certificate", "Income Certificate", "Marksheet"],
        "weights_override": None,
    },

    {
        "scheme_id": "BEGUM-HAZRAT-SCHOLARSHIP",
        "name": "Begum Hazrat Mahal National Scholarship (Minority Girls)",
        "category": "education",
        "description": "₹5,000-₹6,000/year for meritorious minority girls studying Class 9-12. Minimum 50% marks.",
        "benefit_amount": 6000, "benefit_type": "Annual Scholarship",
        "application_url": "https://bhmnsmaef.org/",
        "deadline": "2025-10-31",
        "eligibility_criteria": {"gender": ["female"], "income_max": 200000, "states": ["all"]},
        "required_documents": ["Aadhaar Card", "Minority Certificate", "Income Certificate", "Marksheet"],
        "weights_override": None,
    },

    # ─── MATERNITY & CHILD WELFARE ────────────────────────────────────────────

    {
        "scheme_id": "PMMVY",
        "name": "Pradhan Mantri Matru Vandana Yojana (PMMVY)",
        "category": "health",
        "description": "₹5,000 cash incentive to pregnant women for first living child. ₹6,000 for institutional delivery (Janani Suraksha).",
        "benefit_amount": 5000, "benefit_type": "Maternity Cash Incentive",
        "application_url": "https://pmmvy.wcd.gov.in/",
        "deadline": None,
        "eligibility_criteria": {"gender": ["female"], "age_min": 19, "states": ["all"]},
        "required_documents": ["Aadhaar Card", "MCP Card", "Bank Account", "Pregnancy Certificate"],
        "weights_override": None,
    },

    {
        "scheme_id": "JSY",
        "name": "Janani Suraksha Yojana (JSY)",
        "category": "health",
        "description": "₹700-₹1,400 cash assistance for institutional delivery. Higher in rural areas. For BPL pregnant women.",
        "benefit_amount": 1400, "benefit_type": "Delivery Cash Assistance",
        "application_url": "https://nhm.gov.in/",
        "deadline": None,
        "eligibility_criteria": {"gender": ["female"], "bpl_card": True, "states": ["all"]},
        "required_documents": ["Aadhaar Card", "BPL Card", "Hospital Delivery Certificate"],
        "weights_override": None,
    },

    {
        "scheme_id": "ICDS",
        "name": "Integrated Child Development Services (ICDS)",
        "category": "health",
        "description": "Free supplementary nutrition, immunization, health check-ups, pre-school education for children 0-6 years and pregnant/lactating women.",
        "benefit_amount": 0, "benefit_type": "Free Nutrition + Healthcare",
        "application_url": "https://icds-wcd.nic.in/",
        "deadline": None,
        "eligibility_criteria": {"states": ["all"]},
        "required_documents": ["Aadhaar Card", "Birth Certificate"],
        "weights_override": None,
    },

    {
        "scheme_id": "POSHAN-ABHIYAAN",
        "name": "PM POSHAN / Mid Day Meal Scheme",
        "category": "education",
        "description": "Free cooked meals to children in government and government-aided schools (Class 1-8). Targets malnutrition.",
        "benefit_amount": 0, "benefit_type": "Free School Meals",
        "application_url": "https://pmposhan.education.gov.in/",
        "deadline": None,
        "eligibility_criteria": {"age_min": 6, "age_max": 14, "states": ["all"]},
        "required_documents": ["School Enrollment Proof"],
        "weights_override": None,
    },

    # ─── TRIBAL WELFARE ───────────────────────────────────────────────────────

    {
        "scheme_id": "ST-SCHOLARSHIP-PRE",
        "name": "Pre-Matric Scholarship for ST Students",
        "category": "education",
        "description": "₹3,500-₹6,500/year for SC/ST students studying Class 9-10. Covers tuition, books, and maintenance.",
        "benefit_amount": 6500, "benefit_type": "Annual Scholarship",
        "application_url": "https://scholarships.gov.in/",
        "deadline": "2025-11-30",
        "eligibility_criteria": {"category": ["st"], "income_max": 250000, "states": ["all"]},
        "required_documents": ["Aadhaar Card", "Caste Certificate", "Income Certificate", "School Certificate"],
        "weights_override": None,
    },

    {
        "scheme_id": "VAN-BANDHU-KALYAN",
        "name": "Van Bandhu Kalyan Yojana (Tribal Development)",
        "category": "employment",
        "description": "Comprehensive tribal development: livelihood support, education, health, housing, and skill development for tribal areas.",
        "benefit_amount": 50000, "benefit_type": "Livelihood Support",
        "application_url": "https://tribal.nic.in/",
        "deadline": None,
        "eligibility_criteria": {"category": ["st"], "states": ["all"]},
        "required_documents": ["Aadhaar Card", "Caste Certificate", "Income Certificate"],
        "weights_override": None,
    },

    # ─── UNORGANISED WORKERS ──────────────────────────────────────────────────

    {
        "scheme_id": "E-SHRAM",
        "name": "e-Shram Card (Unorganised Workers)",
        "category": "employment",
        "description": "Registration for unorganised workers. Accidental insurance of ₹2 lakh. Access to welfare schemes. Free registration.",
        "benefit_amount": 200000, "benefit_type": "Accident Insurance + Scheme Access",
        "application_url": "https://eshram.gov.in/",
        "deadline": None,
        "eligibility_criteria": {"age_min": 16, "age_max": 59, "states": ["all"]},
        "required_documents": ["Aadhaar Card", "Mobile Number", "Bank Account"],
        "weights_override": None,
    },

    {
        "scheme_id": "PM-SHRAM-YOGI",
        "name": "PM Shram Yogi Maandhan (PM-SYM)",
        "category": "financial_aid",
        "description": "₹3,000/month pension after age 60 for unorganised workers. Monthly contribution ₹55-₹200. Government matches contribution.",
        "benefit_amount": 36000, "benefit_type": "Monthly Pension",
        "application_url": "https://maandhan.in/",
        "deadline": None,
        "eligibility_criteria": {"age_min": 18, "age_max": 40, "income_max": 180000, "states": ["all"]},
        "required_documents": ["Aadhaar Card", "Bank Account", "Mobile Number"],
        "weights_override": None,
    },

    # ─── STATE-SPECIFIC SCHEMES ───────────────────────────────────────────────

    {
        "scheme_id": "KA-BHAGYA-JYOTHI",
        "name": "Bhagya Jyothi / Bhagya Lakshmi (Karnataka)",
        "category": "financial_aid",
        "description": "₹1 lakh on birth of girl child in BPL family. Annual scholarship till Class 10. ₹2 lakh on turning 18.",
        "benefit_amount": 100000, "benefit_type": "Girl Child Bond",
        "application_url": "https://dwcd.karnataka.gov.in/",
        "deadline": None,
        "eligibility_criteria": {"gender": ["female"], "bpl_card": True, "states": ["karnataka"]},
        "required_documents": ["Aadhaar Card", "BPL Card", "Birth Certificate", "Bank Account"],
        "weights_override": None,
    },

    {
        "scheme_id": "KA-RAITA-SIRI",
        "name": "Raita Siri (Karnataka Farmer Income Support)",
        "category": "agriculture",
        "description": "₹10,000/year to all registered farmers in Karnataka regardless of land size.",
        "benefit_amount": 10000, "benefit_type": "Annual Income Support",
        "application_url": "https://raitamitra.karnataka.gov.in/",
        "deadline": None,
        "eligibility_criteria": {"employment_status": ["farmer"], "states": ["karnataka"]},
        "required_documents": ["Aadhaar Card", "Land Records", "Bank Account"],
        "weights_override": None,
    },

    {
        "scheme_id": "KA-VIDYASIRI",
        "name": "Vidyasiri Scholarship (Karnataka)",
        "category": "education",
        "description": "Scholarship for SC/ST/OBC students in Karnataka for higher education. ₹5,000-₹50,000/year depending on course.",
        "benefit_amount": 50000, "benefit_type": "Annual Scholarship",
        "application_url": "https://ssp.postmatric.karnataka.gov.in/",
        "deadline": "2025-11-30",
        "eligibility_criteria": {"category": ["sc", "st", "obc"], "income_max": 250000, "states": ["karnataka"]},
        "required_documents": ["Aadhaar Card", "Caste Certificate", "Income Certificate", "College Admission"],
        "weights_override": None,
    },

    {
        "scheme_id": "AP-YSR-RYTHU-BHAROSA",
        "name": "YSR Rythu Bharosa (Andhra Pradesh)",
        "category": "agriculture",
        "description": "₹13,500/year per farmer family (₹7,500 state + ₹6,000 PM-KISAN). For all farmers in AP.",
        "benefit_amount": 13500, "benefit_type": "Annual Income Support",
        "application_url": "https://ysrrythubharosa.ap.gov.in/",
        "deadline": None,
        "eligibility_criteria": {"employment_status": ["farmer"], "states": ["andhra_pradesh"]},
        "required_documents": ["Aadhaar Card", "Land Records", "Bank Account"],
        "weights_override": None,
    },

    {
        "scheme_id": "AP-AMMA-VODI",
        "name": "Amma Vodi (Andhra Pradesh)",
        "category": "education",
        "description": "₹15,000/year to mothers of school-going children to prevent dropouts. For families below poverty line.",
        "benefit_amount": 15000, "benefit_type": "Annual Cash Transfer",
        "application_url": "https://jaganannaammavodi.ap.gov.in/",
        "deadline": None,
        "eligibility_criteria": {"gender": ["female"], "income_max": 250000, "states": ["andhra_pradesh"]},
        "required_documents": ["Aadhaar Card", "Child's School Enrollment", "White Ration Card"],
        "weights_override": None,
    },

    {
        "scheme_id": "TS-KALYANA-LAKSHMI",
        "name": "Kalyana Lakshmi / Shaadi Mubarak (Telangana)",
        "category": "financial_aid",
        "description": "₹1,00,116 marriage assistance for girls from SC/ST/BC/Minority families. Age 18+.",
        "benefit_amount": 100116, "benefit_type": "Marriage Assistance",
        "application_url": "https://telanganaepass.cgg.gov.in/",
        "deadline": None,
        "eligibility_criteria": {"gender": ["female"], "age_min": 18, "income_max": 200000, "states": ["telangana"]},
        "required_documents": ["Aadhaar Card", "Income Certificate", "Caste Certificate", "Marriage Invitation"],
        "weights_override": None,
    },

    {
        "scheme_id": "TS-RYTHU-BANDHU",
        "name": "Rythu Bandhu (Telangana Farmer Investment Support)",
        "category": "agriculture",
        "description": "₹10,000/acre/year to all land-owning farmers. Paid in two installments of ₹5,000 each season.",
        "benefit_amount": 10000, "benefit_type": "Per-Acre Support",
        "application_url": "https://rythubandhu.telangana.gov.in/",
        "deadline": None,
        "eligibility_criteria": {"employment_status": ["farmer"], "owns_land": True, "states": ["telangana"]},
        "required_documents": ["Aadhaar Card", "Pattadar Passbook", "Bank Account"],
        "weights_override": None,
    },

    {
        "scheme_id": "TN-AMMA-TWO-WHEELER",
        "name": "Amma Two Wheeler Scheme (Tamil Nadu)",
        "category": "employment",
        "description": "50% subsidy (up to ₹25,000) on purchase of two-wheeler for working women in Tamil Nadu.",
        "benefit_amount": 25000, "benefit_type": "Vehicle Subsidy",
        "application_url": "https://tnlabour.in/",
        "deadline": None,
        "eligibility_criteria": {"gender": ["female"], "age_min": 18, "income_max": 250000, "states": ["tamil_nadu"]},
        "required_documents": ["Aadhaar Card", "Income Certificate", "Driving Licence", "Employment Proof"],
        "weights_override": None,
    },

    {
        "scheme_id": "TN-FREE-LAPTOP",
        "name": "Free Laptop Scheme (Tamil Nadu)",
        "category": "education",
        "description": "Free laptops for students joining government arts/science colleges and polytechnics in Tamil Nadu.",
        "benefit_amount": 20000, "benefit_type": "Free Laptop",
        "application_url": "https://www.tn.gov.in/",
        "deadline": None,
        "eligibility_criteria": {"education_level": ["12th", "graduate"], "states": ["tamil_nadu"]},
        "required_documents": ["Aadhaar Card", "College Admission Letter", "Class 12 Marksheet"],
        "weights_override": None,
    },

    {
        "scheme_id": "OASIS-WB",
        "name": "OASIS Scholarship (West Bengal)",
        "category": "education",
        "description": "Scholarship for SC/ST/OBC minority students in West Bengal. Pre-matric: ₹1,000-₹5,000. Post-matric: up to ₹25,000.",
        "benefit_amount": 25000, "benefit_type": "Annual Scholarship",
        "application_url": "https://oasis.gov.in/",
        "deadline": "2025-11-30",
        "eligibility_criteria": {"category": ["sc", "st", "obc"], "income_max": 250000, "states": ["west_bengal"]},
        "required_documents": ["Aadhaar Card", "Caste Certificate", "Income Certificate", "Marksheet"],
        "weights_override": None,
    },

    {
        "scheme_id": "WB-KANYASHREE",
        "name": "Kanyashree Prakalpa (West Bengal)",
        "category": "education",
        "description": "₹750/year for girls 13-18 in school + ₹25,000 one-time grant at age 18 if unmarried and in education.",
        "benefit_amount": 25000, "benefit_type": "Girl Education Grant",
        "application_url": "https://wbkanyashree.gov.in/",
        "deadline": None,
        "eligibility_criteria": {"gender": ["female"], "age_min": 13, "age_max": 18, "income_max": 120000, "states": ["west_bengal"]},
        "required_documents": ["Aadhaar Card", "School Certificate", "Income Certificate"],
        "weights_override": None,
    },

    {
        "scheme_id": "KERALA-WELFARE-PENSION",
        "name": "Kerala Social Security Pension",
        "category": "financial_aid",
        "description": "₹1,600/month pension for elderly, widows, disabled, and unmarried women above 50 in Kerala. BPL families.",
        "benefit_amount": 19200, "benefit_type": "Monthly Pension",
        "application_url": "https://welfarepension.lsgkerala.gov.in/",
        "deadline": None,
        "eligibility_criteria": {"age_min": 60, "bpl_card": True, "states": ["kerala"]},
        "required_documents": ["Aadhaar Card", "Age Proof", "BPL Certificate", "Bank Account"],
        "weights_override": None,
    },

    {
        "scheme_id": "UP-KANYA-SUMANGALA",
        "name": "Mukhyamantri Kanya Sumangala Yojana (UP)",
        "category": "financial_aid",
        "description": "₹15,000 in 6 installments from birth to graduation for girls in UP. Family income < ₹3 lakh.",
        "benefit_amount": 15000, "benefit_type": "Girl Child Benefit",
        "application_url": "https://mksy.up.gov.in/",
        "deadline": None,
        "eligibility_criteria": {"gender": ["female"], "income_max": 300000, "states": ["uttar_pradesh"]},
        "required_documents": ["Aadhaar Card", "Birth Certificate", "Bank Account", "Address Proof"],
        "weights_override": None,
    },

    {
        "scheme_id": "MH-MAJHI-KANYA-BHAGYASHREE",
        "name": "Majhi Kanya Bhagyashree (Maharashtra)",
        "category": "financial_aid",
        "description": "₹50,000 insurance + ₹1 lakh bond on birth of girl. Family income < ₹7.5 lakh. One girl or two girls per family.",
        "benefit_amount": 100000, "benefit_type": "Girl Child Bond",
        "application_url": "https://womenchild.maharashtra.gov.in/",
        "deadline": None,
        "eligibility_criteria": {"gender": ["female"], "income_max": 750000, "states": ["maharashtra"]},
        "required_documents": ["Aadhaar Card", "Birth Certificate", "Income Certificate", "Bank Account"],
        "weights_override": None,
    },

    {
        "scheme_id": "RAJASTHAN-PALANHAR",
        "name": "Palanhar Yojana (Rajasthan)",
        "category": "financial_aid",
        "description": "₹500-₹1,000/month for orphans, destitute children, and children of special categories raised by relatives.",
        "benefit_amount": 12000, "benefit_type": "Monthly Child Support",
        "application_url": "https://sje.rajasthan.gov.in/",
        "deadline": None,
        "eligibility_criteria": {"income_max": 120000, "states": ["rajasthan"]},
        "required_documents": ["Aadhaar Card", "Orphan Certificate", "School Certificate", "Guardian Aadhaar"],
        "weights_override": None,
    },

    {
        "scheme_id": "GJ-MANAV-GARIMA",
        "name": "Manav Garima Yojana (Gujarat)",
        "category": "employment",
        "description": "Free tool kits worth ₹4,000-₹15,000 for SC families to start self-employment. Covers 28 trades.",
        "benefit_amount": 15000, "benefit_type": "Free Tool Kit",
        "application_url": "https://sje.gujarat.gov.in/",
        "deadline": None,
        "eligibility_criteria": {"category": ["sc"], "income_max": 120000, "states": ["gujarat"]},
        "required_documents": ["Aadhaar Card", "Caste Certificate", "Income Certificate", "Bank Account"],
        "weights_override": None,
    },

    # ─── URBAN DEVELOPMENT ────────────────────────────────────────────────────

    {
        "scheme_id": "AMRUT",
        "name": "AMRUT 2.0 (Urban Water/Sewerage)",
        "category": "housing",
        "description": "Provides functional water tap connections and sewerage in 500 cities. Benefits households in mission cities.",
        "benefit_amount": 0, "benefit_type": "Infrastructure Development",
        "application_url": "https://amrut.gov.in/",
        "deadline": None,
        "eligibility_criteria": {"area_type": ["urban"], "states": ["all"]},
        "required_documents": ["Aadhaar Card", "Address Proof"],
        "weights_override": None,
    },

    {
        "scheme_id": "SMART-CITY",
        "name": "Smart Cities Mission",
        "category": "housing",
        "description": "Urban development for 100 selected cities. Smart solutions for water, electricity, sanitation, transport, and governance.",
        "benefit_amount": 0, "benefit_type": "Urban Infrastructure",
        "application_url": "https://smartcities.gov.in/",
        "deadline": None,
        "eligibility_criteria": {"area_type": ["urban"], "states": ["all"]},
        "required_documents": [],
        "weights_override": None,
    },

    # ─── ADDITIONAL AGRICULTURE ───────────────────────────────────────────────

    {
        "scheme_id": "SOIL-HEALTH-CARD",
        "name": "Soil Health Card Scheme",
        "category": "agriculture",
        "description": "Free soil testing and health card with crop-wise fertilizer recommendations. Helps improve crop productivity.",
        "benefit_amount": 0, "benefit_type": "Free Soil Testing",
        "application_url": "https://soilhealth.dac.gov.in/",
        "deadline": None,
        "eligibility_criteria": {"employment_status": ["farmer"], "states": ["all"]},
        "required_documents": ["Aadhaar Card", "Land Records"],
        "weights_override": None,
    },

    {
        "scheme_id": "NEEM-COATED-UREA",
        "name": "Neem Coated Urea Scheme",
        "category": "agriculture",
        "description": "Subsidised neem-coated urea at ₹242/bag (45kg). Reduces fertiliser cost and improves soil health.",
        "benefit_amount": 500, "benefit_type": "Fertiliser Subsidy",
        "application_url": "https://www.india.gov.in/",
        "deadline": None,
        "eligibility_criteria": {"employment_status": ["farmer"], "states": ["all"]},
        "required_documents": ["Aadhaar Card"],
        "weights_override": None,
    },

    {
        "scheme_id": "PM-KUSUM",
        "name": "PM KUSUM (Solar Pump Scheme)",
        "category": "agriculture",
        "description": "90% subsidy on solar water pumps for farmers. 10,000 solar pumps from 2 HP to 10 HP. Also feed-in income from solar.",
        "benefit_amount": 200000, "benefit_type": "Solar Pump Subsidy",
        "application_url": "https://pmkusum.mnre.gov.in/",
        "deadline": None,
        "eligibility_criteria": {"employment_status": ["farmer"], "owns_land": True, "states": ["all"]},
        "required_documents": ["Aadhaar Card", "Land Records", "Bank Account", "Electricity Bill"],
        "weights_override": None,
    },

    {
        "scheme_id": "MICRO-IRRIGATION",
        "name": "PM Krishi Sinchayee Yojana — Micro Irrigation",
        "category": "agriculture",
        "description": "55% subsidy on drip irrigation and 35% on sprinkler systems. Saves water, increases yield.",
        "benefit_amount": 50000, "benefit_type": "Irrigation Subsidy",
        "application_url": "https://pmksy.gov.in/",
        "deadline": None,
        "eligibility_criteria": {"employment_status": ["farmer"], "states": ["all"]},
        "required_documents": ["Aadhaar Card", "Land Records", "Bank Account"],
        "weights_override": None,
    },

    {
        "scheme_id": "ANIMAL-HUSBANDRY",
        "name": "Animal Husbandry Infrastructure Development Fund",
        "category": "agriculture",
        "description": "3% interest subvention on loans for dairy processing, meat processing, animal feed plants. Loan up to ₹30 crore.",
        "benefit_amount": 500000, "benefit_type": "Interest Subvention",
        "application_url": "https://dahd.nic.in/ahidf",
        "deadline": None,
        "eligibility_criteria": {"employment_status": ["farmer", "self_employed"], "states": ["all"]},
        "required_documents": ["PAN Card", "Aadhaar Card", "Project Report", "Bank Account"],
        "weights_override": None,
    },

    # ─── DIGITAL / IT EMPLOYMENT ──────────────────────────────────────────────

    {
        "scheme_id": "NASSCOM-FUTURESKILLS",
        "name": "NASSCOM FutureSkills PRIME",
        "category": "employment",
        "description": "Free training in AI, Blockchain, Cloud, Cybersecurity, IoT. Industry-recognized certificates. For IT professionals and students.",
        "benefit_amount": 0, "benefit_type": "Free IT Training",
        "application_url": "https://futureskillsprime.in/",
        "deadline": None,
        "eligibility_criteria": {"age_min": 18, "age_max": 45, "states": ["all"]},
        "required_documents": ["Aadhaar Card", "Email ID"],
        "weights_override": None,
    },

    {
        "scheme_id": "NAPS",
        "name": "National Apprenticeship Promotion Scheme (NAPS)",
        "category": "employment",
        "description": "Government shares 25% of stipend (up to ₹1,500/month) for apprentices in designated trades. On-job training with stipend.",
        "benefit_amount": 18000, "benefit_type": "Apprenticeship Stipend",
        "application_url": "https://www.apprenticeshipindia.gov.in/",
        "deadline": None,
        "eligibility_criteria": {"age_min": 14, "age_max": 40, "states": ["all"]},
        "required_documents": ["Aadhaar Card", "Education Certificate", "Bank Account"],
        "weights_override": None,
    },

    # ─── VETERAN / EX-SERVICEMEN ──────────────────────────────────────────────

    {
        "scheme_id": "ECHS",
        "name": "Ex-Servicemen Contributory Health Scheme (ECHS)",
        "category": "health",
        "description": "Cashless medical treatment for ex-servicemen and dependents at 427 ECHS polyclinics and empanelled hospitals.",
        "benefit_amount": 0, "benefit_type": "Free Medical Treatment",
        "application_url": "https://echs.gov.in/",
        "deadline": None,
        "eligibility_criteria": {"states": ["all"]},
        "required_documents": ["ECHS Card", "Discharge Book", "Aadhaar Card"],
        "weights_override": None,
    },

    {
        "scheme_id": "CSD-CANTEEN",
        "name": "CSD Canteen (Defence Personnel)",
        "category": "financial_aid",
        "description": "Subsidised goods (food, electronics, vehicles) for serving/retired defence personnel through Canteen Stores Department.",
        "benefit_amount": 0, "benefit_type": "Subsidised Goods",
        "application_url": "https://csdindia.gov.in/",
        "deadline": None,
        "eligibility_criteria": {"states": ["all"]},
        "required_documents": ["CSD Card", "Defence ID"],
        "weights_override": None,
    },

    # ─── TRANSPORT & INFRASTRUCTURE ───────────────────────────────────────────

    {
        "scheme_id": "PM-GRAM-SADAK",
        "name": "Pradhan Mantri Gram Sadak Yojana (PMGSY)",
        "category": "housing",
        "description": "All-weather road connectivity to unconnected rural habitations with 500+ population (250+ in hilly areas).",
        "benefit_amount": 0, "benefit_type": "Road Infrastructure",
        "application_url": "https://pmgsy.nic.in/",
        "deadline": None,
        "eligibility_criteria": {"area_type": ["rural"], "states": ["all"]},
        "required_documents": [],
        "weights_override": None,
    },

    {
        "scheme_id": "UDAN",
        "name": "UDAN (Regional Connectivity Scheme)",
        "category": "financial_aid",
        "description": "Subsidised air fares starting ₹2,500 for 1-hour flights connecting tier-2/3 cities. Make flying affordable for all.",
        "benefit_amount": 2500, "benefit_type": "Subsidised Air Fare",
        "application_url": "https://www.udanrcs.gov.in/",
        "deadline": None,
        "eligibility_criteria": {"states": ["all"]},
        "required_documents": ["Aadhaar Card"],
        "weights_override": None,
    },

    # ─── CLEAN ENERGY ─────────────────────────────────────────────────────────

    {
        "scheme_id": "ROOFTOP-SOLAR",
        "name": "PM Surya Ghar — Rooftop Solar Scheme",
        "category": "housing",
        "description": "Subsidy of ₹30,000-₹78,000 for 1-3 kW rooftop solar panels. ₹300 free electricity per month. 1 crore homes targeted.",
        "benefit_amount": 78000, "benefit_type": "Solar Panel Subsidy",
        "application_url": "https://pmsuryaghar.gov.in/",
        "deadline": None,
        "eligibility_criteria": {"states": ["all"]},
        "required_documents": ["Aadhaar Card", "Electricity Bill", "Bank Account", "Photo of Roof"],
        "weights_override": None,
    },

    {
        "scheme_id": "GOBAR-DHAN",
        "name": "GOBAR-DHAN (Bio-Gas from Cattle Dung)",
        "category": "agriculture",
        "description": "Support for setting up biogas plants from cattle dung. ₹1-₹4 lakh subsidy. Converts waste to energy and organic manure.",
        "benefit_amount": 400000, "benefit_type": "Bio-Gas Plant Subsidy",
        "application_url": "https://sbm.gov.in/gbdw20/",
        "deadline": None,
        "eligibility_criteria": {"employment_status": ["farmer"], "area_type": ["rural"], "states": ["all"]},
        "required_documents": ["Aadhaar Card", "Land Records", "Bank Account"],
        "weights_override": None,
    },

    # ─── MISCELLANEOUS IMPORTANT ──────────────────────────────────────────────

    {
        "scheme_id": "PM-JAI",
        "name": "PM Janaushadi (Affordable Medicines)",
        "category": "health",
        "description": "Generic medicines at 50-90% lower cost through 9,000+ Janaushadi Kendras. Over 1,800 medicines and 285 surgical items.",
        "benefit_amount": 0, "benefit_type": "Affordable Medicine Access",
        "application_url": "https://janaushadhi.gov.in/",
        "deadline": None,
        "eligibility_criteria": {"states": ["all"]},
        "required_documents": ["Doctor's Prescription"],
        "weights_override": None,
    },

    {
        "scheme_id": "ONE-NATION-ONE-RATION",
        "name": "One Nation One Ration Card (ONORC)",
        "category": "financial_aid",
        "description": "Use ration card at any Fair Price Shop in India. Portability across states for migrant workers.",
        "benefit_amount": 0, "benefit_type": "Portable Ration Card",
        "application_url": "https://nfsa.gov.in/",
        "deadline": None,
        "eligibility_criteria": {"states": ["all"]},
        "required_documents": ["Aadhaar Card", "Ration Card"],
        "weights_override": None,
    },

    {
        "scheme_id": "PM-JANDHAN",
        "name": "Pradhan Mantri Jan Dhan Yojana (PMJDY)",
        "category": "financial_aid",
        "description": "Zero-balance bank account with RuPay debit card, ₹1 lakh accident insurance, ₹30,000 life cover, ₹10,000 overdraft.",
        "benefit_amount": 130000, "benefit_type": "Free Bank Account + Insurance",
        "application_url": "https://pmjdy.gov.in/",
        "deadline": None,
        "eligibility_criteria": {"age_min": 10, "states": ["all"]},
        "required_documents": ["Aadhaar Card", "Passport Photo"],
        "weights_override": None,
    },

    {
        "scheme_id": "DIGILOCKER",
        "name": "DigiLocker (Digital Document Storage)",
        "category": "financial_aid",
        "description": "Free cloud storage for government-issued documents. Access Aadhaar, PAN, driving licence, marksheets digitally.",
        "benefit_amount": 0, "benefit_type": "Free Digital Document Storage",
        "application_url": "https://www.digilocker.gov.in/",
        "deadline": None,
        "eligibility_criteria": {"states": ["all"]},
        "required_documents": ["Aadhaar Card", "Mobile Number"],
        "weights_override": None,
    },
]


def get_all_schemes() -> List[Dict[str, Any]]:
    """Return all schemes in the database."""
    return SCHEMES


def get_schemes_by_category(category: str) -> List[Dict[str, Any]]:
    """Return schemes filtered by category."""
    return [s for s in SCHEMES if s["category"] == category]


def get_scheme_by_id(scheme_id: str) -> Dict[str, Any]:
    """Return a single scheme by ID, or raise KeyError if not found."""
    for s in SCHEMES:
        if s["scheme_id"] == scheme_id:
            return s
    raise KeyError(f"Scheme '{scheme_id}' not found.")


def get_unique_categories() -> List[str]:
    """Return unique scheme categories."""
    return sorted(set(s["category"] for s in SCHEMES))

