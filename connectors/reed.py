import requests
import os
import pandas as pd
import time
import logging
from dotenv import load_dotenv
from datetime import datetime, date

# ── Setup ──────────────────────────────────────────────────────────────────
load_dotenv()
REED_API_KEY = os.getenv("REED_API_KEY")
BASE_URL = "https://www.reed.co.uk/api/1.0"

logging.basicConfig(
    filename="reed.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ── Digital transformation job titles to monitor ──────────────────────────
# These indicate value creation activity at PE portfolio companies
TRANSFORMATION_KEYWORDS = [
    "digital transformation",
    "chief digital officer",
    "chief technology officer",
    "head of technology",
    "data engineer",
    "cloud architect",
    "ERP implementation",
    "AI strategy",
    "machine learning engineer",
    "data analytics manager",
    "technology director",
    "digital strategy",
    "platform engineer",
    "software engineer financial services",
    "technology transformation",
]

# ── Core API function ──────────────────────────────────────────────────────
def search_reed_jobs(keyword, location="UK", results_to_take=20):
    """
    Search Reed job postings by keyword.
    Returns list of job postings.
    """
    url = f"{BASE_URL}/search"
    params = {
        "keywords": keyword,
        "locationName": location,
        "resultsToTake": results_to_take,
        "fullTime": True,
    }

    try:
        time.sleep(0.3)
        response = requests.get(
            url,
            params=params,
            auth=(REED_API_KEY, ""),
            timeout=10
        )

        if response.status_code == 200:
            return response.json()
        else:
            logging.error(f"Reed API error {response.status_code} for: {keyword}")
            return None

    except Exception as e:
        logging.error(f"Reed request failed: {e}")
        return None

def get_job_details(job_id):
    """Get full details for a specific job posting"""
    url = f"{BASE_URL}/jobs/{job_id}"
    try:
        time.sleep(0.3)
        response = requests.get(url, auth=(REED_API_KEY, ""), timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        logging.error(f"Reed job detail failed: {e}")
        return None

def is_financial_services_employer(job):
    """
    Check if a job is from a financial services company.
    Looks at employer name and job description for FS indicators.
    """
    employer = job.get("employerName", "").lower()
    description = job.get("jobDescription", "").lower()
    text = employer + " " + description

    fs_indicators = [
        "financial", "fintech", "insurance", "banking", "payments",
        "asset management", "wealth", "investment", "fund", "capital",
        "financial services", "financial technology", "trading",
        "credit", "lending", "mortgage"
    ]

    return any(indicator in text for indicator in fs_indicators)

def search_company_jobs(company_name, keyword="digital transformation"):
    """
    Search for specific job postings at a named company.
    Used after a Companies House PE detection to check
    if value creation activity has started.
    """
    url = f"{BASE_URL}/search"
    params = {
        "keywords": keyword,
        "employerId": company_name,
        "resultsToTake": 10,
    }

    try:
        time.sleep(0.3)
        response = requests.get(
            url,
            params=params,
            auth=(REED_API_KEY, ""),
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        logging.error(f"Reed company search failed: {e}")
        return None

def run_reed_pipeline(keywords=None):
    """
    Main Reed pipeline function.
    Searches for digital transformation jobs in UK financial services.
    Saves results to CSV for value creation signal analysis.
    """
    target_keywords = keywords or TRANSFORMATION_KEYWORDS
    all_jobs = []

    print("=" * 60)
    print("Reed Job Postings PE Value Creation Signal Collector")
    print(f"Searching {len(target_keywords)} transformation keywords")
    print(f"Location: UK financial services")
    print("=" * 60)

    for keyword in target_keywords:
        print(f"\nSearching: '{keyword}'")
        results = search_reed_jobs(keyword)

        if not results:
            print(f"  No results or error — skipping")
            continue

        jobs = results.get("results", [])
        total = results.get("totalResults", 0)

        # Filter for financial services employers
        fs_jobs = [j for j in jobs if is_financial_services_employer(j)]
        print(f"  Found {total} total jobs. {len(fs_jobs)} in financial services.")

        for job in fs_jobs:
            all_jobs.append({
                "job_id": job.get("jobId"),
                "job_title": job.get("jobTitle"),
                "employer": job.get("employerName"),
                "location": job.get("locationName"),
                "salary_min": job.get("minimumSalary"),
                "salary_max": job.get("maximumSalary"),
                "date_posted": job.get("date", "")[:10],
                "job_url": job.get("jobUrl"),
                "description_snippet": job.get("jobDescription", "")[:300],
                "keyword_searched": keyword,
                "collected_at": datetime.now().isoformat()
            })

        logging.info(f"Reed search '{keyword}': {total} total, {len(fs_jobs)} FS relevant")

    if all_jobs:
        df = pd.DataFrame(all_jobs)
        df = df.drop_duplicates(subset="job_id")
        df = df.sort_values("date_posted", ascending=False)
        filename = f"data/reed_jobs_{date.today().isoformat()}.csv"
        df.to_csv(filename, index=False)
        print(f"\n{'=' * 60}")
        print(f"COLLECTION COMPLETE")
        print(f"Total unique FS transformation jobs: {len(df)}")
        print(f"Saved to: {filename}")
        print(f"\nSample jobs:")
        print(df[["job_title", "employer", "location", "date_posted"]].head(10).to_string(index=False))
    else:
        print("\nNo financial services transformation jobs found")

    return all_jobs

def check_portfolio_company_hiring(company_name):
    """
    Value creation signal check.
    After a PE acquisition is detected, check if the portfolio
    company is hiring digital/transformation roles.
    High hiring = value creation phase has begun.
    """
    print(f"\nChecking hiring activity: {company_name}")

    transformation_roles = [
        "digital transformation",
        "technology",
        "data",
        "cloud",
        "AI",
        "software"
    ]

    found_jobs = []
    for role in transformation_roles:
        results = search_reed_jobs(f"{company_name} {role}", results_to_take=5)
        if results:
            jobs = results.get("results", [])
            found_jobs.extend(jobs)
        time.sleep(0.2)

    if found_jobs:
        print(f"  ✓ Active hiring detected — {len(found_jobs)} transformation roles")
        for job in found_jobs[:3]:
            print(f"    → {job.get('jobTitle')} at {job.get('employerName')}")
        return True
    else:
        print(f"  - No transformation hiring detected")
        return False

if __name__ == "__main__":
    run_reed_pipeline()

    print("\n" + "=" * 60)
    print("VALUE CREATION SIGNAL CHECK")
    print("Checking portfolio companies for transformation hiring")
    print("=" * 60)

    # Check our Companies House detections for hiring activity
    portfolio_companies = [
        "Cantor Finance",
        "SMH Woodland",
        "Innovation Investment Capital"
    ]

    for company in portfolio_companies:
        check_portfolio_company_hiring(company)