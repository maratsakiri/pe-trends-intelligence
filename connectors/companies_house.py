import requests
import os
import pandas as pd
import time
import logging
from dotenv import load_dotenv
from datetime import datetime, date

# ── Setup ──────────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("COMPANIES_HOUSE_API_KEY")
BASE_URL = "https://api.company-information.service.gov.uk"

# Set up logging so errors are recorded not just printed
logging.basicConfig(
    filename="companies_house.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ── UK Financial Services SIC codes ───────────────────────────────────────
FINANCIAL_SERVICES_SIC = {
    "64110": "Central banking",
    "64191": "Banks",
    "64192": "Building societies",
    "64201": "Activities of financial holding companies",
    "64205": "Activities of financial services holding companies",
    "64301": "Activities of investment trusts",
    "64302": "Activities of unit trusts",
    "64303": "Activities of venture and development capital companies",
    "64304": "Activities of open-ended investment companies",
    "64910": "Financial leasing",
    "64921": "Credit granting by non-deposit taking finance houses",
    "64922": "Activities of mortgage finance companies",
    "64929": "Other credit granting",
    "64991": "Security dealing on own account",
    "64992": "Factoring",
    "64999": "Other financial service activities",
    "66110": "Administration of financial markets",
    "66120": "Security and commodity contracts dealing",
    "66190": "Other activities auxiliary to financial services",
}

# ── PE firm indicators ─────────────────────────────────────────────────────
# These are keywords that suggest an owner is a PE firm or investment vehicle
# Individual names (Mr, Mrs, Dr) are excluded
PE_KEYWORDS = [
    "capital", "partners", "equity", "investments", "holdings",
    "fund", "ventures", "acquisition", "buyout", "growth",
    "private equity", "asset management", "asset manager",
    "infrastructure", "portfolio", "llp", "l.p.", "gp limited",
    "general partner", "management limited"
]

INDIVIDUAL_INDICATORS = ["mr ", "mrs ", "ms ", "dr ", "miss "]

# ── Core API functions ─────────────────────────────────────────────────────
def make_request(url, params=None):
    """
    Make a rate-limited API request with error handling.
    Returns the JSON response or None if it fails.
    """
    try:
        time.sleep(0.2)  # 200ms delay between requests — stays well within limits
        response = requests.get(url, params=params, auth=(API_KEY, ""), timeout=10)

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            print("  Rate limit hit — waiting 60 seconds...")
            logging.warning("Rate limit hit — waiting 60 seconds")
            time.sleep(60)
            return make_request(url, params)  # retry
        else:
            logging.error(f"API error {response.status_code} for {url}")
            return None

    except requests.exceptions.Timeout:
        logging.error(f"Timeout for {url}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error for {url}: {e}")
        return None

def get_company_profile(company_number):
    """Get full profile for a specific company"""
    url = f"{BASE_URL}/company/{company_number}"
    return make_request(url)

def get_persons_with_significant_control(company_number):
    """Get PSC data — shows who owns or controls the company"""
    url = f"{BASE_URL}/company/{company_number}/persons-with-significant-control"
    return make_request(url)

def get_filing_history(company_number, category="confirmation-statement"):
    """Get recent filing history for a company"""
    url = f"{BASE_URL}/company/{company_number}/filing-history"
    params = {"category": category, "items_per_page": 10}
    return make_request(url, params)

def search_by_sic(sic_code, size=100):
    """Search for active companies by SIC code using advanced search"""
    url = f"{BASE_URL}/advanced-search/companies"
    params = {
        "sic_codes": sic_code,
        "company_status": "active",
        "size": size
    }
    return make_request(url, params)

# ── PE detection logic ─────────────────────────────────────────────────────
def is_pe_owner(psc):
    """
    Determine if a PSC entry looks like a PE firm owner.
    Returns True if PE indicators found and it's not an individual.
    """
    name = psc.get("name", "").lower()
    kind = psc.get("kind", "").lower()

    # Skip individual persons
    if any(indicator in name for indicator in INDIVIDUAL_INDICATORS):
        return False

    # Skip if kind is individual person
    if "individual" in kind:
        return False

    # Check for PE keywords
    return any(keyword in name for keyword in PE_KEYWORDS)

def is_recent_acquisition(notified_on, cutoff_date="2021-01-01"):
    """Check if ownership change is after our cutoff date"""
    if not notified_on:
        return False
    return notified_on >= cutoff_date

# Generic corporate words that should NOT count as a shared "root" — two
# unrelated firms both containing "holdings" or "capital" are not intra-group.
_GENERIC_ROOT_WORDS = {
    "holdings", "holding", "group", "ltd", "limited", "llp", "plc", "uk",
    "capital", "ventures", "venture", "investments", "investment", "finance",
    "financial", "partners", "fund", "managers", "management", "company",
    "co", "the", "services", "service", "and", "of",
}

def is_intra_group(company_name, owner_name):
    """
    Heuristic: flag an owner as an intra-group holding structure (NOT a PE
    buyout) when the company and its PSC owner share a distinctive root word.

    e.g. 'Cantor Finance Limited' owned by 'Cantor Holdings Limited' -> True
         (they share the distinctive word 'cantor')

    Generic corporate words (holdings, capital, group, ...) are ignored so we
    don't falsely link two unrelated firms that both contain 'Holdings'.
    Returns (is_intra_group: bool, shared_word: str|None) for transparent
    logging — we never drop a row silently.
    """
    def distinctive_words(text):
        words = "".join(c if c.isalnum() else " " for c in text.lower()).split()
        return {w for w in words if w not in _GENERIC_ROOT_WORDS and len(w) > 2}

    shared = distinctive_words(company_name) & distinctive_words(owner_name)
    if shared:
        return True, sorted(shared)[0]
    return False, None

def check_for_pe_ownership(company_number, company_name=""):
    """
    Check if a company has PE firm ownership since January 2021.
    Excludes intra-group holding structures (owner shares a distinctive root
    word with the company) and logs each exclusion for transparency.
    Returns list of PE owner details if found.
    """
    psc_data = get_persons_with_significant_control(company_number)
    if not psc_data:
        return []

    pe_indicators = []
    items = psc_data.get("items", [])

    for psc in items:
        notified_on = psc.get("notified_on", "")
        owner_name = psc.get("name", "")

        if is_pe_owner(psc) and is_recent_acquisition(notified_on):
            intra, shared_word = is_intra_group(company_name, owner_name)
            if intra:
                # Likely a parent/subsidiary structure, not a PE buyout.
                logging.info(
                    f"Excluded intra-group: {company_name} <- {owner_name} "
                    f"(shared root '{shared_word}')"
                )
                print(f"    (excluded intra-group: {owner_name} — shares "
                      f"'{shared_word}')")
                continue

            pe_indicators.append({
                "owner_name": owner_name,
                "ownership_type": psc.get("kind"),
                "notified_on": notified_on,
                "nature_of_control": ", ".join(psc.get("natures_of_control", []))
            })

    return pe_indicators

# ── Main pipeline function ─────────────────────────────────────────────────
def run_companies_house_pipeline(sic_codes=None, companies_per_sic=50):
    """
    Main pipeline function.
    Searches UK financial services companies by SIC code,
    detects PE ownership changes since Jan 2021,
    and saves results to CSV.
    """
    target_sics = sic_codes or list(FINANCIAL_SERVICES_SIC.keys())
    all_results = []
    total_checked = 0
    total_pe_found = 0

    print("=" * 60)
    print("Companies House PE Acquisition Detector")
    print(f"Scanning {len(target_sics)} SIC codes")
    print(f"Checking up to {companies_per_sic} companies per SIC")
    print(f"Date filter: acquisitions from 2021-01-01 onwards")
    print("=" * 60)

    for sic_code in target_sics:
        sic_name = FINANCIAL_SERVICES_SIC.get(sic_code, sic_code)
        print(f"\n[{sic_code}] {sic_name}")

        results = search_by_sic(sic_code, size=companies_per_sic)
        if not results:
            print(f"  No results or error — skipping")
            continue

        companies = results.get("items", [])
        hits = results.get("hits", 0)
        print(f"  Found {hits} total companies. Checking {len(companies)}...")

        for company in companies:
            company_number = company.get("company_number")
            company_name = company.get("company_name")
            total_checked += 1

            # Get full company profile for richer data
            profile = get_company_profile(company_number)
            incorporation_date = profile.get("date_of_creation", "") if profile else ""
            registered_address = ""
            if profile:
                addr = profile.get("registered_office_address", {})
                registered_address = ", ".join(filter(None, [
                    addr.get("address_line_1", ""),
                    addr.get("locality", ""),
                    addr.get("postal_code", "")
                ]))

            # Check for PE ownership (excludes intra-group structures)
            pe_owners = check_for_pe_ownership(company_number, company_name)

            if pe_owners:
                total_pe_found += 1
                print(f"\n  ✓ PE ACQUISITION DETECTED: {company_name}")
                for owner in pe_owners:
                    print(f"    → Owner: {owner['owner_name']}")
                    print(f"    → Since: {owner['notified_on']}")

                for owner in pe_owners:
                    all_results.append({
                        "company_number": company_number,
                        "company_name": company_name,
                        "sic_code": sic_code,
                        "sic_description": sic_name,
                        "incorporation_date": incorporation_date,
                        "registered_address": registered_address,
                        "pe_owner_name": owner["owner_name"],
                        "pe_owner_type": owner["ownership_type"],
                        "pe_ownership_since": owner["notified_on"],
                        "nature_of_control": owner["nature_of_control"],
                        "checked_at": datetime.now().isoformat()
                    })

                logging.info(f"PE acquisition detected: {company_name} ({company_number}) — owner: {pe_owners[0]['owner_name']}")

    # Save results
    print("\n" + "=" * 60)
    print(f"SCAN COMPLETE")
    print(f"Total companies checked: {total_checked}")
    print(f"PE acquisitions detected: {total_pe_found}")

    if all_results:
        df = pd.DataFrame(all_results)
        # Sort by most recent acquisition first
        df = df.sort_values("pe_ownership_since", ascending=False)
        filename = f"pe_acquisitions_{date.today().isoformat()}.csv"
        df.to_csv(filename, index=False)
        print(f"Results saved to: {filename}")
        print("\nMost recent acquisitions:")
        print(df[["company_name", "pe_owner_name", "pe_ownership_since", "sic_description"]].head(10).to_string(index=False))
    else:
        print("No PE acquisitions detected in this scan")

    return all_results

# ── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Scaled-up scan: 500 companies per SIC across the 5 core codes.
    # Expect a ~1–1.5 hour runtime (2 API calls per company, rate-limited).
    # Intra-group holding structures are filtered out automatically and logged.
    run_companies_house_pipeline(
        sic_codes=["64191", "64999", "66190", "64201", "64303"],
        companies_per_sic=500
    )