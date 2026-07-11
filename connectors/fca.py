import requests
import os
import re
import pandas as pd
import time
import logging
from dotenv import load_dotenv
from datetime import datetime, date

# ── Setup ──────────────────────────────────────────────────────────────────
load_dotenv()
FCA_API_KEY = os.getenv("FCA_API_KEY")
FCA_API_EMAIL = os.getenv("FCA_API_EMAIL")  # the email you signed up with
BASE_URL = "https://register.fca.org.uk/services/V0.1"

logging.basicConfig(
    filename="fca.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# The FCA API authenticates with two headers, NOT basic auth like Companies
# House. Both the signup email and the API key are required on every request.
HEADERS = {
    "Accept": "application/json",
    "X-Auth-Email": FCA_API_EMAIL or "",
    "X-Auth-Key": FCA_API_KEY or "",
}

# Corporate-form / descriptor words stripped when normalising a Companies
# House legal name down to a searchable core. FCA exact search is unforgiving,
# so "Cantor Finance Limited" must be reduced before it will match.
NAME_NOISE = {
    "limited", "ltd", "llp", "plc", "lp", "gp", "sarl", "sa", "nv", "bv",
    "holdings", "holding", "group", "groups", "finance", "financial",
    "services", "service", "capital", "partners", "partnership",
    "investments", "investment", "uk", "the", "company", "co",
    "international", "global", "bidco", "midco", "topco", "newco",
}

# ── Core API function ──────────────────────────────────────────────────────
def make_request(url, params=None):
    """
    Make a rate-limited FCA API request with error handling.
    Returns the parsed JSON or None on failure / genuine no-data.

    Rate limit is 50 requests per 10 seconds; 0.25s spacing keeps us
    comfortably under it even with retries.
    """
    if not FCA_API_KEY or not FCA_API_EMAIL:
        msg = ("FCA credentials missing. Add FCA_API_KEY and FCA_API_EMAIL "
               "to your .env (register at register.fca.org.uk/Developer/s/).")
        print(f"  {msg}")
        logging.error(msg)
        return None

    try:
        time.sleep(0.25)
        response = requests.get(url, params=params, headers=HEADERS, timeout=10)

        if response.status_code == 200:
            data = response.json()
            # The API returns HTTP 200 even for "nothing found". The reliable
            # signal of a genuine miss is an empty/absent/null Data field —
            # NOT the Status code (FSR-API-02-01-00 is a *successful* firm
            # lookup, FSR-API-04-01-00 a successful search). A no-result search
            # comes back with Data: null and Message "No search result found".
            payload = data.get("Data")
            if not payload:
                logging.info(f"No data for {url}: {data.get('Message', '')}")
                return None
            return data
        elif response.status_code == 429:
            print("  Rate limit hit — waiting 10 seconds...")
            logging.warning("Rate limit hit — waiting 10 seconds")
            time.sleep(10)
            return make_request(url, params)  # retry
        elif response.status_code == 401:
            logging.error("FCA auth failed (401) — check email/key headers")
            print("  Auth failed (401) — check FCA_API_EMAIL and FCA_API_KEY")
            return None
        else:
            logging.error(f"FCA API error {response.status_code} for {url}")
            return None

    except requests.exceptions.Timeout:
        logging.error(f"Timeout for {url}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error for {url}: {e}")
        return None

# ── Normalisation helpers ──────────────────────────────────────────────────
def normalise_ch_number(value):
    """
    Companies House numbers are 8 characters. They may appear with or without
    leading zeros, and (for some company types) with a letter prefix like
    'SC' or 'NI'. Normalise to compare reliably across the two registers:
    uppercase, strip non-alphanumerics, zero-pad a purely numeric number to 8.
    """
    if not value:
        return ""
    s = re.sub(r"[^0-9A-Za-z]", "", str(value)).upper()
    if s.isdigit():
        s = s.zfill(8)
    return s

def normalise_name_for_search(name):
    """
    Reduce a Companies House legal name to a searchable core for FCA search.
    Strips corporate-form and generic descriptor words, returns the remaining
    significant tokens. Falls back to the original if stripping leaves nothing.

    e.g. 'Cantor Finance Limited' -> 'cantor'
         'Innovation Investment Capital' -> 'innovation'
    """
    if not name:
        return ""
    tokens = re.findall(r"[a-zA-Z0-9&]+", name.lower())
    core = [t for t in tokens if t not in NAME_NOISE]
    if not core:
        return name.strip()  # everything was noise — search the raw name
    # Use up to the first two significant tokens; more than that over-narrows
    # FCA's exact-ish search.
    return " ".join(core[:2])

# ── Lookup endpoints ───────────────────────────────────────────────────────
def common_search(name, resource_type="firm"):
    """
    Search the register by name. resource_type in {'firm','individual','fund'}.
    Returns the list of match records (possibly empty).
    """
    url = f"{BASE_URL}/Search"
    params = {"q": name, "type": resource_type}
    data = make_request(url, params)
    if not data:
        return []
    return data.get("Data", []) or []

def get_firm(frn):
    """Get the core authorisation record for a firm by FRN."""
    url = f"{BASE_URL}/Firm/{frn}"
    data = make_request(url)
    if not data:
        return None
    records = data.get("Data", [])
    return records[0] if records else None

def get_firm_names(frn):
    """Get current and previous trading names for a firm."""
    url = f"{BASE_URL}/Firm/{frn}/Names"
    data = make_request(url)
    if not data:
        return None
    return data.get("Data", [])

# ── Matching logic ─────────────────────────────────────────────────────────
# SCOPE NOTE (read before writing this up):
# The FCA Register API has NO change-in-control / acquisition-approval
# endpoint, and NO "search by Companies House number" endpoint. This connector
# therefore does regulatory-STATUS validation, not acquisition detection:
# given a Companies House PE detection (name + CH number), it finds the
# matching FCA firm (if any) and reports its authorisation status.
#
# Matching strategy (chosen: auto-confirm CH-number matches, flag rest):
#   1. Search FCA on a normalised version of the name (raw name often misses).
#   2. For each candidate firm, pull its record and read its CH number.
#   3. If a candidate's CH number == the detection's CH number -> CONFIRMED.
#   4. If candidates exist but none match on CH number -> REVIEW (name hits,
#      identity unverified — could be a namesake like the 12 "Cantor" firms).
#   5. If search returns nothing -> NO_MATCH (genuinely not found).
# CH-number verification removes false positives; it cannot recover a firm
# the name search never surfaced, so recall is reported, not assumed.

def match_detection_to_fca(company_name, ch_number):
    """
    Match a single Companies House detection to an FCA firm.
    Returns a result dict with match_quality in
    {'confirmed','review','no_match'} plus status/FRN where known.
    """
    target_ch = normalise_ch_number(ch_number)
    query = normalise_name_for_search(company_name)
    print(f"\nValidating (FCA): {company_name}  [CH {ch_number} -> search '{query}']")

    candidates = common_search(query, "firm")

    if not candidates:
        print("  ✗ No FCA search results — NO MATCH (likely not FCA-regulated)")
        return {
            "company_name": company_name,
            "ch_number": ch_number,
            "search_query": query,
            "match_quality": "no_match",
            "fca_authorised": False,
            "frn": "",
            "fca_status": "",
            "fca_name": "",
            "fca_ch_number": "",
            "n_candidates": 0,
            "candidates": "",
        }

    # Try to confirm by Companies House number across candidates.
    confirmed = None
    candidate_summ = []
    for cand in candidates:
        frn = cand.get("Reference Number")
        candidate_summ.append(
            f"{cand.get('Name','')} ({frn}) [{cand.get('Status','')}]"
        )
        if target_ch:  # only verifiable if we have a CH number to match on
            firm = get_firm(frn)
            if firm and normalise_ch_number(
                firm.get("Companies House Number")
            ) == target_ch:
                confirmed = (frn, firm)
                break  # exact identifier match — stop

    candidates_str = "; ".join(candidate_summ[:8])

    if confirmed:
        frn, firm = confirmed
        status = firm.get("Status", "")
        print(f"  ✓ CONFIRMED by CH number — FRN {frn}, status: {status}")
        return {
            "company_name": company_name,
            "ch_number": ch_number,
            "search_query": query,
            "match_quality": "confirmed",
            "fca_authorised": status.lower().startswith("authorised"),
            "frn": frn,
            "fca_status": status,
            "fca_name": firm.get("Organisation Name", ""),
            "fca_ch_number": firm.get("Companies House Number", ""),
            "n_candidates": len(candidates),
            "candidates": candidates_str,
        }

    # Candidates existed but none matched on CH number → manual review.
    reason = ("no CH number on detection to verify against"
              if not target_ch else
              "name hits found but none matched the CH number")
    print(f"  ~ {len(candidates)} candidate(s), unverified — REVIEW ({reason})")
    return {
        "company_name": company_name,
        "ch_number": ch_number,
        "search_query": query,
        "match_quality": "review",
        "fca_authorised": None,
        "frn": "",
        "fca_status": "",
        "fca_name": "",
        "fca_ch_number": "",
        "n_candidates": len(candidates),
        "candidates": candidates_str,
    }

# ── Input loading ──────────────────────────────────────────────────────────
def load_detections_from_csv(path):
    """
    Load Companies House PE detections (name + CH number) from a CSV produced
    by the Companies House connector. Returns a list of (name, ch_number)
    tuples, de-duplicated on CH number.
    """
    df = pd.read_csv(path, dtype=str).fillna("")
    name_col = "company_name"
    num_col = "company_number"
    if name_col not in df.columns or num_col not in df.columns:
        raise ValueError(
            f"Expected columns '{name_col}' and '{num_col}' in {path}; "
            f"found {list(df.columns)}"
        )
    df = df.drop_duplicates(subset=num_col)
    return list(zip(df[name_col], df[num_col]))

# ── Main pipeline function ─────────────────────────────────────────────────
def run_fca_pipeline(detections=None, input_csv=None):
    """
    Main FCA pipeline.

    Pass EITHER:
      - detections: list of (company_name, ch_number) tuples, or
      - input_csv: path to a Companies House pe_acquisitions CSV.

    Validates each detection's FCA regulatory status, auto-confirming matches
    by Companies House number and flagging the rest for review, then writes
    results to a dated CSV for triangulation.
    """
    if input_csv and not detections:
        detections = load_detections_from_csv(input_csv)

    if not detections:
        print("No detections supplied — nothing to validate.")
        return []

    all_results = []

    print("=" * 60)
    print("FCA Register Regulatory Validation")
    print(f"Validating {len(detections)} detections against the FS Register")
    print("Matching: auto-confirm by CH number, flag the rest for review")
    print("=" * 60)

    for name, ch_number in detections:
        result = match_detection_to_fca(name, ch_number)
        result["checked_at"] = datetime.now().isoformat()
        all_results.append(result)

    # Summary by match quality
    confirmed = sum(1 for r in all_results if r["match_quality"] == "confirmed")
    review = sum(1 for r in all_results if r["match_quality"] == "review")
    no_match = sum(1 for r in all_results if r["match_quality"] == "no_match")

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print(f"  Confirmed (CH-number match): {confirmed}")
    print(f"  Review (name hits, unverified): {review}")
    print(f"  No match (not found):        {no_match}")

    if all_results:
        df = pd.DataFrame(all_results)
        filename = f"data/fca_validation_{date.today().isoformat()}.csv"
        try:
            df.to_csv(filename, index=False)
        except OSError:
            filename = f"fca_validation_{date.today().isoformat()}.csv"
            df.to_csv(filename, index=False)
        print(f"Results saved to: {filename}")
        print("\nSummary:")
        print(df[["company_name", "match_quality", "fca_status",
                  "n_candidates"]].to_string(index=False))

    return all_results

# ── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Validate the real Companies House detections by reading the
    # pe_acquisitions CSV (uses company_name + company_number columns).
    # Change the path here if your detections live in a dated file, e.g.
    #   run_fca_pipeline(input_csv="data/pe_acquisitions_2026-05-17.csv")
    run_fca_pipeline(input_csv="data/pe_acquisitions.csv")