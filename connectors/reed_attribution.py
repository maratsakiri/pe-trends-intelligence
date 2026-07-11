"""
Reed Jobs Attribution Check — does any job posting belong to a candidate?

Purpose: decide whether Reed job postings are a TIER-1 (per-deal) signal or
only a sector-level one. A posting counts as attributable ONLY if its EMPLOYER
name matches a detected candidate company. Matching on job-description keywords
(e.g. "private equity") does NOT attribute a posting to a deal — it just means
the description used the phrase. So we match candidate name vs the `employer`
field only.

Same rigour as the news triangulation:
  - normalise names, reject cores too generic/short to match safely
  - report RAW hits, then require manual verification before counting confirmed
  - distinguish genuine employer matches from coincidental token overlap
"""

import os
import re
import glob
import pandas as pd
from datetime import date

DATA_DIR = "data"

SUFFIX_NOISE = {
    "limited", "ltd", "llp", "plc", "lp", "gp", "co", "uk", "holdings",
    "holding", "group", "the", "and", "sa", "llc", "bv", "srl", "sro",
}
GENERIC_CORE = {
    "finance", "financial", "capital", "investment", "investments", "group",
    "holdings", "partners", "ventures", "securities", "global", "international",
    "trust", "wealth", "asset", "management", "fund", "advisers", "advisory",
    "services", "uk", "company", "consult", "studio", "watch", "select",
    "innovation", "growth", "enterprise", "frontier", "venture", "development",
    "marketing", "power", "regen", "berry", "berries", "agricultural",
}

def core_name(name):
    if not isinstance(name, str) or not name.strip():
        return ""
    tokens = re.findall(r"[a-z0-9&]+", name.lower())
    core = [t for t in tokens if t not in SUFFIX_NOISE]
    return " ".join(core).strip()

def is_matchable(core):
    if not core:
        return False
    toks = core.split()
    if len(toks) == 1:
        t = toks[0]
        if t in GENERIC_CORE or len(t) <= 3:
            return False
    return True

def name_match(core, employer):
    """Whole-phrase, word-boundary match of a candidate core in an employer name."""
    if not core or not employer:
        return False
    return re.search(r"\b" + re.escape(core) + r"\b", employer.lower()) is not None

def latest_csv(pattern, explicit=None):
    if explicit and os.path.exists(explicit):
        return explicit
    m = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    return m[-1] if m else None

def run_attribution(candidates_csv=None, reed_csv=None):
    print("=" * 60)
    print("Reed Jobs Attribution Check (Tier-1 eligibility)")
    print("=" * 60)

    cand_path = latest_csv("pe_acquisitions*.csv", candidates_csv)
    reed_path = latest_csv("reed_jobs_*.csv", reed_csv)
    if not cand_path or not reed_path:
        print(f"  Missing input — candidates:{cand_path} reed:{reed_path}")
        return None
    print(f"  Candidates: {cand_path}")
    print(f"  Reed jobs:  {reed_path}")

    cand = pd.read_csv(cand_path, dtype=str).fillna("")
    reed = pd.read_csv(reed_path, dtype=str).fillna("")
    if "employer" not in reed.columns:
        print("  Reed CSV has no 'employer' column.")
        return None

    companies = cand["company_name"].dropna().drop_duplicates().tolist()
    employers = reed["employer"].dropna().tolist()
    print(f"  Candidates: {len(companies)} | Reed postings: {len(reed)} "
          f"({reed['employer'].nunique()} unique employers)")

    rows = []
    unmatchable = 0
    for comp in companies:
        core = core_name(comp)
        if not is_matchable(core):
            unmatchable += 1
            continue
        hits = []
        for _, r in reed.iterrows():
            if name_match(core, r["employer"]):
                hits.append(r)
        if hits:
            for h in hits:
                rows.append({
                    "candidate": comp,
                    "candidate_core": core,
                    "reed_employer": h["employer"],
                    "job_title": h.get("job_title", ""),
                    "date_posted": h.get("date_posted", ""),
                    "job_url": h.get("job_url", ""),
                })

    print("\n" + "=" * 60)
    print("RESULT")
    print(f"  Candidates unmatchable (name too generic/short): {unmatchable}")
    if rows:
        out = pd.DataFrame(rows)
        print(f"  RAW employer-name hits: {len(out)} "
              f"across {out['candidate'].nunique()} candidates")
        print("\n  Raw hits (VERIFY each manually before counting as confirmed):")
        print(out[["candidate", "reed_employer", "job_title"]].to_string(index=False))
        path = os.path.join(DATA_DIR, f"reed_attribution_{date.today().isoformat()}.csv")
        try:
            out.to_csv(path, index=False)
        except OSError:
            path = f"reed_attribution_{date.today().isoformat()}.csv"
            out.to_csv(path, index=False)
        print(f"\n  Saved to: {path}")
        print("\n  → MANUAL CHECK NEEDED: is each reed_employer genuinely the same")
        print("    company as the candidate, or a coincidental name overlap?")
    else:
        print("  RAW employer-name hits: 0")
        print("\n  → No Reed posting is attributable to any candidate by employer name.")
        print("    CONCLUSION: jobs is NOT a Tier-1 per-deal signal for this dataset;")
        print("    Reed data is sector-level only (like news). Tier-1 = CH + FCA.")
    return rows

if __name__ == "__main__":
    run_attribution()