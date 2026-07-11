import os
import re
import glob
import pandas as pd
import logging
from datetime import date

# ── Setup ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="midmarket.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

DATA_DIR = "data"

# ── FTSE-listed large-cap parent list (curated heuristic) ──────────────────
# Palladium's focus is the MID-MARKET. Scope definition (per methodology):
# a candidate is OUT of scope if its PE-owner is a FTSE 350 constituent group
# (FTSE 100 + FTSE 250) — an objective, citable large-cap threshold. AIM-listed
# parents are NOT excluded (they fall below the FTSE 350 line). This is a
# CURATED list of FTSE 350 financial / financial-adjacent groups likely to
# appear as PSC owners in FS contexts — NOT the full FTSE 350 (most of which,
# e.g. miners/retailers, never appear here). Documented heuristic, updated
# manually; catches the obvious large-caps but is not an authoritative
# ownership graph. Source: FTSE 100/250 constituents (FTSE Russell, 2026).
FTSE_PARENTS = {
    # Banks (FTSE 100)
    "barclays": "Barclays (FTSE 100)",
    "hsbc": "HSBC (FTSE 100)",
    "lloyds": "Lloyds Banking Group (FTSE 100)",
    "natwest": "NatWest Group (FTSE 100)",
    "standard chartered": "Standard Chartered (FTSE 100)",
    # Insurers / life / pensions (FTSE 100)
    "aviva": "Aviva (FTSE 100)",
    "legal & general": "Legal & General (FTSE 100)",
    "legal and general": "Legal & General (FTSE 100)",
    "prudential": "Prudential (FTSE 100)",
    "phoenix group": "Phoenix Group (FTSE 100)",
    "m&g": "M&G (FTSE 100)",
    "aegon": "Aegon UK (listed parent, Aegon NV)",
    # Asset / wealth managers (FTSE 100/250)
    "schroders": "Schroders (FTSE 100)",
    "abrdn": "abrdn (FTSE 100/250)",
    "st. james's place": "St. James's Place (FTSE 100)",
    "st james's place": "St. James's Place (FTSE 100)",
    "rathbones": "Rathbones (FTSE 250)",
    "quilter": "Quilter (FTSE 250)",
    "jupiter fund": "Jupiter Fund Management (FTSE 250)",
    "man group": "Man Group (FTSE 250)",
    "intermediate capital": "Intermediate Capital Group (FTSE 100)",
    "bridgepoint": "Bridgepoint Group (FTSE 250, listed)",
    "3i": "3i Group (FTSE 100)",
    # Market infrastructure (FTSE 100)
    "london stock exchange": "London Stock Exchange Group (FTSE 100)",
    "lseg": "London Stock Exchange Group (FTSE 100)",
    # Consumer finance / specialty (FTSE 250)
    "provident": "Provident Financial / Vanquis Banking (FTSE 250)",
    "vanquis banking": "Vanquis Banking Group (FTSE 250)",
    "close brothers": "Close Brothers (FTSE 250)",
    "paragon bank": "Paragon Banking Group (FTSE 250)",
    "investec": "Investec (FTSE 250)",
}

# ── Matching ───────────────────────────────────────────────────────────────
def normalise(text):
    return re.sub(r"[^a-z0-9& ]", " ", str(text).lower())

def match_ftse_parent(owner_name):
    """
    Return (matched: bool, parent_label: str|None) if the owner name contains
    a FTSE-listed parent token. Uses word-aware substring matching on the
    normalised owner name.
    """
    norm = normalise(owner_name)
    for token, label in FTSE_PARENTS.items():
        # token may be multi-word; match as a contained phrase with boundaries
        pattern = r"(?:^|\b)" + re.escape(token) + r"(?:\b|$)"
        if re.search(pattern, norm):
            return True, label
    return False, None

# ── Pipeline ───────────────────────────────────────────────────────────────
def latest_csv(pattern, explicit=None):
    if explicit and os.path.exists(explicit):
        return explicit
    matches = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    return matches[-1] if matches else None

def run_midmarket_filter(candidates_csv=None, drop=False):
    """
    Tag each candidate with whether its PE owner is a FTSE-listed group
    (i.e. a large-cap, out of Palladium's mid-market scope).

    By default TAGS (adds columns) and writes the full annotated set, so the
    exclusion is transparent and auditable. Pass drop=True to also write a
    filtered CSV with large-caps removed.
    """
    path = latest_csv("pe_acquisitions*.csv", candidates_csv)
    if not path:
        print("No candidates CSV found in data/ — run companies_house.py first.")
        return None

    print("=" * 60)
    print("Mid-Market Filter — FTSE-listed-group subsidiary exclusion")
    print(f"Candidates: {path}")
    print("=" * 60)

    df = pd.read_csv(path, dtype=str).fillna("")
    if "pe_owner_name" not in df.columns:
        print("CSV has no 'pe_owner_name' column.")
        return None

    labels, flags = [], []
    for _, row in df.iterrows():
        is_ftse, label = match_ftse_parent(row["pe_owner_name"])
        flags.append(is_ftse)
        labels.append(label or "")
        if is_ftse:
            logging.info(f"FTSE-parent excluded: {row.get('company_name','')} "
                         f"<- {row['pe_owner_name']} ({label})")

    df["is_ftse_subsidiary"] = flags
    df["ftse_parent"] = labels

    n_ftse = sum(flags)
    n_unique_ftse = df[df["is_ftse_subsidiary"]]["company_name"].nunique()
    n_unique_total = df["company_name"].nunique()

    print(f"\nFlagged as FTSE-listed-group subsidiaries (large-cap, out of "
          f"mid-market scope):")
    if n_ftse:
        shown = df[df["is_ftse_subsidiary"]][
            ["company_name", "pe_owner_name", "ftse_parent"]
        ].drop_duplicates("company_name")
        print(shown.to_string(index=False))
    else:
        print("  (none)")

    print(f"\nUnique companies: {n_unique_total} total | "
          f"{n_unique_ftse} large-cap excluded | "
          f"{n_unique_total - n_unique_ftse} remain in mid-market scope")

    # Always write the annotated full set.
    annotated = os.path.join(DATA_DIR, f"candidates_annotated_{date.today().isoformat()}.csv")
    try:
        df.to_csv(annotated, index=False)
    except OSError:
        annotated = f"candidates_annotated_{date.today().isoformat()}.csv"
        df.to_csv(annotated, index=False)
    print(f"\nAnnotated set saved to: {annotated}")

    if drop:
        midmarket = df[~df["is_ftse_subsidiary"]].copy()
        filtered = os.path.join(DATA_DIR, f"candidates_midmarket_{date.today().isoformat()}.csv")
        try:
            midmarket.to_csv(filtered, index=False)
        except OSError:
            filtered = f"candidates_midmarket_{date.today().isoformat()}.csv"
            midmarket.to_csv(filtered, index=False)
        print(f"Mid-market-only set saved to: {filtered}")

    return df

if __name__ == "__main__":
    import sys
    do_drop = "--drop" in sys.argv
    run_midmarket_filter(drop=do_drop)