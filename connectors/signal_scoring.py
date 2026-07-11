"""
Signal Scoring — two-tier, NON-BLENDED.

TIER 1 (per-deal detection confidence): "How confident am I this is a genuine,
in-scope, recent mid-market FS PE deal worth Palladium's attention?" Built ONLY
from signals that genuinely vary per candidate and are attributable to the deal:
Companies House attributes + FCA regulatory status. (News and Reed jobs were
shown 0/60 attributable, so they are NOT here.)

TIER 2 (sector context): "What broader sector context is this happening in?"
A SINGLE set of figures (news sentiment, LDA topics, Trends momentum) that is
the SAME for every deal. Reported ALONGSIDE Tier 1, never blended into it —
blending would falsely imply the sector context discriminates between deals.

IMPORTANT: this is a DETECTION-CONFIDENCE score, NOT a value-creation measure.
The data cannot quantify per-deal value creation; it can rank how likely a
candidate is a real, in-scope, recent deal.

Weighting: components are EQUAL-weighted by default and fully broken out in the
output, so the score is auditable and the equal-weighting is a stated, defensible
choice (absent prior evidence to weight otherwise).
"""

import os
import re
import glob
import pandas as pd
from datetime import date, datetime

DATA_DIR = "data"

# AUTOMATED non-FS contaminant rule: keywords indicating the underlying business
# is NOT financial services (it was swept in via a financial holding SIC code).
# This is a RULE applied to name + SIC description — NOT a hand-picked company
# list — so it varies legitimately and can be validated against the manual list.
NON_FS_KEYWORDS = {
    "agri", "agro", "agricultural", "berry", "berries", "farm", "crop",
    "orchard", "harvest", "produce",
    "petroleum", "energy", "oil", "gas", "fuel", "power", "regen", "solar",
    "estate", "estates", "property", "land", "housing", "woodland", "forest",
    "studio", "restaurant", "retail", "store", "hospital", "leisure",
    "logistics", "transport", "manufacturing", "mining", "construction",
}

# MANUAL eyeball list (from earlier inspection) — used ONLY as a validation
# cross-check against the automated rule, NEVER as a scoring input.
MANUAL_NON_FS = {
    "FRESH BERRY INTERNATIONAL LIMITED", "AGROBERRIES LIMITED",
    "PETROLEUM POWER CO LTD", "TELFORD SEVEN LTD",
    "ANGLIAN REGEN AG CO LTD", "HADRIAN REGEN AG CO LTD",
}

# Owner-name PE-type patterns (genuine PE/investment vehicle markers).
PE_OWNER_PATTERNS = {
    "capital", "partners", "equity", "ventures", "venture", "investment",
    "investments", "fund", "buyout", "growth", "asset management",
    "private equity", "holdings", "advisers", "advisors",
}
WINDOW_START = datetime(2021, 1, 1)
WINDOW_END = datetime(2026, 6, 30)

def latest_csv(pattern, explicit=None):
    if explicit and os.path.exists(explicit):
        return explicit
    m = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    return m[-1] if m else None

# ── Tier-1 components (each returns 0..1) ──────────────────────────────────
def recency_score(date_str):
    """More recent acquisition → higher (0..1 across the 2021–2026 window)."""
    try:
        d = datetime.fromisoformat(str(date_str)[:10])
    except Exception:
        return 0.0
    d = max(WINDOW_START, min(WINDOW_END, d))
    span = (WINDOW_END - WINDOW_START).days
    return round((d - WINDOW_START).days / span, 3) if span else 0.0

def non_fs_flag(name, sic_desc, owner_name=""):
    """Return (is_non_fs: bool, matched_keyword). Automated rule: does the
    company NAME, SIC description, OR owner name contain a non-financial
    business marker? (Owner name included because the underlying business
    sector often shows there, e.g. 'Horizon Estate Holdings'.)"""
    text = (str(name) + " " + str(sic_desc) + " " + str(owner_name)).lower()
    for kw in NON_FS_KEYWORDS:
        if re.search(r"\b" + re.escape(kw), text):
            return True, kw
    return False, ""

def fs_score(name, sic_desc, owner_name=""):
    """1.0 if NOT flagged non-FS by the automated rule, else 0.0."""
    is_non_fs, _ = non_fs_flag(name, sic_desc, owner_name)
    return 0.0 if is_non_fs else 1.0

def owner_pe_score(owner_name):
    """1.0 if owner name matches a PE/investment-vehicle pattern, else 0.0."""
    o = str(owner_name).lower()
    return 1.0 if any(p in o for p in PE_OWNER_PATTERNS) else 0.0

def fca_score(match_quality):
    """confirmed=1.0, review=0.5, no_match/blank=0.0. (Low-information for this
    dataset — most candidates are no_match — but included as a real signal.)"""
    q = str(match_quality).strip().lower()
    return {"confirmed": 1.0, "review": 0.5}.get(q, 0.0)

# ── Build Tier-1 ───────────────────────────────────────────────────────────
def build_tier1(candidates_csv=None, annotated_csv=None, fca_csv=None):
    cand_path = latest_csv("pe_acquisitions*.csv", candidates_csv)
    if not cand_path:
        print("  No candidates CSV.")
        return None
    df = pd.read_csv(cand_path, dtype=str).fillna("")
    df = df.drop_duplicates("company_name").reset_index(drop=True)

    # Merge mid-market annotation (is_ftse_subsidiary) if available.
    ann_path = latest_csv("candidates_annotated_*.csv", annotated_csv)
    if ann_path:
        ann = pd.read_csv(ann_path, dtype=str).fillna("")
        if "is_ftse_subsidiary" in ann.columns:
            m = ann.drop_duplicates("company_name").set_index("company_name")["is_ftse_subsidiary"]
            df["is_ftse_subsidiary"] = df["company_name"].map(m).fillna("False")
    if "is_ftse_subsidiary" not in df.columns:
        df["is_ftse_subsidiary"] = "False"

    # Merge FCA match quality if available.
    fca_path = latest_csv("fca_validation_*.csv", fca_csv)
    fca_map = {}
    if fca_path:
        f = pd.read_csv(fca_path, dtype=str).fillna("")
        key = "company_name" if "company_name" in f.columns else None
        if key and "match_quality" in f.columns:
            fca_map = dict(zip(f[key], f["match_quality"]))
    df["fca_match_quality"] = df["company_name"].map(fca_map).fillna("no_match")

    # Components.
    df["c_not_ftse"] = df["is_ftse_subsidiary"].str.lower().map(
        lambda x: 0.0 if x == "true" else 1.0)
    df["c_fs_not_contaminant"] = df.apply(
        lambda r: fs_score(r.get("company_name", ""), r.get("sic_description", ""), r.get("pe_owner_name", "")),
        axis=1)
    df["c_owner_pe_type"] = df.get("pe_owner_name", "").apply(owner_pe_score) \
        if "pe_owner_name" in df.columns else 0.0
    df["c_recency"] = df.get("pe_ownership_since", "").apply(recency_score) \
        if "pe_ownership_since" in df.columns else 0.0
    df["c_fca"] = df["fca_match_quality"].apply(fca_score)
    # automated non-FS flag for the validation cross-check
    df["auto_non_fs"] = df.apply(
        lambda r: non_fs_flag(r.get("company_name",""), r.get("sic_description",""), r.get("pe_owner_name",""))[0],
        axis=1)
    df["auto_non_fs_kw"] = df.apply(
        lambda r: non_fs_flag(r.get("company_name",""), r.get("sic_description",""), r.get("pe_owner_name",""))[1],
        axis=1)
    # Intra-group survival: every row in pe_acquisitions already SURVIVED the
    # intra-group filter (excluded ones never get written), so this is 1.0 for
    # all — informative as a documented constant, not a discriminator.
    df["c_survived_intragroup"] = 1.0

    # Equal-weighted mean of the DISCRIMINATING components (exclude the constant
    # survival term from the average so it doesn't inflate everyone equally;
    # report it separately).
    comp_cols = ["c_not_ftse", "c_fs_not_contaminant", "c_owner_pe_type", "c_recency", "c_fca"]
    df["tier1_detection_confidence"] = df[comp_cols].mean(axis=1).round(3)

    return df, comp_cols

# ── Tier-2 context (single set, same for all) ──────────────────────────────
def build_tier2():
    ctx = {}
    s = latest_csv("sentiment_articles_*.csv")
    if s:
        sd = pd.read_csv(s)
        if "lm_polarity" in sd:
            ctx["news_mean_lm_polarity"] = round(pd.to_numeric(
                sd["lm_polarity"], errors="coerce").mean(), 3)
    fb = latest_csv("finbert_validation_*.csv")
    if fb:
        fbd = pd.read_csv(fb)
        if "finbert_signed" in fbd:
            ctx["news_mean_finbert"] = round(pd.to_numeric(
                fbd["finbert_signed"], errors="coerce").mean(), 3)
    tr = latest_csv("google_trends_momentum_*.csv")
    if tr:
        trd = pd.read_csv(tr)
        if {"term", "trend_recent_minus_early"}.issubset(trd.columns):
            top = trd.sort_values("trend_recent_minus_early", ascending=False).head(3)
            ctx["top_rising_trends"] = "; ".join(
                f"{r['term']} ({r['trend_recent_minus_early']:+})"
                for _, r in top.iterrows())
    lda = latest_csv("lda_topics_*.csv")
    if lda:
        ld = pd.read_csv(lda)
        if "top_terms" in ld:
            ctx["lda_topic_count"] = len(ld)
    return ctx

# ── Main ───────────────────────────────────────────────────────────────────
def run_scoring():
    print("=" * 64)
    print("SIGNAL SCORING — two-tier, non-blended")
    print("=" * 64)

    built = build_tier1()
    if built is None:
        return None
    df, comp_cols = built

    print("\n" + "=" * 64)
    print("TIER 1 — PER-DEAL DETECTION CONFIDENCE")
    print("(how likely a genuine, in-scope, recent mid-market FS PE deal)")
    print("Components (0..1, equal-wt): not_ftse, fs_not_contaminant, owner_pe, recency, fca")
    print("=" * 64)
    show = df.sort_values("tier1_detection_confidence", ascending=False)
    cols = ["company_name", "tier1_detection_confidence"] + comp_cols
    print(show[cols].to_string(index=False))

    # Distribution note (honesty about discrimination).
    spread = df["tier1_detection_confidence"]
    print(f"\n  Score range: {spread.min():.3f} – {spread.max():.3f} "
          f"(mean {spread.mean():.3f}, sd {spread.std():.3f})")
    if spread.std() < 0.1:
        print("  NOTE: low spread — components don't strongly differentiate candidates")
        print("  (FCA near-constant at 0; main variation is recency + sic_fit).")

    out = os.path.join(DATA_DIR, f"tier1_scores_{date.today().isoformat()}.csv")
    try:
        show[cols + ["c_survived_intragroup", "fca_match_quality"]].to_csv(out, index=False)
    except OSError:
        out = f"tier1_scores_{date.today().isoformat()}.csv"
        show[cols].to_csv(out, index=False)
    print(f"\n  Tier-1 scores saved to: {out}")

    # ── Validation cross-check: automated non-FS rule vs MANUAL eyeball list ──
    print("\n" + "-" * 64)
    print("  CROSS-CHECK: automated non-FS rule vs manual eyeball list")
    auto_flagged = set(df[df["auto_non_fs"]]["company_name"])
    manual = MANUAL_NON_FS
    caught = manual & auto_flagged
    missed = manual - auto_flagged
    extra = auto_flagged - manual
    print(f"  Manual list ({len(manual)}): {', '.join(sorted(manual))}")
    print(f"  Rule caught (true positives): {len(caught)}/{len(manual)} — "
          f"{', '.join(sorted(caught)) or 'none'}")
    if missed:
        print(f"  Rule MISSED (false negatives): {', '.join(sorted(missed))}")
    if extra:
        print(f"  Rule ALSO flagged (not on manual list — review): "
              f"{', '.join(f'{c}[{df[df.company_name==c].auto_non_fs_kw.iat[0]}]' for c in sorted(extra))}")
    print("  (The manual list is NOT a scoring input — this only measures how")
    print("   well the automated rule reproduces the manual judgement.)")

    print("\n" + "=" * 64)
    print("TIER 2 — SECTOR CONTEXT (same for ALL deals; NOT blended into Tier 1)")
    print("=" * 64)
    ctx = build_tier2()
    if ctx:
        for k, v in ctx.items():
            print(f"  {k}: {v}")
    else:
        print("  (no Tier-2 artifacts found)")
    print("\n  Tier 2 answers 'what sector context?' — it is reported alongside,")
    print("  deliberately NOT combined with Tier 1 (it does not vary by deal).")

    print("\nNOTE: Tier-1 is a DETECTION-CONFIDENCE score, not a value-creation")
    print("measure. Equal component weighting is a stated default. FCA is low-")
    print("information for this dataset (most candidates not on the register).")
    return df

if __name__ == "__main__":
    run_scoring()