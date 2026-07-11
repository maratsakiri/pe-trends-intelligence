import os
import re
import glob
import pandas as pd
import logging
from datetime import date

# ── Setup ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="triangulate.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

DATA_DIR = "data"

# Words that, ALONE, are too generic to be a safe match against news text.
# (A core that reduces to just one of these — or any single common word — is
# treated as unmatchable rather than allowed to match articles spuriously.)
GENERIC_CORE = {
    "finance", "financial", "capital", "investment", "investments", "group",
    "holdings", "partners", "ventures", "securities", "global", "international",
    "trust", "wealth", "asset", "management", "fund", "advisers", "advisory",
    "services", "uk", "company", "consult", "studio", "watch", "select",
    "innovation", "growth", "enterprise", "frontier", "venture", "development",
    "marketing", "power", "regen", "berry", "berries", "agricultural",
    "northland", "redwood", "firebird", "stopper", "nominee", "placement",
}

# Corporate-form suffixes stripped when reducing a legal name to its core.
# NOTE: we deliberately do NOT strip "financial/finance/capital/investment"
# here — removing them leaves dangerously generic single words (e.g. "Davies
# Financial" -> "davies", which matches any article mentioning a Davies). For
# news matching we keep the full distinctive phrase.
SUFFIX_NOISE = {
    "limited", "ltd", "llp", "plc", "lp", "gp", "co", "uk", "holdings",
    "holding", "group", "the", "and", "sa", "llc", "bv", "srl", "sro",
}

# ── Name normalisation ─────────────────────────────────────────────────────
def core_name(name):
    """
    Reduce a legal company name to a distinctive core string for matching
    against free-text articles. Strips only corporate-form suffixes, NOT
    descriptive words — so the match requires the full distinctive phrase.

    e.g. 'LINDEN HOUSE FINANCIAL SERVICES LIMITED' -> 'linden house financial services'
         'DAVIES FINANCIAL LIMITED'                -> 'davies financial'
         'AWV LTD'                                 -> 'awv'
    """
    if not isinstance(name, str) or not name.strip():
        return ""
    tokens = re.findall(r"[a-z0-9&]+", name.lower())
    core = [t for t in tokens if t not in SUFFIX_NOISE]
    return " ".join(core).strip()

def is_matchable(core):
    """
    Decide whether a core name is distinctive enough to match safely.

    Rejects:
      - empty cores
      - any SINGLE-token core that is a common/generic word or <=3 chars
        (single common words like 'davies', 'innovation' cause false hits)
    A multi-word core (e.g. 'davies financial', 'linden house') is matchable
    because the full phrase is distinctive.
    """
    if not core:
        return False
    tokens = core.split()
    if len(tokens) == 1:
        tok = tokens[0]
        if tok in GENERIC_CORE or len(tok) <= 3:
            return False
    return True

def name_in_text(core, text):
    """
    Word-boundary search for the core name within article text.
    Multi-word cores match as a phrase; single-word cores match whole-word.
    """
    if not core or not text:
        return False
    pattern = r"\b" + re.escape(core) + r"\b"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None

# ── Loading ────────────────────────────────────────────────────────────────
def latest_csv(pattern, explicit=None):
    """Return an explicit path if given, else the most recent file matching
    a glob pattern in DATA_DIR (by filename, which is date-stamped)."""
    if explicit and os.path.exists(explicit):
        return explicit
    matches = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    return matches[-1] if matches else None

def load_candidates(path):
    df = pd.read_csv(path, dtype=str).fillna("")
    needed = {"company_name", "pe_owner_name"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{path} missing columns {needed - set(df.columns)}")
    # One row per company (a company may have several owner rows).
    return df

def load_news_corpus(guardian_path, gnews_path):
    """
    Build a single searchable corpus from the two news sources. Returns a list
    of dicts: {source, title, url, date, text} where text is the concatenated
    searchable fields for that article.
    """
    corpus = []

    if guardian_path and os.path.exists(guardian_path):
        g = pd.read_csv(guardian_path, dtype=str).fillna("")
        for _, r in g.iterrows():
            corpus.append({
                "source": "Guardian",
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "date": r.get("date", ""),
                "text": " ".join([r.get("title", ""), r.get("trail_text", ""),
                                  r.get("body_text", "")]),
            })
        print(f"  Loaded {len(g)} Guardian articles from {guardian_path}")
    else:
        print("  No Guardian CSV found — skipping Guardian")

    if gnews_path and os.path.exists(gnews_path):
        n = pd.read_csv(gnews_path, dtype=str).fillna("")
        for _, r in n.iterrows():
            corpus.append({
                "source": "Google News",
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "date": r.get("published", ""),
                "text": " ".join([r.get("title", ""), r.get("description", "")]),
            })
        print(f"  Loaded {len(n)} Google News articles from {gnews_path}")
    else:
        print("  No Google News CSV found — skipping Google News")

    return corpus

# ── Triangulation ──────────────────────────────────────────────────────────
def triangulate(candidates_df, corpus):
    """
    For each candidate, search the news corpus for its company core name and
    its PE owner core name. Returns a results DataFrame with match details.
    """
    results = []
    # De-duplicate candidates to one row per company, keeping owner list.
    grouped = candidates_df.groupby("company_name", sort=False).agg({
        "pe_owner_name": lambda s: sorted(set(x for x in s if x)),
        "sic_description": "first" if "sic_description" in candidates_df else (lambda s: ""),
        "pe_ownership_since": "first" if "pe_ownership_since" in candidates_df else (lambda s: ""),
        "company_number": "first" if "company_number" in candidates_df else (lambda s: ""),
    }).reset_index()

    for _, row in grouped.iterrows():
        company = row["company_name"]
        owners = row["pe_owner_name"]
        comp_core = core_name(company)
        comp_matchable = is_matchable(comp_core)

        company_hits = []
        owner_hits = []

        for art in corpus:
            if comp_matchable and name_in_text(comp_core, art["text"]):
                company_hits.append(art)
            for owner in owners:
                ocore = core_name(owner)
                if is_matchable(ocore) and name_in_text(ocore, art["text"]):
                    owner_hits.append((owner, art))

        # Confidence: company name in news is the strong signal; owner-only is
        # weaker (PE firms appear in many unrelated articles).
        if company_hits:
            confidence = "HIGH"
        elif owner_hits:
            confidence = "LOW"
        elif not comp_matchable:
            confidence = "UNMATCHABLE"
        else:
            confidence = "NONE"

        # Build a compact, de-duplicated evidence string.
        evidence_urls = sorted(set(
            a["url"] for a in company_hits
        ))[:3]

        results.append({
            "company_name": company,
            "company_number": row.get("company_number", ""),
            "sic_description": row.get("sic_description", ""),
            "pe_owner_name": "; ".join(owners),
            "pe_ownership_since": row.get("pe_ownership_since", ""),
            "news_confidence": confidence,
            "company_news_hits": len(company_hits),
            "owner_news_hits": len(owner_hits),
            "company_core_searched": comp_core if comp_matchable else "(too generic)",
            "evidence_urls": "; ".join(evidence_urls),
        })

    return pd.DataFrame(results)

# ── Targeted second pass (live API search per candidate) ───────────────────
def targeted_search(company_name):
    """
    Second-pass validation: search the LIVE news APIs for this specific
    company by name, reusing the existing connector functions. Tries the
    cleaned distinctive name and the name minus its corporate suffix
    (journalism rarely uses '... LIMITED'). Degrades gracefully — a source
    that errors or is rate-limited contributes zero, logged not raised.
    """
    try:
        from guardian import check_company_in_news
    except Exception as e:
        logging.warning(f"Guardian targeted search unavailable: {e}")
        check_company_in_news = None
    try:
        from google_news import check_company_in_google_news
    except Exception as e:
        logging.warning(f"Google News targeted search unavailable: {e}")
        check_company_in_google_news = None

    cleaned = core_name(company_name)
    variants = []
    if is_matchable(cleaned):
        variants.append(cleaned)
    stripped = re.sub(r"\b(limited|ltd|plc|llp)\b\.?$", "", company_name,
                      flags=re.IGNORECASE).strip()
    if stripped and stripped.lower() != cleaned:
        variants.append(stripped)
    if not variants:
        variants = [company_name]

    g_hits, n_hits = [], []
    if check_company_in_news:
        for v in variants:
            try:
                res = check_company_in_news(v)
                if res:
                    g_hits.extend(res); break
            except Exception as e:
                logging.warning(f"Guardian targeted '{v}' failed: {e}")
    if check_company_in_google_news:
        for v in variants:
            try:
                res = check_company_in_google_news(v)
                if res:
                    n_hits.extend(res); break
            except Exception as e:
                logging.warning(f"Google News targeted '{v}' failed: {e}")

    g_urls = [a.get("webUrl", "") for a in g_hits][:2]
    n_urls = [a.get("url", "") for a in n_hits][:2]
    return {
        "targeted_guardian_hits": len(g_hits),
        "targeted_gnews_hits": len(n_hits),
        "targeted_confidence": "HIGH" if (g_hits or n_hits) else "NONE",
        "targeted_evidence": "; ".join(filter(None, g_urls + n_urls)),
    }

def run_targeted_pass(results_df):
    """
    Run the live targeted search for every candidate, appending targeted_*
    columns. Slow and rate-limited — intended as an opt-in second pass.
    """
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    print("\n" + "=" * 60)
    print("TARGETED SECOND PASS — live per-candidate news search")
    print(f"Querying {len(results_df)} candidates against live APIs...")
    print("(rate-limited; Google News may be patchy — see triangulate.log)")
    print("=" * 60)

    rows = []
    n = len(results_df)
    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        name = row["company_name"]
        t = targeted_search(name)
        flag = "HIT" if t["targeted_confidence"] == "HIGH" else "·"
        print(f"  [{i}/{n}] {flag} {name}"
              + (f"  (G:{t['targeted_guardian_hits']} "
                 f"N:{t['targeted_gnews_hits']})"
                 if t["targeted_confidence"] == "HIGH" else ""))
        rows.append(t)

    targeted_df = pd.DataFrame(rows, index=results_df.index)
    return pd.concat([results_df, targeted_df], axis=1)

# ── Main ───────────────────────────────────────────────────────────────────
def run_triangulation(candidates_csv=None, guardian_csv=None, gnews_csv=None,
                      targeted=False):
    print("=" * 60)
    print("Signal Triangulation — CH candidates vs News corpus")
    print("=" * 60)

    cand_path = latest_csv("pe_acquisitions*.csv", candidates_csv)
    if not cand_path:
        print("No candidates CSV found in data/ — run companies_house.py first.")
        return None
    print(f"Candidates: {cand_path}")

    g_path = latest_csv("guardian_articles_*.csv", guardian_csv)
    n_path = latest_csv("google_news_*.csv", gnews_csv)

    candidates_df = load_candidates(cand_path)
    corpus = load_news_corpus(g_path, n_path)
    print(f"  Total articles in corpus: {len(corpus)}")

    if not corpus:
        print("Empty news corpus — nothing to triangulate against.")
        return None

    results = triangulate(candidates_df, corpus)

    # Order by confidence for readability.
    order = {"HIGH": 0, "LOW": 1, "NONE": 2, "UNMATCHABLE": 3}
    results["_o"] = results["news_confidence"].map(order).fillna(9)
    results = results.sort_values(["_o", "company_name"]).drop(columns="_o")

    # Summary
    counts = results["news_confidence"].value_counts().to_dict()
    print("\n" + "=" * 60)
    print("TRIANGULATION COMPLETE")
    print(f"  Total candidates: {len(results)}")
    print(f"  HIGH (company named in news):  {counts.get('HIGH', 0)}")
    print(f"  LOW (only PE owner in news):   {counts.get('LOW', 0)}")
    print(f"  NONE (no news match):          {counts.get('NONE', 0)}")
    print(f"  UNMATCHABLE (name too generic):{counts.get('UNMATCHABLE', 0)}")

    # Optional targeted second pass: live per-candidate API search.
    if targeted:
        results = run_targeted_pass(results)

    out = os.path.join(DATA_DIR, f"triangulation_{date.today().isoformat()}.csv")
    try:
        results.to_csv(out, index=False)
    except OSError:
        out = f"triangulation_{date.today().isoformat()}.csv"
        results.to_csv(out, index=False)
    print(f"\nResults saved to: {out}")

    high = results[results["news_confidence"] == "HIGH"]
    if not high.empty:
        print("\nHIGH-confidence candidates (company appears in PE news):")
        print(high[["company_name", "company_news_hits",
                    "evidence_urls"]].to_string(index=False))
    else:
        print("\nNo HIGH-confidence candidates (corpus pass) — no company name "
              "appeared in the collected news corpus.")

    # Comparison summary if the targeted pass ran.
    if targeted and "targeted_confidence" in results.columns:
        t_high = results[results["targeted_confidence"] == "HIGH"]
        print("\n" + "=" * 60)
        print("CORPUS vs TARGETED COMPARISON")
        print(f"  Corpus pass   HIGH: {len(high)} / {len(results)}")
        print(f"  Targeted pass HIGH: {len(t_high)} / {len(results)}")
        if not t_high.empty:
            print("\nCandidates corroborated by targeted live search:")
            print(t_high[["company_name", "targeted_guardian_hits",
                          "targeted_gnews_hits", "targeted_evidence"]]
                  .to_string(index=False))
        else:
            print("  Targeted search also found no corroboration — strengthens "
                  "the coverage-bias reading (not a corpus-collection artifact).")

    return results

if __name__ == "__main__":
    import sys
    # Run the targeted live second pass with:  python triangulate.py --targeted
    do_targeted = "--targeted" in sys.argv
    run_triangulation(targeted=do_targeted)