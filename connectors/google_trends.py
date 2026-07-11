import os
import time
import logging
import pandas as pd
from datetime import date

# ── Setup ──────────────────────────────────────────────────────────────────
# Google Trends connector — SECTOR-LEVEL momentum signal only.
#
# IMPORTANT CONTEXT (document in methodology):
#   * There is no official, generally-available Google Trends API (the 2025
#     official API remains limited alpha). This uses `pytrends`, an unofficial
#     library that was ARCHIVED / unmaintained by its authors on 2025-04-17.
#     It still works for light research use but can break without warning and
#     rate-limits aggressively (HTTP 429). We sleep heavily and fail soft.
#   * Trends returns RELATIVE interest (0–100, normalised within each request),
#     NOT absolute search volumes. All analysis must treat values as relative.
#   * This is a SECTOR signal (e.g. interest in "private equity",
#     "digital transformation"), NOT a per-company signal — nobody searches
#     obscure mid-market entity names, so Trends cannot score individual deals.
#     Lit basis: Choi & Varian (2012); Preis, Moat & Stanley (2013).
#
# Install first:  pip install pytrends --break-system-packages

logging.basicConfig(
    filename="google_trends.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

DATA_DIR = "data"

# ── Search terms (sector-level) ────────────────────────────────────────────
# Grouped so we can interpret momentum by theme. Trends compares up to 5 terms
# per request (relative to each other), so we batch in <=5s and keep an anchor
# term in each batch for cross-batch comparability where useful.
PE_TERMS = [
    "private equity",
    "private equity acquisition",
    "buyout",
    "leveraged buyout",
    "portfolio company",
]

TRANSFORMATION_TERMS = [
    "digital transformation",
    "cloud migration",
    "data analytics",
    "artificial intelligence strategy",
    "ERP implementation",
]

FS_SECTOR_TERMS = [
    "fintech",
    "wealth management",
    "asset management",
    "financial services M&A",
    "insurance technology",
]

# AI / digital-finance batch — head terms (short, natural-language) chosen to
# capture the post-2022 generative-AI surge. "ChatGPT" is included as a DATABLE
# ANCHOR: its launch (Nov 2022) is a known inflection point, giving a reference
# curve against which the other AI terms' timing can be read. NOTE: "AI" is a
# very broad head term — it captures general AI interest, not finance-specific;
# "AI in finance" is on-target but lower-volume. This broad-vs-precise trade-off
# is deliberate and should be noted in interpretation.
# AI headline batch — ONLY the two validated headline terms. "ChatGPT" anchors
# the timing (Nov-2022 launch), "AI" gives the dominant surge magnitude. The
# finance/specific AI terms are deliberately NOT here (bare "AI" crushes them);
# they live in AI_DETAIL_TERMS below, scaled against each other. No term appears
# in more than one batch — so the per-batch summary never double-counts.
AI_TERMS = [
    "AI",
    "ChatGPT",
]

# AI detail batch — finance/specific AI terms WITHOUT bare "AI", normalised
# against each other so they are interpretable (in the same batch as "AI" they
# flatten to ~0, an artifact not a finding). Read relative interest AMONG these
# specific concepts here; read the headline surge from AI_TERMS above. These
# two batches are scaled separately — compare across them by shape/timing only,
# never by level.
AI_DETAIL_TERMS = [
    "generative AI",
    "AI in finance",
    "machine learning",
    "AI automation",
    "AI banking",
]

TERM_BATCHES = {
    "pe": PE_TERMS,
    "transformation": TRANSFORMATION_TERMS,
    "fs_sector": FS_SECTOR_TERMS,
    "ai": AI_TERMS,
    "ai_detail": AI_DETAIL_TERMS,
}

# ── Geo / timeframe ────────────────────────────────────────────────────────
GEO = "GB"                     # United Kingdom
TIMEFRAME = "2021-01-01 2026-06-16"   # aligned to pipeline's Jan-2021 cutoff

# ── Core fetch (with graceful failure) ─────────────────────────────────────
def get_pytrends():
    """
    Lazily import and construct a pytrends client. Returns the client or None
    if pytrends isn't installed (so the rest of the pipeline can continue).
    """
    try:
        from pytrends.request import TrendReq
    except ImportError:
        msg = ("pytrends not installed. Run: "
               "pip install pytrends --break-system-packages")
        print(f"  {msg}")
        logging.error(msg)
        return None
    # hl/tz are UI locale + timezone offset (minutes); GB ~ UTC.
    return TrendReq(hl="en-GB", tz=0)

def fetch_batch(pytrends, terms, batch_name, retries=2):
    """
    Fetch interest-over-time for up to 5 terms. Returns a long-format DataFrame
    [date, term, interest, batch] or empty DataFrame on failure. Sleeps
    heavily and backs off on rate-limit (429) — pytrends is fragile.
    """
    for attempt in range(retries + 1):
        try:
            time.sleep(2.0)  # baseline politeness between requests
            pytrends.build_payload(terms, timeframe=TIMEFRAME, geo=GEO)
            df = pytrends.interest_over_time()
            if df is None or df.empty:
                logging.info(f"Batch '{batch_name}': empty result")
                return pd.DataFrame()
            if "isPartial" in df.columns:
                df = df.drop(columns="isPartial")
            long = df.reset_index().melt(
                id_vars="date", var_name="term", value_name="interest"
            )
            long["batch"] = batch_name
            return long
        except Exception as e:
            wait = 30 * (attempt + 1)
            logging.warning(f"Batch '{batch_name}' attempt {attempt+1} failed: "
                            f"{e} — waiting {wait}s")
            print(f"  Batch '{batch_name}' failed (attempt {attempt+1}) — "
                  f"likely rate limit. Waiting {wait}s...")
            time.sleep(wait)
    logging.error(f"Batch '{batch_name}' gave up after {retries+1} attempts")
    print(f"  Batch '{batch_name}' gave up — skipping (see google_trends.log)")
    return pd.DataFrame()

# ── Momentum summary ───────────────────────────────────────────────────────
def summarise_momentum(long_df):
    """
    Per (term, batch), compute momentum indicators on the relative-interest
    series: mean, last value, and recent-vs-earlier trend (mean of last ~6
    months minus mean of first ~6 months). All values are RELATIVE (0–100).

    Grouped by (term, batch) — NOT term alone — so a term appearing in more
    than one batch is never pooled across two different normalisations. With
    the current batches no term repeats, but this keeps the summary correct by
    construction.
    """
    rows = []
    for (term, batch), g in long_df.groupby(["term", "batch"]):
        g = g.sort_values("date")
        series = g["interest"].astype(float)
        first6 = series.head(26).mean()
        last6 = series.tail(26).mean()
        rows.append({
            "term": term,
            "batch": batch,
            "mean_interest": round(series.mean(), 1),
            "latest_interest": round(series.iloc[-1], 1),
            "trend_recent_minus_early": round(last6 - first6, 1),
            "n_points": len(series),
        })
    out = pd.DataFrame(rows).sort_values(
        "trend_recent_minus_early", ascending=False
    )
    return out

# ── Main pipeline ──────────────────────────────────────────────────────────
def run_google_trends_pipeline(batches=None):
    print("=" * 60)
    print("Google Trends — Sector Momentum Signal (RELATIVE interest)")
    print(f"Geo: {GEO} | Timeframe: {TIMEFRAME}")
    print("NOTE: pytrends is unmaintained (archived 2025-04-17); values are")
    print("relative (0-100), sector-level only — not per-company.")
    print("=" * 60)

    pytrends = get_pytrends()
    if pytrends is None:
        return None

    target = batches or TERM_BATCHES
    all_long = []

    for batch_name, terms in target.items():
        print(f"\nFetching batch '{batch_name}' ({len(terms)} terms)...")
        long = fetch_batch(pytrends, terms, batch_name)
        if not long.empty:
            n_terms = long["term"].nunique()
            print(f"  OK — {n_terms} terms, {long['date'].nunique()} time points")
            all_long.append(long)

    if not all_long:
        print("\nNo Trends data retrieved (install missing or all batches "
              "rate-limited). See google_trends.log.")
        return None

    long_df = pd.concat(all_long, ignore_index=True)

    # Save the full time series.
    ts_path = os.path.join(DATA_DIR, f"google_trends_{date.today().isoformat()}.csv")
    try:
        long_df.to_csv(ts_path, index=False)
    except OSError:
        ts_path = f"google_trends_{date.today().isoformat()}.csv"
        long_df.to_csv(ts_path, index=False)
    print(f"\nTime series saved to: {ts_path}")

    # Momentum summary.
    summary = summarise_momentum(long_df)
    sum_path = os.path.join(DATA_DIR, f"google_trends_momentum_{date.today().isoformat()}.csv")
    try:
        summary.to_csv(sum_path, index=False)
    except OSError:
        sum_path = f"google_trends_momentum_{date.today().isoformat()}.csv"
        summary.to_csv(sum_path, index=False)
    print(f"Momentum summary saved to: {sum_path}")

    print("\nSector momentum (relative interest, recent vs early):")
    print(summary.to_string(index=False))
    print("\n(Positive trend = rising search interest over the window. "
          "Relative scale; compare within-batch, not across batches.)")

    return long_df

if __name__ == "__main__":
    run_google_trends_pipeline()