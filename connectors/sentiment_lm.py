"""
Loughran-McDonald Financial Sentiment — Tier-2 sector-context signal.

Scores the collected NEWS CORPUS (Guardian + Google News) using the
Loughran-McDonald (2011) financial sentiment dictionary via pysentiment2.

ROLE (per two-tier architecture): this is a SECTOR-LEVEL contextual signal, not
a per-deal score. The articles are general UK PE / financial-services news, not
text about specific detected companies (news triangulation verified 0/60
per-deal matches). So sentiment here characterises the SECTOR MOOD over time —
it corroborates the context in which value creation occurs, and is read as a
time series, not attached to individual deals.

Why Loughran-McDonald (lit review 2.3.1): general-purpose sentiment dictionaries
misclassify financial text (e.g. "liability", "tax" are neutral in finance); the
L-M dictionary is domain-specific, transparent, and interpretable — the gold
standard for financial sentiment where interpretability matters.

Install first:  pip install pysentiment2 --break-system-packages
"""

import os
import re
import glob
import logging
import pandas as pd
from datetime import date

logging.basicConfig(
    filename="sentiment.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

DATA_DIR = "data"

# ── Dictionary loader (graceful) ───────────────────────────────────────────
def get_lm():
    """Return a pysentiment2 LM instance, or None if not installed."""
    try:
        import pysentiment2 as ps
    except ImportError:
        msg = ("pysentiment2 not installed. Run: "
               "pip install pysentiment2 --break-system-packages")
        print(f"  {msg}")
        logging.error(msg)
        return None
    return ps.LM()

# ── Scoring ────────────────────────────────────────────────────────────────
def score_text(lm, text):
    """
    Score one piece of text with the L-M dictionary.
    Returns dict with positive/negative word counts, polarity, subjectivity,
    a net-sentiment ratio, and the token count (for transparency/weighting).

    L-M polarity = (Pos - Neg) / (Pos + Neg); range [-1, 1].
    Net sentiment here = (Pos - Neg) / total_tokens, which is less volatile for
    short texts and is reported alongside polarity.
    """
    if not isinstance(text, str) or not text.strip():
        return {"lm_positive": 0, "lm_negative": 0, "lm_polarity": 0.0,
                "lm_subjectivity": 0.0, "lm_net_sentiment": 0.0, "lm_tokens": 0}
    tokens = lm.tokenize(text)
    score = lm.get_score(tokens)
    pos = score.get("Positive", 0)
    neg = score.get("Negative", 0)
    n = len(tokens) if tokens else 0
    net = (pos - neg) / n if n else 0.0
    return {
        "lm_positive": pos,
        "lm_negative": neg,
        "lm_polarity": round(score.get("Polarity", 0.0), 4),
        "lm_subjectivity": round(score.get("Subjectivity", 0.0), 4),
        "lm_net_sentiment": round(net, 5),
        "lm_tokens": n,
    }

# ── Corpus loading ─────────────────────────────────────────────────────────
def latest_csv(pattern, explicit=None):
    if explicit and os.path.exists(explicit):
        return explicit
    matches = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    return matches[-1] if matches else None

def load_corpus(guardian_path, gnews_path):
    """
    Build a unified article frame [source, title, url, date, text] from the two
    news CSVs. 'text' concatenates the available textual fields for each source.
    """
    rows = []
    if guardian_path and os.path.exists(guardian_path):
        g = pd.read_csv(guardian_path, dtype=str).fillna("")
        for _, r in g.iterrows():
            rows.append({
                "source": "Guardian",
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "date": r.get("date", ""),
                "text": " ".join([r.get("title", ""), r.get("trail_text", ""),
                                  r.get("body_text", "")]).strip(),
            })
        print(f"  Loaded {len(g)} Guardian articles")
    if gnews_path and os.path.exists(gnews_path):
        n = pd.read_csv(gnews_path, dtype=str).fillna("")
        for _, r in n.iterrows():
            rows.append({
                "source": "Google News",
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "date": (r.get("published", "") or "")[:10],
                "text": " ".join([r.get("title", ""),
                                  r.get("description", "")]).strip(),
            })
        print(f"  Loaded {len(n)} Google News articles")
    return pd.DataFrame(rows)

# ── Main ───────────────────────────────────────────────────────────────────
def run_sentiment(guardian_csv=None, gnews_csv=None):
    print("=" * 60)
    print("Loughran-McDonald Financial Sentiment (sector-context signal)")
    print("=" * 60)

    lm = get_lm()
    if lm is None:
        return None

    g = latest_csv("guardian_articles_*.csv", guardian_csv)
    n = latest_csv("google_news_*.csv", gnews_csv)
    corpus = load_corpus(g, n)
    if corpus.empty:
        print("  No news corpus found in data/ — run guardian.py / google_news.py first.")
        return None
    print(f"  Total articles: {len(corpus)}")

    # Score every article.
    scores = corpus["text"].apply(lambda t: score_text(lm, t)).apply(pd.Series)
    out = pd.concat([corpus.drop(columns="text"), scores], axis=1)

    # Per-article CSV.
    art_path = os.path.join(DATA_DIR, f"sentiment_articles_{date.today().isoformat()}.csv")
    try:
        out.to_csv(art_path, index=False)
    except OSError:
        art_path = f"sentiment_articles_{date.today().isoformat()}.csv"
        out.to_csv(art_path, index=False)
    print(f"\nPer-article scores saved to: {art_path}")

    # Sector-level summary.
    print("\n" + "=" * 60)
    print("SECTOR SENTIMENT SUMMARY (relative, dictionary-based)")
    overall_pol = out["lm_polarity"].mean()
    overall_net = out["lm_net_sentiment"].mean()
    pos_share = (out["lm_polarity"] > 0).mean()
    neg_share = (out["lm_polarity"] < 0).mean()
    print(f"  Articles scored:        {len(out)}")
    print(f"  Mean polarity:          {overall_pol:+.3f}  (-1 to +1)")
    print(f"  Mean net sentiment:     {overall_net:+.4f}")
    print(f"  Positive-leaning:       {pos_share:.0%} of articles")
    print(f"  Negative-leaning:       {neg_share:.0%} of articles")

    # By source.
    print("\n  By source (mean polarity):")
    for src, grp in out.groupby("source"):
        print(f"    {src:14} {grp['lm_polarity'].mean():+.3f}  (n={len(grp)})")

    # Monthly time series (the sector-mood signal over time).
    out["month"] = pd.to_datetime(out["date"], errors="coerce").dt.to_period("M")
    monthly = (out.dropna(subset=["month"])
               .groupby("month")
               .agg(mean_polarity=("lm_polarity", "mean"),
                    mean_net=("lm_net_sentiment", "mean"),
                    n=("lm_polarity", "size"))
               .reset_index())
    monthly["month"] = monthly["month"].astype(str)
    ts_path = os.path.join(DATA_DIR, f"sentiment_monthly_{date.today().isoformat()}.csv")
    try:
        monthly.to_csv(ts_path, index=False)
    except OSError:
        ts_path = f"sentiment_monthly_{date.today().isoformat()}.csv"
        monthly.to_csv(ts_path, index=False)
    print(f"\n  Monthly sentiment series saved to: {ts_path}")
    if len(monthly) > 1:
        print("  (Use this series as the Tier-2 sector-sentiment signal over time.)")

    print("\nNOTE: sector-level signal — characterises news mood, NOT individual")
    print("deals. L-M dictionary scores are relative and interpretable; negation")
    print("is not handled (a known dictionary limitation, validated by FinBERT later).")

    return out

if __name__ == "__main__":
    run_sentiment()