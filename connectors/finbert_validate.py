"""
FinBERT Validation — robustness check on Loughran-McDonald sentiment.

ROLE (lit review 2.3.3): FinBERT is a VALIDATION LAYER ONLY, not a primary
method. This module cross-checks the L-M dictionary sentiment using a
transformer model (FinBERT; Araci 2019 / ProsusAI), focused specifically on
testing the KEY L-M FINDING: that Guardian PE coverage is more negative
per-word than Google News.

Two things are reported, which can diverge:
  (1) PER-ARTICLE agreement — do FinBERT and L-M score the same article
      similarly? (Pearson + Spearman correlation, sign-agreement rate)
  (2) SOURCE-LEVEL finding — does FinBERT ALSO rank Guardian more negative than
      Google News? This is what actually validates the L-M result; it can hold
      even if per-article agreement is modest.

SCALE NOTE: L-M polarity is (pos-neg)/(pos+neg) in [-1,1]. FinBERT gives a
label (positive/negative/neutral) + confidence; mapped to a signed score
(positive=+conf, negative=-conf, neutral=0) so both live on [-1,1]. This
mapping is a deliberate choice and a limitation of the comparison.

Model: ProsusAI/finbert (~440MB, downloads once from HuggingFace on first run).
CPU is fine for this sample size (no GPU needed).

Install (prebuilt wheels exist for Python 3.14 — no compiler needed):
    pip install torch transformers --break-system-packages
"""

import os
import glob
import logging
import pandas as pd
from datetime import date

logging.basicConfig(
    filename="finbert.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

DATA_DIR = "data"
MODEL_NAME = "ProsusAI/finbert"

# ── Loading the already-scored L-M articles ────────────────────────────────
def latest_csv(pattern, explicit=None):
    if explicit and os.path.exists(explicit):
        return explicit
    matches = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    return matches[-1] if matches else None

def load_lm_scored(path):
    """Load the per-article L-M sentiment CSV (output of sentiment_lm.py).
    Must contain: source, title, lm_polarity (and ideally url/date)."""
    df = pd.read_csv(path, dtype=str).fillna("")
    if "lm_polarity" not in df.columns or "source" not in df.columns:
        raise ValueError(f"{path} missing lm_polarity/source — run sentiment_lm.py first")
    df["lm_polarity"] = pd.to_numeric(df["lm_polarity"], errors="coerce").fillna(0.0)
    if "lm_net_sentiment" in df.columns:
        df["lm_net_sentiment"] = pd.to_numeric(df["lm_net_sentiment"],
                                               errors="coerce").fillna(0.0)
    return df

# ── FinBERT ────────────────────────────────────────────────────────────────
def get_finbert():
    """Return a FinBERT sentiment pipeline, or None if unavailable."""
    try:
        from transformers import (AutoTokenizer,
                                  AutoModelForSequenceClassification,
                                  pipeline)
    except ImportError:
        print("  transformers not installed. Run: "
              "pip install torch transformers --break-system-packages")
        return None
    try:
        print(f"  Loading {MODEL_NAME} (downloads ~440MB on first run)...")
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        # truncation so long Guardian bodies fit the 512-token limit
        return pipeline("sentiment-analysis", model=mdl, tokenizer=tok,
                        truncation=True, max_length=512)
    except Exception as e:
        print(f"  Failed to load FinBERT: {e}")
        logging.error(f"FinBERT load failed: {e}")
        return None

def finbert_signed_score(result):
    """Map a FinBERT result {label, score} to a signed score in [-1,1]:
    positive=+score, negative=-score, neutral=0."""
    label = result.get("label", "").lower()
    score = float(result.get("score", 0.0))
    if label == "positive":
        return score, "positive"
    if label == "negative":
        return -score, "negative"
    return 0.0, "neutral"

# ── Main ───────────────────────────────────────────────────────────────────
def run_finbert_validation(lm_csv=None, text_field="title"):
    print("=" * 60)
    print("FinBERT Validation — cross-check of L-M sentiment")
    print("Focus: Guardian vs Google News divergence")
    print("=" * 60)

    path = latest_csv("sentiment_articles_*.csv", lm_csv)
    if not path:
        print("  No L-M sentiment CSV found — run sentiment_lm.py first.")
        return None
    print(f"  L-M scores: {path}")
    df = load_lm_scored(path)

    # We need article text to feed FinBERT. The L-M article CSV keeps 'title'
    # (and url/date) but not full body. Title is a fair, length-controlled unit
    # for cross-source comparison (avoids the Guardian-length confound!). Use
    # title by default; note this in interpretation.
    if text_field not in df.columns:
        text_field = "title"
    df["_text"] = df[text_field].astype(str)
    df = df[df["_text"].str.strip().astype(bool)].copy()
    print(f"  Articles to score: {len(df)} (text field: '{text_field}')")
    print("  NOTE: scoring TITLES — equal-length units across sources, which")
    print("  controls for the Guardian full-text length confound by design.")

    fb = get_finbert()
    if fb is None:
        return None

    # Score each article.
    print("\n  Scoring with FinBERT (CPU; a few minutes)...")
    signed, labels, confs = [], [], []
    for i, txt in enumerate(df["_text"].tolist(), 1):
        try:
            res = fb(txt[:2000])[0]
            s, lab = finbert_signed_score(res)
        except Exception as e:
            logging.warning(f"FinBERT failed on row {i}: {e}")
            s, lab, res = 0.0, "error", {"score": 0.0}
        signed.append(round(s, 4))
        labels.append(lab)
        confs.append(round(float(res.get("score", 0.0)), 4))
        if i % 25 == 0:
            print(f"    {i}/{len(df)}")

    df["finbert_signed"] = signed
    df["finbert_label"] = labels
    df["finbert_confidence"] = confs

    # ── (1) Per-article agreement ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("(1) PER-ARTICLE AGREEMENT (FinBERT vs L-M)")
    valid = df[df["finbert_label"] != "error"]
    pear = valid["lm_polarity"].corr(valid["finbert_signed"], method="pearson")
    spear = valid["lm_polarity"].corr(valid["finbert_signed"], method="spearman")
    # sign agreement (treating |x|<0.01 as neutral)
    def sign(x): return 0 if abs(x) < 0.01 else (1 if x > 0 else -1)
    agree = (valid["lm_polarity"].map(sign) ==
             valid["finbert_signed"].map(sign)).mean()
    print(f"  Pearson r:        {pear:+.3f}")
    print(f"  Spearman rho:     {spear:+.3f}")
    print(f"  Sign agreement:   {agree:.0%} of articles")

    # ── (2) Source-level finding (the actual validation) ───────────────────
    print("\n" + "=" * 60)
    print("(2) SOURCE-LEVEL: does FinBERT also find Guardian more negative?")
    rows = []
    for src, g in valid.groupby("source"):
        rows.append({
            "source": src, "n": len(g),
            "lm_mean_polarity": round(g["lm_polarity"].mean(), 3),
            "finbert_mean_signed": round(g["finbert_signed"].mean(), 3),
            "finbert_pct_negative": round((g["finbert_label"] == "negative").mean(), 3),
        })
    src_df = pd.DataFrame(rows)
    print(src_df.to_string(index=False))

    # Verdict on the divergence finding.
    if len(src_df) == 2:
        g_row = src_df[src_df["source"].str.contains("Guardian", case=False)]
        n_row = src_df[~src_df["source"].str.contains("Guardian", case=False)]
        if not g_row.empty and not n_row.empty:
            lm_gap = g_row["lm_mean_polarity"].iat[0] - n_row["lm_mean_polarity"].iat[0]
            fb_gap = g_row["finbert_mean_signed"].iat[0] - n_row["finbert_mean_signed"].iat[0]
            print(f"\n  L-M     Guardian-minus-other polarity gap: {lm_gap:+.3f}")
            print(f"  FinBERT Guardian-minus-other signed gap:   {fb_gap:+.3f}")
            if lm_gap < 0 and fb_gap < 0:
                print("  → BOTH methods rank Guardian more negative — finding VALIDATED.")
            elif lm_gap < 0 and fb_gap >= 0:
                print("  → Methods DISAGREE on direction — L-M finding NOT confirmed by")
                print("    FinBERT; likely where dictionary lacks context (report this).")
            else:
                print("  → Mixed/!inconclusive — interpret with care.")

    # Save.
    out = os.path.join(DATA_DIR, f"finbert_validation_{date.today().isoformat()}.csv")
    keep = [c for c in ["source", "title", "date", "url", "lm_polarity",
                        "lm_net_sentiment", "finbert_signed", "finbert_label",
                        "finbert_confidence"] if c in df.columns]
    try:
        df[keep].to_csv(out, index=False)
    except OSError:
        out = f"finbert_validation_{date.today().isoformat()}.csv"
        df[keep].to_csv(out, index=False)
    print(f"\n  Per-article comparison saved to: {out}")

    # Where they diverge most (for the write-up).
    valid = valid.assign(divergence=(valid["lm_polarity"] -
                                     valid["finbert_signed"]).abs())
    top = valid.sort_values("divergence", ascending=False).head(5)
    print("\n  Largest FinBERT vs L-M disagreements (context-sensitivity cases):")
    for _, r in top.iterrows():
        print(f"    L-M {r['lm_polarity']:+.2f} | FinBERT {r['finbert_signed']:+.2f} "
              f"({r['finbert_label']}) :: {str(r['title'])[:60]}")

    print("\nNOTE: FinBERT is a VALIDATION layer only (lit 2.3.3). Titles scored to")
    print("control for length. FinBERT→signed-score mapping is a deliberate choice.")
    return df

if __name__ == "__main__":
    run_finbert_validation()