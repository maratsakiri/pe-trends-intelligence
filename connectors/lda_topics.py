"""
LDA Topic Modelling — Tier-2 sector-context signal.

Discovers latent themes in the collected NEWS CORPUS (Guardian + Google News)
using Latent Dirichlet Allocation (Blei et al. 2003), implemented with
scikit-learn's LatentDirichletAllocation. The number of topics is chosen by
COHERENCE-SCORE OPTIMISATION (UMass coherence, Mimno et al. 2011), not fixed in
advance.

IMPLEMENTATION NOTE: scikit-learn is used instead of Gensim because Gensim has
no prebuilt wheel for Python 3.14 (it requires compiling C extensions, which
needs the MS C++ build tools). scikit-learn's LDA is the same algorithm
(Blei et al. 2003) and needs no compilation. The dissertation commits to "LDA
with coherence-score optimisation" generically, so this engine choice is
transparent to the methodology.

ROLE (two-tier architecture): SECTOR-LEVEL signal — the corpus is general PE/FS
news, not text about specific detected companies, so topics characterise what
the sector press discusses, not individual deals.

Why LDA over BERTopic (lit 2.3.2): interpretability + parameter control; LDA
best when relevant constructs are known in advance (Garcia-Mendez et al. 2023).

CORPUS CAVEAT: 153 short articles is small for LDA; topics may be unstable or
thematically blurred (the corpus was collected with similar PE/FS queries, so
it is thematically homogeneous). Report corpus size as a limitation.

Coherence metric: UMass (Mimno et al. 2011). UMass scores are <= 0; HIGHER
(closer to 0) = more coherent. The best topic count maximises UMass.

Install (no compiler needed):
    pip install scikit-learn --break-system-packages
"""

import os
import re
import glob
import math
import logging
import numpy as np
import pandas as pd
from datetime import date
from itertools import combinations

logging.basicConfig(
    filename="lda.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

DATA_DIR = "data"
TOPIC_RANGE = range(2, 11)   # try 2..10 topics

EXTRA_STOPWORDS = {
    "said", "says", "say", "would", "could", "also", "one", "two", "new",
    "year", "years", "company", "companies", "firm", "firms", "business",
    "market", "markets", "uk", "british", "britain", "london", "deal", "deals",
    "private", "equity", "fund", "funds", "financial", "finance", "services",
    "service", "group", "billion", "million", "percent", "per", "cent",
    "reuters", "guardian", "according", "told", "first", "last",
}

# Corpus-cleaning stopwords (added after a first LDA run produced entity-driven,
# off-topic topics). TWO categories, kept separate for an auditable rationale:
#
# (1) URL/markup fragments — unambiguous junk leaking from Google News redirect
#     links (e.g. the "CBMi..." base64 article tokens). Always correct to remove.
URL_JUNK = {
    "cbmi", "https", "http", "www", "html", "amp", "oc", "rss", "articles",
}
# (2) DEMONSTRABLY OFF-TOPIC named entities/events that matched the broad PE/FS
#     queries but are NOT private-equity value-creation content. Removing these
#     is defensible because they are unrelated news stories, NOT because removing
#     them makes topics prettier. We deliberately do NOT remove any PE-related
#     entity, even if doing so would sharpen topics — that would be manufacturing
#     the hypothesised result. (Thames/Yorkshire Water = utility scandal; Odey =
#     hedge-fund misconduct story; Trump/Texas/Dallas = US politics/geography.)
OFFTOPIC_ENTITIES = {
    "thames", "water", "yorkshire", "odey", "trump", "texas", "dallas",
    "sexual", "lawyers", "housing", "hospital",
}

# ── Loading ────────────────────────────────────────────────────────────────
def latest_csv(pattern, explicit=None):
    if explicit and os.path.exists(explicit):
        return explicit
    matches = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    return matches[-1] if matches else None

def load_corpus_texts(guardian_path, gnews_path):
    texts = []
    if guardian_path and os.path.exists(guardian_path):
        g = pd.read_csv(guardian_path, dtype=str).fillna("")
        for _, r in g.iterrows():
            texts.append(" ".join([r.get("title", ""), r.get("trail_text", ""),
                                   r.get("body_text", "")]))
        print(f"  Loaded {len(g)} Guardian articles")
    if gnews_path and os.path.exists(gnews_path):
        n = pd.read_csv(gnews_path, dtype=str).fillna("")
        for _, r in n.iterrows():
            texts.append(" ".join([r.get("title", ""), r.get("description", "")]))
        print(f"  Loaded {len(n)} Google News articles")
    return texts

# ── Stopwords ──────────────────────────────────────────────────────────────
def get_stopwords():
    base = set()
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        base = set(ENGLISH_STOP_WORDS)
    except Exception:
        base = {"the","a","an","and","or","but","is","are","was","were","be",
                "to","of","in","on","for","with","as","by","at","from","that",
                "this","it","its","they","them","their","we","our","you","your",
                "not","no","than","then","so","into","over","about","after",
                "before","out","up","down","more","most","some","any","all",
                "can","will","just","has","have","had","which","who","what",
                "when","where","why","how","there","here","because"}
    return base | EXTRA_STOPWORDS | URL_JUNK | OFFTOPIC_ENTITIES

# ── UMass coherence ────────────────────────────────────────────────────────
def umass_coherence(topic_term_indices, doc_term_binary, eps=1.0):
    """
    UMass coherence (Mimno et al. 2011) averaged over topics.
    topic_term_indices: list of lists of term-column-indices (top words/topic).
    doc_term_binary: scipy/np binary doc-term matrix (1 if term in doc).
    Returns mean coherence (<=0; higher is better).
    """
    # Document frequency per term, and co-doc-frequency for pairs on demand.
    df = np.asarray(doc_term_binary.sum(axis=0)).ravel()  # docs per term
    topic_scores = []
    for terms in topic_term_indices:
        score = 0.0
        pairs = 0
        for wi, wj in combinations(terms, 2):
            # co-document frequency of wi, wj
            col_i = doc_term_binary[:, wi]
            col_j = doc_term_binary[:, wj]
            co = int(col_i.multiply(col_j).sum()) if hasattr(col_i, "multiply") \
                else int(np.logical_and(col_i, col_j).sum())
            score += math.log((co + eps) / (df[wj] if df[wj] > 0 else 1))
            pairs += 1
        if pairs:
            topic_scores.append(score / pairs)
    return float(np.mean(topic_scores)) if topic_scores else float("nan")

# ── Main ───────────────────────────────────────────────────────────────────
def run_lda(guardian_csv=None, gnews_csv=None):
    print("=" * 60)
    print("LDA Topic Modelling (sector-context signal) — scikit-learn")
    print("=" * 60)

    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation
    except ImportError:
        print("  scikit-learn not installed. Run: "
              "pip install scikit-learn --break-system-packages")
        return None

    g = latest_csv("guardian_articles_*.csv", guardian_csv)
    n = latest_csv("google_news_*.csv", gnews_csv)
    texts = load_corpus_texts(g, n)
    if not texts:
        print("  No news corpus found — run guardian.py / google_news.py first.")
        return None
    print(f"  Total documents: {len(texts)}")

    stop = get_stopwords()
    # Vectorise: drop terms in <2 docs or >50% of docs, keep alpha tokens >3 chars.
    vec = CountVectorizer(stop_words=list(stop), token_pattern=r"[a-z]{4,}",
                          min_df=2, max_df=0.5, lowercase=True)
    try:
        X = vec.fit_transform(texts)
    except ValueError as e:
        print(f"  Vectorisation failed (corpus too small/homogeneous): {e}")
        return None
    vocab = np.array(vec.get_feature_names_out())
    print(f"  Documents: {X.shape[0]}  |  Vocabulary: {X.shape[1]} terms")
    if X.shape[1] < 10:
        print("  Vocabulary too small for meaningful LDA.")
        return None

    Xbin = (X > 0).astype(int)  # binary doc-term for coherence

    print("\n  Optimising topic count by UMass coherence (higher = better)...")
    results, best = [], None
    for k in TOPIC_RANGE:
        lda = LatentDirichletAllocation(n_components=k, random_state=42,
                                        learning_method="batch", max_iter=25)
        lda.fit(X)
        # top-10 term indices per topic
        tops = [list(np.argsort(comp)[::-1][:10]) for comp in lda.components_]
        coh = umass_coherence(tops, Xbin)
        perp = lda.perplexity(X)
        results.append({"num_topics": k, "umass_coherence": round(coh, 4),
                        "perplexity": round(perp, 1)})
        print(f"    k={k:2d}  UMass={coh:+.4f}  perplexity={perp:.1f}")
        if best is None or coh > best["coh"]:
            best = {"k": k, "coh": coh, "model": lda, "tops": tops}

    print(f"\n  Best topic count: k={best['k']} (UMass={best['coh']:+.4f})")

    coh_df = pd.DataFrame(results)
    coh_path = os.path.join(DATA_DIR, f"lda_coherence_{date.today().isoformat()}.csv")
    try:
        coh_df.to_csv(coh_path, index=False)
    except OSError:
        coh_path = f"lda_coherence_{date.today().isoformat()}.csv"
        coh_df.to_csv(coh_path, index=False)
    print(f"  Coherence curve saved to: {coh_path}")

    # Report topics.
    model = best["model"]
    print("\n" + "=" * 60)
    print(f"DISCOVERED TOPICS (k={best['k']})")
    topic_rows = []
    for tid, comp in enumerate(model.components_):
        idx = np.argsort(comp)[::-1][:10]
        words = ", ".join(vocab[i] for i in idx)
        print(f"\n  Topic {tid}: {words}")
        topic_rows.append({"topic_id": tid,
                           "top_terms": "; ".join(vocab[i] for i in idx)})
    topics_path = os.path.join(DATA_DIR, f"lda_topics_{date.today().isoformat()}.csv")
    try:
        pd.DataFrame(topic_rows).to_csv(topics_path, index=False)
    except OSError:
        topics_path = f"lda_topics_{date.today().isoformat()}.csv"
        pd.DataFrame(topic_rows).to_csv(topics_path, index=False)
    print(f"\n  Topics saved to: {topics_path}")

    # Per-document dominant topic.
    doc_topic = model.transform(X)
    dom = [{"doc_index": i, "dominant_topic": int(row.argmax()),
            "topic_probability": round(float(row.max()), 4)}
           for i, row in enumerate(doc_topic)]
    dom_path = os.path.join(DATA_DIR, f"lda_doc_topics_{date.today().isoformat()}.csv")
    try:
        pd.DataFrame(dom).to_csv(dom_path, index=False)
    except OSError:
        dom_path = f"lda_doc_topics_{date.today().isoformat()}.csv"
        pd.DataFrame(dom).to_csv(dom_path, index=False)
    print(f"  Per-document dominant topics saved to: {dom_path}")

    print("\nNOTE: sector-level topics (what the PE/FS press discusses), NOT per-deal.")
    print(f"CORPUS CAVEAT: {X.shape[0]} short docs is small for LDA — treat topic")
    print("distinctions as indicative; report corpus size as a limitation.")
    print("Coherence: UMass (Mimno et al. 2011); higher (closer to 0) = better.")

    return model

if __name__ == "__main__":
    run_lda()