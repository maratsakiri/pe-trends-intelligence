"""
LDA Elbow Analysis + k=3/5/7 Topic Comparison
=============================================

Supervisor-requested (Athina): examine topic count using the ELBOW METHOD
alongside coherence optimisation, and inspect k=3, 5, 7 explicitly.

Produces:
  1. An elbow table/plot of perplexity AND coherence vs k (k=2..10), so the
     "elbow" (where added topics stop improving fit) is visible.
  2. The explicit topic word-sets for k=3, k=5, k=7 side by side, so the
     broad-vs-fragmented tradeoff can be discussed.

Honest note for the write-up: reducing k makes topics BROADER, not more
specific. The genericness stems from corpus homogeneity, not a mis-set k.
This analysis is expected to CONFIRM that — topics stay broad/entity-driven at
every k — which supports the thematic-homogeneity finding rather than fixing it.

Same sklearn/UMass setup as lda_topics.py. No new dependencies.
Outputs: data/lda_elbow_<date>.csv, data/lda_topics_k3_k5_k7_<date>.csv,
         data/lda_elbow_<date>.png (if matplotlib available).
"""
import os, re, glob, math
import numpy as np, pandas as pd
from datetime import date
from itertools import combinations

DATA_DIR = "data"
K_RANGE = range(2, 11)
K_INSPECT = [3, 5, 7]

# Reuse the same stopword design as lda_topics.py (incl. cleaning sets) so the
# comparison is apples-to-apples with the main run.
EXTRA_STOPWORDS = {
    "said","says","say","would","could","also","one","two","new","year","years",
    "company","companies","firm","firms","business","market","markets","uk",
    "british","britain","london","deal","deals","private","equity","fund","funds",
    "financial","finance","services","service","group","billion","million",
    "percent","per","cent","reuters","guardian","according","told","first","last",
}
URL_JUNK = {"cbmi","https","http","www","html","amp","rss","articles","lxte"}
OFFTOPIC_ENTITIES = {"thames","water","yorkshire","odey","trump","texas","dallas",
                     "sexual","lawyers","housing","hospital"}

def latest_csv(pattern):
    m = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    return m[-1] if m else None

def load_texts():
    texts = []
    g = latest_csv("guardian_articles_*.csv")
    if g:
        d = pd.read_csv(g, dtype=str).fillna("")
        for _, r in d.iterrows():
            texts.append(" ".join([r.get("title",""), r.get("trail_text",""), r.get("body_text","")]))
        print(f"  Loaded {len(d)} Guardian articles")
    n = latest_csv("google_news_*.csv")
    if n:
        d = pd.read_csv(n, dtype=str).fillna("")
        for _, r in d.iterrows():
            texts.append(" ".join([r.get("title",""), r.get("description","")]))
        print(f"  Loaded {len(d)} Google News articles")
    return texts

def get_stopwords():
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return set(ENGLISH_STOP_WORDS) | EXTRA_STOPWORDS | URL_JUNK | OFFTOPIC_ENTITIES

def umass(topic_idx, Xbin, eps=1.0):
    df = np.asarray(Xbin.sum(axis=0)).ravel()
    scores = []
    for terms in topic_idx:
        s, pairs = 0.0, 0
        for wi, wj in combinations(terms, 2):
            co = int(Xbin[:, wi].multiply(Xbin[:, wj]).sum())
            s += math.log((co + eps) / (df[wj] if df[wj] > 0 else 1)); pairs += 1
        if pairs: scores.append(s/pairs)
    return float(np.mean(scores)) if scores else float("nan")

def run():
    print("="*60); print("LDA Elbow Analysis + k=3/5/7 comparison"); print("="*60)
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

    texts = load_texts()
    if not texts:
        print("  No news corpus found."); return
    stop = get_stopwords()
    vec = CountVectorizer(stop_words=list(stop), token_pattern=r"[a-z]{4,}",
                          min_df=2, max_df=0.5, lowercase=True)
    X = vec.fit_transform(texts); vocab = np.array(vec.get_feature_names_out())
    Xbin = (X > 0).astype(int)
    print(f"  Documents: {X.shape[0]} | Vocabulary: {X.shape[1]}")

    # ── Elbow: perplexity + coherence vs k ──
    print("\n  Computing perplexity + coherence across k (elbow analysis)...")
    rows, models = [], {}
    for k in K_RANGE:
        lda = LatentDirichletAllocation(n_components=k, random_state=42,
                                        learning_method="batch", max_iter=25)
        lda.fit(X)
        tops = [list(np.argsort(c)[::-1][:10]) for c in lda.components_]
        coh = umass(tops, Xbin); perp = lda.perplexity(X)
        rows.append({"num_topics": k, "perplexity": round(perp,1),
                     "umass_coherence": round(coh,4)})
        models[k] = (lda, tops)
        print(f"    k={k:2d}  perplexity={perp:8.1f}  UMass={coh:+.4f}")
    elbow = pd.DataFrame(rows)
    ep = os.path.join(DATA_DIR, f"lda_elbow_{date.today().isoformat()}.csv")
    elbow.to_csv(ep, index=False); print(f"\n  Elbow table saved: {ep}")

    # Identify the elbow on perplexity (largest second difference = sharpest bend)
    p = elbow["perplexity"].values
    if len(p) >= 3:
        second_diff = np.diff(p, 2)            # curvature
        elbow_k = int(elbow["num_topics"].iloc[np.argmax(second_diff)+1])
        print(f"  Perplexity elbow (sharpest bend) at k≈{elbow_k}")
    coh_k = int(elbow.loc[elbow['umass_coherence'].idxmax(),'num_topics'])
    print(f"  Coherence-optimal k = {coh_k}")

    # ── k=3/5/7 explicit topic sets ──
    print("\n" + "="*60); print("TOPIC SETS at k = 3, 5, 7 (for comparison)"); print("="*60)
    comp_rows = []
    for k in K_INSPECT:
        if k not in models:
            lda = LatentDirichletAllocation(n_components=k, random_state=42,
                                            learning_method="batch", max_iter=25).fit(X)
            comps = lda.components_
        else:
            comps = models[k][0].components_
        print(f"\n  --- k = {k} ---")
        for tid, c in enumerate(comps):
            words = [vocab[i] for i in np.argsort(c)[::-1][:10]]
            print(f"   Topic {tid}: {', '.join(words)}")
            comp_rows.append({"k": k, "topic_id": tid, "top_terms": "; ".join(words)})
    cp = os.path.join(DATA_DIR, f"lda_topics_k3_k5_k7_{date.today().isoformat()}.csv")
    pd.DataFrame(comp_rows).to_csv(cp, index=False)
    print(f"\n  k=3/5/7 topic sets saved: {cp}")

    # ── Optional elbow plot ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(7,4.2))
        ax1.plot(elbow["num_topics"], elbow["perplexity"], "o-", color="#23395b", label="Perplexity")
        ax1.set_xlabel("Number of topics (k)"); ax1.set_ylabel("Perplexity", color="#23395b")
        ax1.tick_params(axis="y", labelcolor="#23395b")
        ax2 = ax1.twinx()
        ax2.plot(elbow["num_topics"], elbow["umass_coherence"], "s--", color="#b07a16", label="UMass coherence")
        ax2.set_ylabel("UMass coherence", color="#b07a16"); ax2.tick_params(axis="y", labelcolor="#b07a16")
        plt.title("LDA topic-count selection: elbow (perplexity) & coherence")
        fig.tight_layout()
        pp = os.path.join(DATA_DIR, f"lda_elbow_{date.today().isoformat()}.png")
        plt.savefig(pp, dpi=150); print(f"  Elbow plot saved: {pp}")
    except Exception as e:
        print(f"  (matplotlib plot skipped: {e})")

    print("\nHONEST READING: if topics stay broad/entity-driven at k=3,5,7 AND the")
    print("elbow/coherence curves are flat, that CONFIRMS thematic homogeneity —")
    print("the corpus lacks strong structure; reducing k yields broader, not")
    print("sharper, themes. Report this as a finding, not a fix.")

if __name__ == "__main__":
    run()