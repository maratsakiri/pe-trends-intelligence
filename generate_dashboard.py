"""
PE Trends Intelligence — Dashboard Generator (Option 2)
=======================================================
Reads the real pipeline CSVs and writes ONE self-contained HTML file that can
be shared by link/email — no server, no dependencies to open it.

Run from the project root:
    python generate_dashboard.py

Reads (newest dated file of each; falls back gracefully):
    data/tier1_scores_*.csv          candidate scores + components
    data/pe_acquisitions_*.csv       owner, detection date, SIC
    data/candidates_annotated_*.csv  FTSE flag
    data/fca_validation_*.csv        FCA authorised status
    data/google_trends_momentum_*.csv  sector search momentum
    data/sentiment_articles_*.csv    L-M net sentiment

Writes: PE_Pipeline_Dashboard_LIVE.html
"""
import glob, os, json, html
import pandas as pd
from datetime import date, datetime

def newest(pat):
    fs = sorted(glob.glob(pat))
    return fs[-1] if fs else None

def load(pat):
    f = newest(pat)
    if not f:
        print(f"  [warn] no file for {pat}")
        return None, None
    return pd.read_csv(f, dtype=str).fillna(""), f

# ── Load ───────────────────────────────────────────────────────────────────
tier1, f1 = load("data/tier1_scores_*.csv")
acq,   f2 = load("data/pe_acquisitions_*.csv")
ann,   f3 = load("data/candidates_annotated_*.csv")
fca,   f4 = load("data/fca_validation_*.csv")
trends,f5 = load("data/google_trends_momentum_*.csv")
senti, f6 = load("data/sentiment_articles_*.csv")
lda3, f7  = load("data/lda_topics_k3_k5_k7_*.csv")
reed, f8  = load("data/reed_jobs_*.csv")

if tier1 is None or acq is None:
    raise SystemExit("Cannot build: tier1_scores and pe_acquisitions are required.")

# ── Candidate table: join tier1 + acquisition detail ───────────────────────
acq_u = acq.drop_duplicates("company_name").set_index("company_name")
def owner(c):  return acq_u["pe_owner_name"].get(c, "")
def since(c):  return acq_u["pe_ownership_since"].get(c, "")

# FCA authorised set
fca_yes = set()
if fca is not None:
    col = "fca_authorised" if "fca_authorised" in fca.columns else None
    if col:
        fca_yes = set(fca[fca[col].str.lower().isin(["true","1","yes","y"])]["company_name"])

# FTSE flag
ftse_set = set()
if ann is not None and "is_ftse_subsidiary" in ann.columns:
    ftse_set = set(ann[ann["is_ftse_subsidiary"].str.lower()=="true"]["company_name"])

tier1["score"] = pd.to_numeric(tier1["tier1_detection_confidence"], errors="coerce").fillna(0)
tier1 = tier1.drop_duplicates("company_name").sort_values("score", ascending=False)

# confidence band from real distribution
def band(s):
    if s >= 0.70: return "High"
    if s >= 0.55: return "Medium"
    return "Low"

def fmt_date(s):
    for f in ("%Y-%m-%d","%d/%m/%Y","%Y-%m-%dT%H:%M:%S"):
        try: return datetime.strptime(str(s)[:10], f).strftime("%d %b %Y")
        except: pass
    return str(s) or "—"

# recency: new if detected within 30 days of newest detection in the set
dates = pd.to_datetime(acq["pe_ownership_since"], errors="coerce")
newest_det = dates.max()
def is_new(c):
    d = pd.to_datetime(since(c), errors="coerce")
    return bool(pd.notna(d) and pd.notna(newest_det) and (newest_det - d).days <= 30)

TOPN = 15
rows = []
for _, r in tier1.head(TOPN).iterrows():
    c = r["company_name"]
    rows.append({
        "company": c, "owner": owner(c) or "—", "date": fmt_date(since(c)),
        "score": round(float(r["score"]),3), "band": band(float(r["score"])),
        "new": is_new(c),
        "psc": True,  # every candidate came from a PSC detection
        "fca": c in fca_yes, "ftse": c in ftse_set,
    })

active = tier1["company_name"].nunique() - len(ftse_set)  # in-scope after FTSE
new_count = sum(1 for x in rows if x["new"])
fca_count = len(fca_yes)

# ── Sentiment: mean polarity across corpus (matches Chapter 4 headline) ─────
net_sent = -0.051
if senti is not None and "lm_polarity" in senti.columns:
    v = pd.to_numeric(senti["lm_polarity"], errors="coerce").mean()
    if pd.notna(v): net_sent = round(float(v),3)

# ── Trends (tier by batch) ─────────────────────────────────────────────────
batch_tier = {"ai":"ai","ai_detail":"finai","transformation":"sector",
              "fs_sector":"sector","pe":"sector"}
def tier_of(batch, term):
    t = str(term).lower()
    if term in ("AI","ChatGPT"): return "ai"
    if any(k in t for k in ["generative ai","ai in finance","ai banking","ai automation","machine learning"]): return "finai"
    if any(k in t for k in ["m&a","leveraged buyout"]): return "jargon"
    return batch_tier.get(str(batch).lower(), "sector")

trend_rows = []
if trends is not None:
    tr = trends.copy()
    tr["mom"] = pd.to_numeric(tr["trend_recent_minus_early"], errors="coerce").fillna(0)
    tr = tr.sort_values("mom", ascending=False)
    for _, r in tr.iterrows():
        trend_rows.append([r["term"], round(float(r["mom"]),1), tier_of(r["batch"], r["term"])])
maxmom = max([t[1] for t in trend_rows], default=1) or 1


# ── LDA k=3 topics: human label + real top terms ───────────────────────────
LDA_LABELS = ["Deal & finance activity", "Wealth & banking", "Listed-company market news"]
lda_topics = []
if lda3 is not None and {"k","topic_id","top_terms"}.issubset(lda3.columns):
    k3 = lda3[lda3["k"].astype(str)=="3"].sort_values("topic_id")
    for i,(_,r) in enumerate(k3.iterrows()):
        label = LDA_LABELS[i] if i < len(LDA_LABELS) else f"Topic {r['topic_id']}"
        terms = str(r["top_terms"]).replace(";", ",")
        lda_topics.append({"label": label, "terms": terms})
if not lda_topics:  # fallback to labels only
    lda_topics = [{"label": l, "terms": ""} for l in LDA_LABELS]

# ── Reed sector job titles (a few examples, transformation emphasised) ──────
job_titles = []
if reed is not None and "job_title" in reed.columns:
    seen=set()
    for t in reed["job_title"].dropna():
        t=str(t).strip()
        if t and t.lower() not in seen:
            seen.add(t.lower()); job_titles.append(t)
    # prioritise transformation-flavoured titles, keep 6
    key=lambda x: (0 if any(w in x.lower() for w in ["transformation","data","digital","analyt","engineer"]) else 1, x)
    job_titles = sorted(job_titles, key=key)[:6]
job_count = len(reed) if reed is not None else 45

DATA = json.dumps({"cands":rows,"trends":trend_rows,"lda":lda_topics,"jobs":job_titles,"jobcount":job_count})

# ── HTML (same design as approved mockup) ──────────────────────────────────
TEMPLATE = open("_dashboard_template.html").read()
out = (TEMPLATE
       .replace("__ACTIVE__", str(active))
       .replace("__NEW__", str(new_count))
       .replace("__FCA__", str(fca_count))
       .replace("__NETSENT__", f"{net_sent:+.3f}".replace("+","").replace("-","&minus;") if net_sent<0 else f"{net_sent:.3f}")
       .replace("__MAXMOM__", str(maxmom))
       .replace("__GENDATE__", date.today().strftime("%d %b %Y"))
       .replace("__DATA__", DATA))
open("PE_Pipeline_Dashboard_LIVE.html","w",encoding="utf-8").write(out)
print(f"  Built PE_Pipeline_Dashboard_LIVE.html")
print(f"  Candidates: {len(rows)} shown (top {TOPN}) | Active in-scope: {active} | New: {new_count} | FCA: {fca_count}")
print(f"  Net sentiment: {net_sent} | Trend terms: {len(trend_rows)}")
