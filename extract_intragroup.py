"""
Extract intra-group exclusions from companies_house.log into a review sheet
for manual verification (Part A of validation).

Each logged exclusion looks like:
  ... Excluded intra-group: <COMPANY> <- <OWNER> (shared root '<word>')

Produces intragroup_review.csv with one row per exclusion and blank columns
for your manual verdict.
"""
import re, csv, glob, os

log = "companies_house.log"
if not os.path.exists(log):
    print(f"  {log} not found in this folder. Run from the project root "
          f"(where companies_house.log lives).")
    raise SystemExit

rows = []
pat = re.compile(r"Excluded intra-group:\s*(.+?)\s*<-\s*(.+?)\s*\(shared root '(.+?)'\)")
with open(log, encoding="utf-8", errors="replace") as f:
    for line in f:
        m = pat.search(line)
        if m:
            rows.append({
                "company": m.group(1).strip(),
                "excluded_owner": m.group(2).strip(),
                "shared_word": m.group(3).strip(),
                "manual_verdict": "",      # genuine_intragroup / wrongly_excluded / unclear
                "notes": "",
            })

# de-duplicate (the log may have repeats across runs)
seen = set(); uniq = []
for r in rows:
    key = (r["company"], r["excluded_owner"])
    if key not in seen:
        seen.add(key); uniq.append(r)

out = "intragroup_review.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(uniq[0].keys()) if uniq else
                       ["company","excluded_owner","shared_word","manual_verdict","notes"])
    w.writeheader(); w.writerows(uniq)
print(f"  Extracted {len(uniq)} unique intra-group exclusions -> {out}")
print("  Fill 'manual_verdict' with: genuine_intragroup / wrongly_excluded / unclear")
