"""
One-off correction: remove the stale FTSE-subsidiary flag from Frontier
Development Capital (parent Mercia Asset Management is AIM-listed, NOT FTSE 350).
The mid-market filter scope is explicitly FTSE-350-only, so this flag was a
stale error from before that scope was locked. Sets is_ftse_subsidiary=False
and clears ftse_parent for that one company only.
"""
import glob, os, pandas as pd

f = sorted(glob.glob("data/candidates_annotated_*.csv"))[-1]
df = pd.read_csv(f, dtype=str)

mask = df["company_name"].str.contains("FRONTIER DEVELOPMENT CAPITAL", case=False, na=False)
n = int(mask.sum())
print(f"Rows matching Frontier Development Capital: {n}")
if n == 0:
    print("Nothing to change."); raise SystemExit

# show before
print("BEFORE:")
print(df.loc[mask, ["company_name","is_ftse_subsidiary","ftse_parent"]].to_string(index=False))

df.loc[mask, "is_ftse_subsidiary"] = "False"
df.loc[mask, "ftse_parent"] = ""

# backup then save
bak = f.replace(".csv", "_prefix.csv")
if not os.path.exists(bak):
    pd.read_csv(f, dtype=str).to_csv(bak, index=False)
    print(f"Backup written: {bak}")
df.to_csv(f, index=False)
print(f"Updated: {f}")

# verify
chk = pd.read_csv(f, dtype=str)
ftse = (chk["is_ftse_subsidiary"].astype(str).str.lower()=="true").sum()
total = chk["company_name"].nunique()
print(f"\nAFTER: FTSE-flagged = {ftse} | total unique = {total} | in-scope = {total-ftse}")
print("FTSE-flagged companies now:")
print(chk[chk['is_ftse_subsidiary'].astype(str).str.lower()=='true'][['company_name','ftse_parent']].to_string(index=False))
