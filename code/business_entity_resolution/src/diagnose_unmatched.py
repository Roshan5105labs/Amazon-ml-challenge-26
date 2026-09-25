"""Diagnostic: what do the S2/S3 records with NO owner in Source 1 look like?
Run:  python code/business_entity_resolution/src/diagnose_unmatched.py
"""
import sys
import pandas as pd
from config import WORK_DIR

sys.stdout.reconfigure(encoding="utf-8")
gt = pd.read_parquet(WORK_DIR / "train" / "gt_pairs.parquet")
s1 = pd.read_parquet(WORK_DIR / "train" / "source1.parquet", columns=["name_n", "business_address"])
s1_names = set(s1["name_n"])

def last(a):
    return a.str.rsplit(",", n=1).str[-1].str.strip().str.lower()

df = pd.read_parquet(WORK_DIR / "train" / "source2.parquet")
df["owned"] = df["entity_id"].isin(gt["other_id"])
own, un = df[df["owned"]], df[~df["owned"]]
print(f"source2: {len(own):,} owned | {len(un):,} unmatched\n")

for label, g in [("OWNED", own), ("UNMATCHED", un)]:
    print(f"=== {label} ===")
    print("country:", g["country"].value_counts(normalize=True).round(3).to_dict())
    print("empty address:", round((g["addr_n"] == "").mean(), 3))
    print("mean #commas in address:", round(g["business_address"].str.count(",").mean(), 2))
    print("name exactly equals some S1 name:", round(g["name_n"].isin(s1_names).mean(), 3))
    print("name shared with another record in this group:", round(g["name_n"].duplicated(keep=False).mean(), 3))
    print("top 12 last address components:", last(g["business_address"]).value_counts().head(12).to_dict())
    print()

print("S1 top 12 last address components:", last(s1["business_address"]).value_counts().head(12).to_dict())
print("\n=== 15 random UNMATCHED records ===")
print(un.sample(15, random_state=0)[["entity_id", "business_name", "business_address", "country"]].to_string(index=False))
