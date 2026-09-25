"""
EDA for Amazon ML Challenge 2026 - Business Entity Resolution.
Run from the project root:  python scratch/eda.py
Paste the full printed output back for analysis.
"""
import sys
from pathlib import Path
from collections import Counter
import pandas as pd

# Windows consoles default to cp1252 and crash on accented French text; force UTF-8
sys.stdout.reconfigure(encoding="utf-8")

# Locate the dataset relative to this file: <project>/scratch/eda.py -> <project>/student_resource/dataset
D = Path(__file__).resolve().parent.parent / "student_resource" / "dataset"
if not (D / "train").exists():
    sys.exit(f"Dataset not found at {D}. Check the folder structure inside student_resource/.")

def load(path):
    # dtype=str + keep_default_na=False: empty ID lists stay "" instead of NaN
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

tr = {k: load(D / "train" / f"train_{k}.tsv") for k in ["source1", "source2", "source3"]}
te = {k: load(D / "test" / f"test_{k}.tsv") for k in ["source1", "source2", "source3"]}
gt = load(D / "train" / "train_ground_truth.tsv")

print("=== 1. FILE SIZES & COUNTRIES ===")
for split, dfs in [("train", tr), ("test", te)]:
    for k, df in dfs.items():
        print(f"{split}/{k}: {len(df)} rows | countries: {dict(df['country'].value_counts())}")
print(f"ground_truth: {len(gt)} rows")

print("\n=== 2. MISSING / EMPTY FIELDS (fraction) ===")
for k, df in tr.items():
    print(k, {c: round((df[c].str.strip() == "").mean(), 3) for c in ["business_name", "business_address", "country"]})

print("\n=== 3. GROUND TRUTH STRUCTURE ===")
gt["ids"] = gt["matched_entity_ids"].apply(lambda s: [x for x in s.split(",") if x])
gt["n"] = gt["ids"].apply(len)
print("S1 entities covered by GT:", gt["source1_entity_id"].nunique(), "of", len(tr["source1"]))
print("Singleton rate (no matches):", round((gt["n"] == 0).mean(), 4))
print("Matches-per-entity distribution:", dict(sorted(Counter(gt["n"]).items())))
all_ids = [i for ids in gt["ids"] for i in ids]
print("Matched from S2:", sum(i.startswith("S2-") for i in all_ids), "| from S3:", sum(i.startswith("S3-") for i in all_ids))

# Is each S2/S3 record assigned to at most one S1? (decides whether 1-to-1 assignment is safe)
dup = {i: c for i, c in Counter(all_ids).items() if c > 1}
print("S2/S3 ids matched to >1 S1 entity:", len(dup), "(0 means one-to-one assignment is safe)")

# Per S1: how many S2 vs S3 matches (can one S1 have several S2 records?)
gt["n_s2"] = gt["ids"].apply(lambda l: sum(x.startswith("S2-") for x in l))
gt["n_s3"] = gt["ids"].apply(lambda l: sum(x.startswith("S3-") for x in l))
print("Per-S1 count of S2 matches:", dict(sorted(Counter(gt["n_s2"]).items())))
print("Per-S1 count of S3 matches:", dict(sorted(Counter(gt["n_s3"]).items())))

matched = set(all_ids)
for k in ["source2", "source3"]:
    frac = tr[k]["entity_id"].isin(matched).mean()
    print(f"{k}: fraction of records that match some S1 = {round(frac, 4)}")

print("\n=== 4. COUNTRY CONSISTENCY ACROSS MATCHES ===")
country = pd.concat([tr[k][["entity_id", "country"]] for k in tr]).set_index("entity_id")["country"]
pairs = [(s1, m) for s1, ids in zip(gt["source1_entity_id"], gt["ids"]) for m in ids]
same = sum(country.get(a) == country.get(b) for a, b in pairs)
print(f"Matched pairs with same country: {same}/{len(pairs)}  (if 100%, blocking by country is safe)")

print("\n=== 5. SINGLETON RATE BY COUNTRY ===")
gt["country"] = gt["source1_entity_id"].map(country)
for c, g in gt.groupby("country"):
    print(f"{c}: n={len(g)}, singleton_rate={round((g['n'] == 0).mean(), 4)}")

print("\n=== 6. SAMPLE MATCHED RECORDS (4 per country) ===")
rec = pd.concat([tr[k] for k in tr]).set_index("entity_id")
for c in gt["country"].dropna().unique():
    sub = gt[(gt["country"] == c) & (gt["n"] > 0)].head(4)
    for _, row in sub.iterrows():
        s1 = rec.loc[row.source1_entity_id]
        print(f"\n[{c}] S1 {row.source1_entity_id}: {s1['business_name']} | {s1['business_address']}")
        for m in row.ids:
            print(f"      -> {m}: {rec.loc[m, 'business_name']} | {rec.loc[m, 'business_address']}")

print("\n=== 7. SAMPLE SINGLETONS (3) ===")
for sid in gt[gt["n"] == 0]["source1_entity_id"].head(3):
    print(f"{sid}: {rec.loc[sid, 'business_name']} | {rec.loc[sid, 'business_address']} | {rec.loc[sid, 'country']}")

print("\n=== 8. TEST: NEW-COUNTRY SAMPLES ===")
train_countries = set(pd.concat([tr[k]["country"] for k in tr]))
for k, df in te.items():
    new = df[~df["country"].isin(train_countries)]
    print(f"{k}: countries not seen in train = {dict(new['country'].value_counts())}")
    print(new.head(3).to_string(index=False))
