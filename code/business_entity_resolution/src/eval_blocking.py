"""Step 3: measure blocking quality on the dev slice.
 
Reports, for several k:
- pair recall: share of true (S1, S2/S3) pairs whose S1 is in the record's top-k
- oracle F0.5: the score a PERFECT matcher would get using only these candidates
  (this is the ceiling the matching model can reach)
- candidates per S1 and reduction ratio vs. comparing everything
plus recall broken down by hard cases (non-Latin names, missing addresses).
 
"""
import argparse
import time
import pandas as pd
from config import WORK_DIR
from blocking import block_all
from metric import macro_f05
 
def main(k):
    d = WORK_DIR / "dev"
    s1 = pd.read_parquet(d / "source1.parquet")
    others = pd.read_parquet(d / "others.parquet")
    gt = pd.read_parquet(d / "gt_pairs.parquet")
 
    t = time.time()
    cand = block_all(s1, others, k=k)
    print(f"blocking: {len(cand):,} pairs in {time.time() - t:.0f}s")
    cand.to_parquet(d / f"candidates_k{k}.parquet", index=False)
 
    hit = gt.merge(cand, on=["s1_id", "other_id"], how="left")
    info = others.set_index("entity_id")
    hit["non_latin_name"] = ~hit["other_id"].map(info["business_name"]).map(str.isascii)
    hit["no_address"] = hit["other_id"].map(info["addr_n"]).eq("")
    hit["country"] = hit["s1_id"].map(s1.set_index("entity_id")["country"])
 
    n_pairs = sum(len(s1[s1.country == c]) * len(others[others.country == c]) for c in s1.country.unique())
    print(f"\n{'k':>4} {'pair_recall':>12} {'oracle_F0.5':>12} {'cand/S1':>9} {'reduction':>10}")
    for kk in [x for x in [1, 2, 3, 5, 10, 20, 50] if x <= k]:
        ck = cand[cand["rank"] < kk]
        found = hit["rank"].lt(kk)
        oracle = macro_f05(hit.loc[found, ["s1_id", "other_id"]], gt, s1["entity_id"])
        per_s1 = len(ck) / len(s1)
        print(f"{kk:>4} {found.mean():>12.4f} {oracle:>12.4f} {per_s1:>9.1f} {1 - len(ck) / n_pairs:>10.6f}")
 
    found = hit["rank"].lt(k)
    print(f"\nrecall@{k} breakdown:")
    print(f"  by country:        {found.groupby(hit['country']).mean().round(4).to_dict()}")
    print(f"  non-Latin name:    {found[hit['non_latin_name']].mean():.4f}  (n={hit['non_latin_name'].sum():,})")
    print(f"  missing address:   {found[hit['no_address']].mean():.4f}  (n={hit['no_address'].sum():,})")
    print(f"  S2/S3 with zero candidates: {1 - others['entity_id'].isin(cand['other_id']).mean():.4f}")
 
    miss = hit[~found].head(8)
    print("\nsample misses (S1 record  ->  missed S2/S3 record):")
    s1i = s1.set_index("entity_id")
    for _, m in miss.iterrows():
        print(f"  {s1i.at[m.s1_id, 'business_name']} | {s1i.at[m.s1_id, 'business_address']}")
        print(f"    -> {info.at[m.other_id, 'business_name']} | {info.at[m.other_id, 'business_address']}")
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20)
    a = ap.parse_args()
    main(a.k)
 