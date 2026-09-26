"""Run blocker v2 on the full train or test set, one country at a time (resumable).
 
  python code/business_entity_resolution/src/run_blocking_v2.py smoke  [--n-jobs 4]   (1-2 min check)
  python code/business_entity_resolution/src/run_blocking_v2.py train  [--n-jobs 4]
  python code/business_entity_resolution/src/run_blocking_v2.py test   [--n-jobs 4]
  python code/business_entity_resolution/src/run_blocking_v2.py train --k 6 --name cand3 --countries India
 
Writes work/<split>/cand2_<country>.parquet. On train it also reports recall and the
oracle F0.5 ceiling, compared with blocker v1 (cand_<country>.parquet) when available.
If RAM runs out, use --n-jobs 2.
"""
import argparse
import sys
import time
import pandas as pd
from config import WORK_DIR
from blocking_v2 import block_country_v2
from metric import macro_f05
 
COLS = ["entity_id", "country", "name_n", "addr_n", "business_name"]
 
def report(country, cand, gt, s1_ids, others_info, old_path, prev_path=None):
    found = gt.merge(cand[["s1_id", "other_id"]], on=["s1_id", "other_id"])
    line = (f"  recall {len(found) / len(gt):.4f} | oracle F0.5 {macro_f05(found, gt, s1_ids):.4f} | "
            f"cands/record {len(cand) / cand['other_id'].nunique():.2f}")
    if old_path.exists():
        old = pd.read_parquet(old_path, columns=["s1_id", "other_id"])
        f_old = gt.merge(old, on=["s1_id", "other_id"])
        line += f"  (v1: recall {len(f_old) / len(gt):.4f}, oracle {macro_f05(f_old, gt, s1_ids):.4f})"
    if prev_path is not None and prev_path.exists():
        prev = pd.read_parquet(prev_path, columns=["s1_id", "other_id"])
        f_prev = gt.merge(prev, on=["s1_id", "other_id"])
        line += (f"\n  previous set {prev_path.stem.split('_')[0]}: recall {len(f_prev) / len(gt):.4f}, "
                 f"oracle {macro_f05(f_prev, gt, s1_ids):.4f}, cands/record {len(prev) / prev['other_id'].nunique():.2f}")
    print(line)
    hit = gt.merge(cand[["s1_id", "other_id"]].assign(f=1), on=["s1_id", "other_id"], how="left")["f"].notna()
    non_latin = ~gt["other_id"].map(others_info["business_name"]).map(str.isascii)
    no_addr = gt["other_id"].map(others_info["addr_n"]).eq("")
    print(f"  recall non-Latin names {hit[non_latin.to_numpy()].mean():.4f} | "
          f"missing address {hit[no_addr.to_numpy()].mean():.4f}")
 
def smoke(n_jobs, k):
    """1-2 minute check that parallel blocking works on this machine (writes nothing)."""
    d = WORK_DIR / "train"
    s1 = pd.read_parquet(d / "source1.parquet", columns=COLS).head(20_000)
    others = pd.read_parquet(d / "source2.parquet", columns=COLS).head(60_000)
    others = others[others["country"].isin(set(s1["country"]))]
    t0 = time.time()
    cand = block_country_v2(s1, others, k=k, n_jobs=n_jobs, log=print)
    print(f"SMOKE TEST OK: {len(cand):,} candidates in {time.time() - t0:.0f}s with n_jobs={n_jobs}")
 
def main(split, n_jobs, k, name="cand2", countries=None, compare="cand2"):
    d = WORK_DIR / split
    s1_all = pd.read_parquet(d / "source1.parquet", columns=["entity_id", "country"])
    gt_all = pd.read_parquet(d / "gt_pairs.parquet") if split == "train" else None
    for country in s1_all["country"].unique():
        if countries and country not in countries:
            continue
        path = d / f"{name}_{country}.parquet"
        f = [("country", "==", country)]
        need_report = split == "train"
        if path.exists() and not need_report:
            print(f"{country}: exists, skipping")
            continue
        s1 = pd.read_parquet(d / "source1.parquet", columns=COLS, filters=f)
        others = pd.concat([pd.read_parquet(d / f"{s}.parquet", columns=COLS, filters=f)
                            for s in ["source2", "source3"]], ignore_index=True)
        if not path.exists():
            t0 = time.time()
            print(f"{country}: {len(s1):,} S1, {len(others):,} S2/S3", flush=True)
            cand = block_country_v2(s1, others, k=k, n_jobs=n_jobs,
                                    log=lambda m: print(m, flush=True))
            cand.to_parquet(path, index=False)
            print(f"{country}: {len(cand):,} candidates in {time.time() - t0:.0f}s", flush=True)
        else:
            cand = pd.read_parquet(path)
            print(f"{country}: loaded existing candidates")
        if need_report:
            gt = gt_all[gt_all["s1_id"].isin(set(s1["entity_id"]))]
            prev = d / f"{compare}_{country}.parquet" if compare != name else None
            report(country, cand, gt, s1["entity_id"], others.set_index("entity_id"),
                   d / f"cand_{country}.parquet", prev)
        sys.stdout.flush()
        del s1, others, cand
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["train", "test", "smoke"])
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--name", default="cand2", help="output candidate-set name (files <name>_<country>.parquet)")
    ap.add_argument("--countries", nargs="*", help="only these countries, e.g. --countries India")
    ap.add_argument("--compare", default="cand2", help="candidate set to compare recall against")
    a = ap.parse_args()
    if a.split == "smoke":
        smoke(a.n_jobs, a.k)
    else:
        main(a.split, a.n_jobs, a.k, a.name, a.countries, a.compare)
 