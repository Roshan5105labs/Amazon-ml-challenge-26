"""Baseline v1 (no ML): blocking + a two-parameter decision rule.
 
Rule: each S2/S3 record is assigned to its best S1 candidate (rank 0) if
  dice(best) >= t   and   dice(best) - dice(second best) >= m
One owner at most per S2/S3 record, which the EDA showed is always true.
 
  tune:     python code/business_entity_resolution/src/rule_baseline.py tune
  predict:  python code/business_entity_resolution/src/rule_baseline.py predict
            (slow: full-test blocking; saves candidates to work/test/cand_<country>.parquet)
  decide:   python code/business_entity_resolution/src/rule_baseline.py decide --t 0.55 --m 0
            (fast: re-applies the rule to the saved candidates with new parameters)
  fulltrain: python code/business_entity_resolution/src/rule_baseline.py fulltrain
            (slow: runs the test-scale pipeline on ALL training data and scores it with
             the ground truth; candidates saved to work/train/cand_<country>.parquet,
             so an interrupted run resumes where it stopped)
"""
import argparse
import json
import time
import numpy as np
import pandas as pd
from config import WORK_DIR, ROOT
from blocking import block_all, block_country
from metric import macro_f05
from submission import write_outputs
 
K = 3   # candidates per S2/S3 record fed to the decision rule (= candidate_pairs.tsv)
COLS = ["entity_id", "country", "name_n", "addr_n"]
PARAMS = WORK_DIR / "rule_params.json"
 
def decide(cand: pd.DataFrame, t: float, m: float) -> pd.DataFrame:
    best = cand[cand["rank"] == 0].set_index("other_id")
    second = cand[cand["rank"] == 1].set_index("other_id")["dice"]
    margin = best["dice"] - second.reindex(best.index).fillna(0).to_numpy()
    ok = (best["dice"] >= t) & (margin >= m)
    return best.loc[ok, ["s1_id"]].reset_index()
 
def tune():
    d = WORK_DIR / "dev"
    s1 = pd.read_parquet(d / "source1.parquet")
    others = pd.read_parquet(d / "others.parquet")
    gt = pd.read_parquet(d / "gt_pairs.parquet")
    t0 = time.time()
    cand = block_all(s1, others, k=K)
    print(f"dev blocking: {len(cand):,} pairs in {time.time() - t0:.0f}s")
 
    # evaluate on the validation split only (queries = val S2/S3, entities = val S1)
    val_s1 = s1.loc[s1["split"] == "val", "entity_id"]
    val_q = set(others.loc[others["split"] == "val", "entity_id"])
    val_gt = gt[gt["split"] == "val"]
    cval = cand[cand["other_id"].isin(val_q)]
 
    results = []
    for t in np.arange(0.20, 0.95, 0.05):
        for m in [0.0, 0.02, 0.05, 0.1, 0.15, 0.2]:
            pred = decide(cval, t, m)
            pred = pred[pred["s1_id"].isin(val_s1)]
            results.append((macro_f05(pred, val_gt, val_s1), round(t, 2), m))
    results.sort(reverse=True)
    bt, bm = results[0][1], results[0][2]
    owned_rate = pd.Series(sorted(val_q)).isin(val_gt["other_id"]).mean()
    for t_chk in sorted({bt, 0.5, 0.55, 0.6, 0.7}):
        rate = len(decide(cval, t_chk, bm)) / len(val_q)
        print(f"dev val: t={t_chk:.2f} assigns {rate:.3f} of S2/S3 records (truly owned: {owned_rate:.3f})")
    print("\ntop settings (val F0.5, t, m):")
    for r in results[:8]:
        print(f"  {r[0]:.4f}  t={r[1]}  m={r[2]}")
    best = {"t": float(results[0][1]), "m": float(results[0][2]), "val_f05": round(results[0][0], 4)}
    PARAMS.write_text(json.dumps(best))
    print(f"\nsaved {best} -> {PARAMS}")
 
def predict():
    p = json.loads(PARAMS.read_text())
    tdir = WORK_DIR / "test"
    s1_all = pd.read_parquet(tdir / "source1.parquet", columns=["entity_id", "country"])
    matches, cands = [], []
    for country in s1_all["country"].unique():          # data-driven: includes France
        t0 = time.time()
        f = [("country", "==", country)]
        s1 = pd.read_parquet(tdir / "source1.parquet", columns=COLS, filters=f)
        others = pd.concat([pd.read_parquet(tdir / f"{s}.parquet", columns=COLS, filters=f)
                            for s in ["source2", "source3"]], ignore_index=True)
        cand = block_country(s1, others, k=K)
        cand.to_parquet(tdir / f"cand_{country}.parquet", index=False)
        matches.append(decide(cand, p["t"], p["m"]))
        cands.append(cand[["s1_id", "other_id"]])
        print(f"{country}: {len(s1):,} S1, {len(others):,} S2/S3 -> {len(cand):,} candidates, "
              f"{len(matches[-1]):,} matches in {time.time() - t0:.0f}s")
        del s1, others, cand
    write_outputs(s1_all["entity_id"], pd.concat(matches), pd.concat(cands), ROOT / "output")
 
def fulltrain():
    """Test-scale evaluation: same blocking settings as predict, but on the full labelled
    training set. Answers: does the dev-slice score survive full-country density?"""
    tdir = WORK_DIR / "train"
    s1_all = pd.read_parquet(tdir / "source1.parquet", columns=["entity_id", "country"])
    gt_all = pd.read_parquet(tdir / "gt_pairs.parquet")
    owner_country = gt_all["s1_id"].map(s1_all.set_index("entity_id")["country"])
    grid = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8]
    per_country = {}
    for country in s1_all["country"].unique():
        path = tdir / f"cand_{country}.parquet"
        s1_ids = s1_all.loc[s1_all["country"] == country, "entity_id"]
        if not path.exists():
            t0 = time.time()
            f = [("country", "==", country)]
            s1 = pd.read_parquet(tdir / "source1.parquet", columns=COLS, filters=f)
            others = pd.concat([pd.read_parquet(tdir / f"{s}.parquet", columns=COLS, filters=f)
                                for s in ["source2", "source3"]], ignore_index=True)
            block_country(s1, others, k=K).to_parquet(path, index=False)
            print(f"{country}: blocked {len(others):,} records in {time.time() - t0:.0f}s")
            del s1, others
        cand = pd.read_parquet(path)
        gt = gt_all[owner_country == country]
        found = gt.merge(cand[["s1_id", "other_id"]], on=["s1_id", "other_id"])
        n_q = cand["other_id"].nunique()
        owned = cand["other_id"].drop_duplicates().isin(gt["other_id"]).mean()
        print(f"\n{country}: pair recall@{K} {len(found) / len(gt):.4f} | "
              f"oracle F0.5 {macro_f05(found, gt, s1_ids):.4f} | truly owned {owned:.3f}")
        scores = []
        for t in grid:
            pred = decide(cand, t, 0.0)
            scores.append(macro_f05(pred, gt, s1_ids))
            print(f"  t={t:.2f}: F0.5 {scores[-1]:.4f} | assigns {len(pred) / n_q:.3f}")
        per_country[country] = (len(s1_ids), scores)
        del cand, found
    n = sum(v[0] for v in per_country.values())
    print("\nALL COUNTRIES (weighted by #S1):")
    for i, t in enumerate(grid):
        print(f"  t={t:.2f}: F0.5 {sum(v[0] * v[1][i] for v in per_country.values()) / n:.4f}")
 
def redecide(t: float, m: float):
    """Re-apply the rule to candidates saved by predict (seconds instead of an hour)."""
    tdir = WORK_DIR / "test"
    s1_all = pd.read_parquet(tdir / "source1.parquet", columns=["entity_id", "country"])
    matches, cands = [], []
    for country in s1_all["country"].unique():
        cand = pd.read_parquet(tdir / f"cand_{country}.parquet")
        matches.append(decide(cand, t, m))
        cands.append(cand[["s1_id", "other_id"]])
        n_q = cand["other_id"].nunique()
        print(f"{country}: assigns {len(matches[-1]):,} records "
              f"({len(matches[-1]) / n_q:.3f} of those with candidates)")
    write_outputs(s1_all["entity_id"], pd.concat(matches), pd.concat(cands), ROOT / "output")
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tune", "predict", "decide", "fulltrain"])
    ap.add_argument("--t", type=float, default=0.55)
    ap.add_argument("--m", type=float, default=0.0)
    a = ap.parse_args()
    if a.mode == "tune":
        tune()
    elif a.mode == "predict":
        predict()
    elif a.mode == "fulltrain":
        fulltrain()
    else:
        redecide(a.t, a.m)
 