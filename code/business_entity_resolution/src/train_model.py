"""Train the LightGBM matching model on full-scale training candidates.

Prerequisite: `rule_baseline.py fulltrain` has written work/train/cand_<country>.parquet
(the same blocking, at the same scale, as the test run).

Validation design (v2, fixes an optimistic bias):
- hold out a fraction of Source 1 entities (val S1)
- EVALUATION records = every S2/S3 record that has ANY val S1 among its candidates or is
  owned by one, whoever owns it. Wrong assignments of other entities' records to a val S1
  are therefore counted as false positives, exactly as on the test set. (v1 only scored
  val-split records, which silently dropped most false positives.)
- TRAINING records are sampled only from records that never touch a val S1, so the model
  is never evaluated on a record it was trained on.
Sanity check printed at the end: the rule baseline on this validation should match the
`rule_baseline.py fulltrain` numbers.

Run:  python code/business_entity_resolution/src/train_model.py [--train-frac 0.25 --val-frac 0.05]

Features are cached per country in work/model/ (file names include the fractions), so a
crash or a retrain with different LightGBM settings reloads them in seconds instead of
recomputing. Delete the feat_*.parquet files to force recomputation after changing features.py.
"""
import argparse
import json
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from config import WORK_DIR
from features import feature_list, block_features, text_features, _to_f32
from metric import macro_f05

REC_COLS = ["entity_id", "country", "business_name", "name_n", "addr_n"]
def model_dir(cand_name: str):
    """v1 candidates keep the original folder; other candidate sets get their own."""
    return WORK_DIR / ("model" if cand_name == "cand" else f"model_{cand_name}")

MODEL_DIR = model_dir("cand")
FEAT_TAG = "v3"   # bump when features.py changes: invalidates cached features automatically

def load_records(split_dir, country):
    f = [("country", "==", country)]
    s1 = pd.read_parquet(split_dir / "source1.parquet", columns=REC_COLS, filters=f)
    others = pd.concat([pd.read_parquet(split_dir / f"{s}.parquet", columns=REC_COLS, filters=f)
                        for s in ["source2", "source3"]], ignore_index=True)
    return s1, others

def bucket(ids) -> np.ndarray:
    """Deterministic 0..999 bucket per id (same result on every run and machine)."""
    return (pd.util.hash_pandas_object(pd.Series(ids), index=False).to_numpy() % 1000).astype(int)

def decide(pred: pd.DataFrame, tau: float) -> pd.DataFrame:  # noqa: D401
    """Each S2/S3 record -> its highest-probability S1, if p >= tau (at most one owner)."""
    best = pred.sort_values("p", ascending=False).drop_duplicates("other_id")
    return best.loc[best["p"] >= tau, ["s1_id", "other_id"]]

def main(train_frac, val_frac, cand_name):
    MODEL_DIR = model_dir(cand_name)
    tdir = WORK_DIR / "train"
    s1_all = pd.read_parquet(tdir / "source1.parquet", columns=["entity_id", "country"])
    gt_all = pd.read_parquet(tdir / "gt_pairs.parquet")
    owner = gt_all.set_index("other_id")["s1_id"]
    s1_country = s1_all.set_index("entity_id")["country"]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_parts, val_parts, val_s1 = [], [], []
    for country in s1_all["country"].unique():
        t0 = time.time()
        tag = f"{country}_tr{train_frac}_va{val_frac}_{FEAT_TAG}_{cand_name}"
        f_tr, f_va, f_ids = (MODEL_DIR / f"feat_train_{tag}.parquet", MODEL_DIR / f"feat_val_{tag}.parquet",
                             MODEL_DIR / f"val_s1_{tag}.parquet")
        if f_tr.exists() and f_va.exists() and f_ids.exists():
            train_parts.append(_to_f32(pd.read_parquet(f_tr)))
            val_parts.append(_to_f32(pd.read_parquet(f_va)))
            val_s1.append(pd.read_parquet(f_ids)["s1_id"])
            print(f"{country}: loaded cached features in {time.time() - t0:.0f}s")
            continue
        cand = block_features(pd.read_parquet(tdir / f"{cand_name}_{country}.parquet"))
        s1, others = load_records(tdir, country)

        # hold out Source 1 entities; evaluate on EVERY record that touches one of them
        s1_val = bucket(s1["entity_id"]) < val_frac * 1000
        val_ids = set(s1.loc[s1_val, "entity_id"])
        touches = cand["s1_id"].isin(val_ids).groupby(cand["other_id"]).any()
        owned_by_val = touches.index.to_series().map(owner).isin(val_ids)
        q_eval = touches | owned_by_val
        q_ids = q_eval.index.to_series()
        q_val = set(q_ids[q_eval.to_numpy()])
        pool = q_ids[~q_eval.to_numpy()]
        q_train_sample = set(pool[bucket(pool.to_numpy() + "#t") < train_frac * 1000])

        tr = cand[cand["other_id"].isin(q_train_sample)]
        va = cand[cand["other_id"].isin(q_val)]
        for part, store, path in [(tr, train_parts, f_tr), (va, val_parts, f_va)]:
            feats = text_features(part, s1, others)
            feats["label"] = (feats["other_id"].map(owner) == feats["s1_id"]).astype(np.int8)
            feats["country"] = country
            feats.to_parquet(path, index=False)
            store.append(feats)
        val_s1.append(pd.Series(sorted(val_ids), name="s1_id"))
        val_s1[-1].to_frame().to_parquet(f_ids, index=False)
        print(f"{country}: {len(tr):,} train pairs, {len(va):,} val pairs "
              f"({tr['other_id'].nunique():,} / {va['other_id'].nunique():,} records) in {time.time() - t0:.0f}s")
        del cand, s1, others

    train, val = pd.concat(train_parts, ignore_index=True), pd.concat(val_parts, ignore_index=True)
    FEATURES = feature_list(train.columns)
    print(f"candidate set '{cand_name}': {len(FEATURES)} features")
    val_s1 = pd.concat(val_s1, ignore_index=True)
    val_gt = gt_all[gt_all["s1_id"].isin(set(val_s1))]
    print(f"\ntrain: {len(train):,} pairs ({train['label'].mean():.3f} positive) | val: {len(val):,} pairs")

    params = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, seed=42)
    dtr = lgb.Dataset(train[FEATURES], train["label"])
    dva = lgb.Dataset(val[FEATURES], val["label"], reference=dtr)
    t0 = time.time()
    model = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    print(f"trained {model.best_iteration} rounds in {time.time() - t0:.0f}s")
    val["p"] = model.predict(val[FEATURES], num_iteration=model.best_iteration)

    # threshold on the competition metric itself, not on logloss / AUC
    print("\nval macro F0.5 by threshold (full-density, held-out S1 entities):")
    res = []
    for tau in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98]:
        res.append((macro_f05(decide(val, tau), val_gt, val_s1), round(float(tau), 2)))
        print(f"  tau={tau:.2f}: {res[-1][0]:.4f}")
    best_f, best_tau = max(res)

    # reference: the rule baseline on exactly the same validation records
    rule = max((macro_f05(decide(val.assign(p=val["dice"]), t), val_gt, val_s1), t)
               for t in [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8])
    print(f"\n(sanity check: rule baseline here should be close to fulltrain's best, ~0.77)")
    print(f"\nLightGBM best: {best_f:.4f} at tau={best_tau} | rule baseline best: {rule[0]:.4f} at t={rule[1]}")
    for c in val["country"].unique():
        ids = val_s1[val_s1.map(s1_country) == c]
        print(f"  {c}: {macro_f05(decide(val[val['country'] == c], best_tau), val_gt, ids):.4f}")

    imp = pd.Series(model.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    print("\ntop features by gain:\n" + (imp / imp.sum()).round(3).head(12).to_string())

    model.save_model(str(MODEL_DIR / "lgbm.txt"), num_iteration=model.best_iteration)
    (MODEL_DIR / "config.json").write_text(json.dumps(
        {"tau": best_tau, "val_f05": round(best_f, 4), "features": FEATURES, "feat_tag": FEAT_TAG,
         "train_frac": train_frac, "val_frac": val_frac, "params": params}, indent=1))
    print(f"\nsaved model + config to {MODEL_DIR}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-frac", type=float, default=0.25)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--cand", default="cand", help="candidate set: cand (blocker v1) or cand2 (v2)")
    a = ap.parse_args()
    main(a.train_frac, a.val_frac, a.cand)
