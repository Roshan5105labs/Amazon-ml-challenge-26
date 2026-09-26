"""Model diagnostics: overfitting, underfitting, country bias. Uses cached features only.

  python code/business_entity_resolution/src/diagnose_model.py [--cand cand2] [--parts fit,curve,loco]

1. fit   - saved model on its own training pairs vs held-out pairs (overfitting check)
2. curve - retrain on 10% / 30% / 100% of the training pairs (is more data or better
           features the bottleneck?)
3. loco  - leave-one-country-out: train on US only -> score India, and vice versa.
           Simulates an unseen country, our best offline proxy for France.

Parts 2-3 train with faster settings (higher learning rate) than the real model, so
compare their numbers with each other, not with the production model's score.
"""
import argparse
import json
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss, roc_auc_score
from config import WORK_DIR
from metric import macro_f05
from train_model import model_dir, decide

TAUS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
FAST = dict(objective="binary", learning_rate=0.15, num_leaves=127, min_data_in_leaf=100,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
            verbose=-1, seed=42)

def load(cand):
    mdir = model_dir(cand)
    cfg = json.loads((mdir / "config.json").read_text())
    feats = cfg["features"]
    cols = feats + ["other_id", "s1_id", "label", "country"]
    tr, va, vs1 = [], [], []
    for country in ["US", "India"]:
        tag = f"{country}_tr{cfg['train_frac']}_va{cfg['val_frac']}_v2_{cand}"
        tr.append(pd.read_parquet(mdir / f"feat_train_{tag}.parquet", columns=cols))
        va.append(pd.read_parquet(mdir / f"feat_val_{tag}.parquet", columns=cols))
        vs1.append(pd.read_parquet(mdir / f"val_s1_{tag}.parquet").assign(country=country))
    gt = pd.read_parquet(WORK_DIR / "train" / "gt_pairs.parquet")
    vs1 = pd.concat(vs1, ignore_index=True)
    gt = gt[gt["s1_id"].isin(set(vs1["s1_id"]))]
    return cfg, feats, pd.concat(tr, ignore_index=True), pd.concat(va, ignore_index=True), vs1, gt

def record_accuracy(df, tau):
    """Share of S2/S3 records decided as well as possible: pick the true owner when it is
    among the candidates, abstain when it is not."""
    best = df.sort_values("p", ascending=False).drop_duplicates("other_id")
    has_true = df.groupby("other_id")["label"].max().reindex(best["other_id"]).to_numpy()
    assign = best["p"].to_numpy() >= tau
    ok = np.where(assign, best["label"].to_numpy() == 1, has_true == 0)
    return ok.mean()

def f05_by_country(val, vs1, gt, countries=("US", "India")):
    out = {}
    for c in countries:
        ids = vs1.loc[vs1["country"] == c, "s1_id"]
        v = val[val["country"] == c]
        scores = [(macro_f05(decide(v, t), gt, ids), t) for t in TAUS]
        out[c] = max(scores)
    return out

def fit_fast(train, val, feats):
    dtr = lgb.Dataset(train[feats], train["label"])
    dva = lgb.Dataset(val[feats], val["label"], reference=dtr)
    return lgb.train(FAST, dtr, 1500, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(30, verbose=False)])

def part_fit(cfg, feats, train, val, cand):
    print("\n=== 1. OVERFITTING: saved model on training vs held-out pairs ===")
    model = lgb.Booster(model_file=str(model_dir(cand) / "lgbm.txt"))
    tau = cfg["tau"]
    for name, d in [("train", train), ("held-out", val)]:
        d["p"] = model.predict(d[feats])          # in place: no copy of a 10M+ row table
        print(f"  {name:9s} logloss {log_loss(d['label'], d['p']):.4f} | AUC {roc_auc_score(d['label'], d['p']):.5f} "
              f"| records decided correctly {record_accuracy(d, tau):.4f}")
    print("  -> small gaps = not overfitting; a large train advantage = overfitting")

def part_curve(feats, train, val, vs1, gt):
    print("\n=== 2. LEARNING CURVE: more data vs better features ===")
    q = train["other_id"].drop_duplicates().sample(frac=1.0, random_state=0).to_numpy()
    for frac in [0.1, 0.3, 1.0]:
        t0 = time.time()
        keep = set(q[: int(len(q) * frac)])
        m = fit_fast(train[train["other_id"].isin(keep)], val, feats)
        val["p"] = m.predict(val[feats], num_iteration=m.best_iteration)
        res = f05_by_country(val, vs1, gt)
        overall = max((macro_f05(decide(val, t), gt, vs1["s1_id"]), t) for t in TAUS)
        print(f"  {frac:>4.0%} of training pairs: val F0.5 {overall[0]:.4f} (US {res['US'][0]:.4f}, "
              f"India {res['India'][0]:.4f}) | {m.best_iteration} rounds, {time.time() - t0:.0f}s")
    print("  -> still rising at 100% = more data helps; flat = feature-limited (need new information)")

def part_loco(feats, train, val, vs1, gt):
    print("\n=== 3. LEAVE-ONE-COUNTRY-OUT: how much does an unseen country cost? ===")
    both = fit_fast(train, val, feats)
    val["p"] = both.predict(val[feats], num_iteration=both.best_iteration)
    ref = f05_by_country(val, vs1, gt)
    for held, seen in [("India", "US"), ("US", "India")]:
        t0 = time.time()
        m = fit_fast(train[train["country"] == seen], val[val["country"] == seen], feats)
        mask = (val["country"] == held).to_numpy()
        val.loc[mask, "p"] = m.predict(val.loc[mask, feats], num_iteration=m.best_iteration)
        f, tau = f05_by_country(val, vs1, gt, countries=(held,))[held]
        print(f"  train {seen:5s} only -> {held:5s}: F0.5 {f:.4f} at tau={tau} | trained on both: "
              f"{ref[held][0]:.4f} at tau={ref[held][1]} | cost of unseen country {ref[held][0] - f:+.4f} "
              f"({time.time() - t0:.0f}s)")
    print("  -> small cost = model transfers well (good sign for France); large cost = France needs work")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", default="cand2")
    ap.add_argument("--parts", default="fit,curve,loco")
    a = ap.parse_args()
    t0 = time.time()
    cfg, feats, train, val, vs1, gt = load(a.cand)
    print(f"loaded {len(train):,} training / {len(val):,} held-out pairs in {time.time() - t0:.0f}s")
    parts = a.parts.split(",")
    if "fit" in parts:
        part_fit(cfg, feats, train, val, a.cand)
    if "curve" in parts:
        part_curve(feats, train, val, vs1, gt)
    if "loco" in parts:
        part_loco(feats, train, val, vs1, gt)
