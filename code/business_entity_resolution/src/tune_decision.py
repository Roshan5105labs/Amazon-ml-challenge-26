"""Tune the F0.5-aware decision layer on the held-out validation entities.

Uses the cached validation features and the saved model (no retraining). Validation
probabilities are cached in <model dir>/val_pred.parquet for fast re-tuning.

  python code/business_entity_resolution/src/tune_decision.py [--cand cand2]
"""
import argparse
import json
import time
import pandas as pd
import lightgbm as lgb
from config import WORK_DIR
from metric import macro_f05
from train_model import model_dir, decide
from decision import decide_ef

def val_predictions(mdir, cfg):
    cache = mdir / "val_pred.parquet"
    if cache.exists() and cache.stat().st_mtime > (mdir / "lgbm.txt").stat().st_mtime:
        return pd.read_parquet(cache)
    model = lgb.Booster(model_file=str(mdir / "lgbm.txt"))
    feats = cfg["features"]
    parts = []
    for country in ["US", "India"]:
        tag = f"{country}_tr{cfg['train_frac']}_va{cfg['val_frac']}_v2_{mdir.name.replace('model_', '')}"
        v = pd.read_parquet(mdir / f"feat_val_{tag}.parquet", columns=feats + ["other_id", "s1_id"])
        v["p"] = model.predict(v[feats])
        parts.append(v[["other_id", "s1_id", "p"]].assign(country=country))
        del v
    pred = pd.concat(parts, ignore_index=True)
    pred.to_parquet(cache, index=False)
    return pred

def main(cand):
    t0 = time.time()
    mdir = model_dir(cand)
    cfg = json.loads((mdir / "config.json").read_text())
    pred = val_predictions(mdir, cfg)
    vs1 = pd.concat([pd.read_parquet(f).assign(country=f.name.split("_")[2])
                     for f in sorted(mdir.glob("val_s1_*.parquet"))], ignore_index=True)
    gt = pd.read_parquet(WORK_DIR / "train" / "gt_pairs.parquet")
    gt = gt[gt["s1_id"].isin(set(vs1["s1_id"]))]
    print(f"{len(pred):,} validation pairs ready in {time.time() - t0:.0f}s")

    base = macro_f05(decide(pred, cfg["tau"]), gt, vs1["s1_id"])
    print(f"\ncurrent rule (tau={cfg['tau']} per record): {base:.4f}")
    res = []
    for tau_min in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        for beta in [0.0, 0.25, 0.5, 1.0, 2.0]:
            res.append((macro_f05(decide_ef(pred, tau_min, beta), gt, vs1["s1_id"]), tau_min, beta))
    res.sort(reverse=True)
    print("best F0.5-aware settings (val F0.5, tau_min, beta):")
    for r in res[:6]:
        print(f"  {r[0]:.4f}  tau_min={r[1]}  beta={r[2]}")
    f, tau_min, beta = res[0]
    print(f"\ngain over current rule: {f - base:+.4f}")
    for c in ["US", "India"]:
        ids = vs1.loc[vs1["country"] == c, "s1_id"]
        pc = pred[pred["country"] == c]
        print(f"  {c}: rule {macro_f05(decide(pc, cfg['tau']), gt, ids):.4f} -> "
              f"F0.5-aware {macro_f05(decide_ef(pc, tau_min, beta), gt, ids):.4f}")
    (mdir / "decision.json").write_text(json.dumps(
        {"tau_min": tau_min, "beta": beta, "val_f05": round(f, 4), "rule_val_f05": round(base, 4)}, indent=1))
    print(f"saved -> {mdir / 'decision.json'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", default="cand2")
    main(ap.parse_args().cand)
