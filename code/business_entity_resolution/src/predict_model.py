"""Apply the trained model to the saved test candidates and write both output files.
 
Prerequisites: `rule_baseline.py predict` (test candidates) and `train_model.py` (model).
 
  python code/business_entity_resolution/src/predict_model.py                       # tau from training
  python code/business_entity_resolution/src/predict_model.py --tau 0.7             # one threshold for all
  python code/business_entity_resolution/src/predict_model.py --country-tau France=0.8
  python code/business_entity_resolution/src/predict_model.py --cand cand2   # blocker-v2 model
                                                     # per-country override, others keep --tau / trained tau
 
Test probabilities are cached in work/test/scored_<country>.parquet and reused, so changing
thresholds takes seconds. They are recomputed automatically when the model file is newer
than the cache (i.e. after retraining), or when --rescore is given.
"""
import argparse
import json
import time
import pandas as pd
import lightgbm as lgb
from config import WORK_DIR, ROOT
from features import block_features, text_features
from train_model import load_records, decide, model_dir
from submission import write_outputs
 
CHUNK = 3_000_000   # pairs per feature batch, keeps RAM bounded on the 14M-pair India set
 
def main(tau, country_tau, rescore, cand_name):
    MODEL_DIR = model_dir(cand_name)
    cfg = json.loads((MODEL_DIR / "config.json").read_text())
    tau = cfg["tau"] if tau is None else tau
    model_path = MODEL_DIR / "lgbm.txt"
    tdir = WORK_DIR / "test"
    s1_all = pd.read_parquet(tdir / "source1.parquet", columns=["entity_id", "country"])
    matches, cands = [], []
    for country in s1_all["country"].unique():
        scored = tdir / (f"scored_{country}.parquet" if cand_name == "cand"
                         else f"scored_{cand_name}_{country}.parquet")
        cand = pd.read_parquet(tdir / f"{cand_name}_{country}.parquet")
        stale = not scored.exists() or scored.stat().st_mtime < model_path.stat().st_mtime
        if rescore or stale:
            t0 = time.time()
            model = lgb.Booster(model_file=str(model_path))
            s1, others = load_records(tdir, country)
            full = block_features(cand)
            probs = []
            for start in range(0, len(full), CHUNK):
                part = text_features(full.iloc[start:start + CHUNK], s1, others)
                probs.append(pd.Series(model.predict(part[cfg["features"]]), index=part.index))
            full["p"] = pd.concat(probs)
            full[["other_id", "s1_id", "p"]].to_parquet(scored, index=False)
            print(f"{country}: scored {len(full):,} pairs in {time.time() - t0:.0f}s")
            del s1, others, full
        pred = pd.read_parquet(scored)
        t_c = country_tau.get(country, tau)
        m = decide(pred, t_c)
        print(f"{country}: tau={t_c} assigns {len(m):,} of {pred['other_id'].nunique():,} records")
        matches.append(m)
        cands.append(cand[["s1_id", "other_id"]])
    write_outputs(s1_all["entity_id"], pd.concat(matches), pd.concat(cands), ROOT / "output")
 
def parse_country_tau(items):
    out = {}
    for it in items or []:
        name, val = it.split("=")
        out[name] = float(val)
    return out
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--country-tau", nargs="*", help="e.g. France=0.8 India=0.65")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--cand", default="cand", help="candidate set: cand (blocker v1) or cand2 (v2)")
    a = ap.parse_args()
    main(a.tau, parse_country_tau(a.country_tau), a.rescore, a.cand)
 
 