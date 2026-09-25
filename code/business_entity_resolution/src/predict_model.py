"""Apply the trained model to the saved test candidates and write both output files.

Prerequisites: `rule_baseline.py predict` (test candidates) and `train_model.py` (model).

  python code/business_entity_resolution/src/predict_model.py            # score + write
  python code/business_entity_resolution/src/predict_model.py --tau 0.7  # re-decide only (fast)
"""
import argparse
import json
import time
import pandas as pd
import lightgbm as lgb
from config import WORK_DIR, ROOT
from features import block_features, text_features
from train_model import load_records, decide, MODEL_DIR
from submission import write_outputs

CHUNK = 3_000_000   # pairs per feature batch, keeps RAM bounded on the 14M-pair India set

def main(tau_override):
    cfg = json.loads((MODEL_DIR / "config.json").read_text())
    tau = cfg["tau"] if tau_override is None else tau_override
    tdir = WORK_DIR / "test"
    s1_all = pd.read_parquet(tdir / "source1.parquet", columns=["entity_id", "country"])
    matches, cands = [], []
    for country in s1_all["country"].unique():
        scored = tdir / f"scored_{country}.parquet"
        cand = pd.read_parquet(tdir / f"cand_{country}.parquet")
        if tau_override is None or not scored.exists():
            t0 = time.time()
            model = lgb.Booster(model_file=str(MODEL_DIR / "lgbm.txt"))
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
        m = decide(pred, tau)
        print(f"{country}: tau={tau} assigns {len(m):,} of {pred['other_id'].nunique():,} records")
        matches.append(m)
        cands.append(cand[["s1_id", "other_id"]])
    write_outputs(s1_all["entity_id"], pd.concat(matches), pd.concat(cands), ROOT / "output")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=None)
    main(ap.parse_args().tau)
