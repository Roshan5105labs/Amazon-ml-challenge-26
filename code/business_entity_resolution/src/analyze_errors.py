"""Where are the remaining F0.5 points lost? (held-out validation entities, full density)

Attribution (each line = score if ONE kind of error were fixed perfectly):
  - no false matches        : remove every wrong predicted pair
  - no rejected true matches: add true pairs the model saw (in candidates) but did not assign
  - no blocking misses      : add true pairs that never reached the candidate set
False matches are split into traps (record has no owner at all) and confusions (record
belongs to another business). Also breaks errors down by record type and prints samples.

  python code/business_entity_resolution/src/analyze_errors.py [--cand cand3]
"""
import argparse
import json
import sys
import pandas as pd
from config import WORK_DIR
from metric import macro_f05
from train_model import model_dir, decide
from tune_decision import val_predictions

sys.stdout.reconfigure(encoding="utf-8")

def main(cand):
    mdir = model_dir(cand)
    cfg = json.loads((mdir / "config.json").read_text())
    pred = val_predictions(mdir, cfg)
    vs1 = pd.concat([pd.read_parquet(f).assign(country=f.name.split("_")[2])
                     for f in sorted(mdir.glob("val_s1_*.parquet"))], ignore_index=True)
    gt_all = pd.read_parquet(WORK_DIR / "train" / "gt_pairs.parquet")
    owner = gt_all.set_index("other_id")["s1_id"]
    gt = gt_all[gt_all["s1_id"].isin(set(vs1["s1_id"]))]
    tau = cfg["tau"]
    m = decide(pred, tau)
    m = m[m["s1_id"].isin(set(vs1["s1_id"]))]
    ids = vs1["s1_id"]

    tagged = m.merge(gt.assign(ok=1), on=["s1_id", "other_id"], how="left")
    wrong = tagged[tagged["ok"].isna()].copy()
    right = tagged[tagged["ok"].notna()][["s1_id", "other_id"]]
    wrong["kind"] = wrong["other_id"].map(owner).isna().map({True: "trap (no owner)", False: "confusion (other owner)"})

    in_cand = gt.merge(pred[["s1_id", "other_id"]].drop_duplicates(), on=["s1_id", "other_id"])
    rejected = in_cand.merge(m, on=["s1_id", "other_id"], how="left", indicator=True)
    rejected = rejected[rejected["_merge"] == "left_only"][["s1_id", "other_id"]]

    base = macro_f05(m, gt, ids)
    no_fp = macro_f05(right, gt, ids)
    no_rej = macro_f05(pd.concat([m, rejected]), gt, ids)
    ceiling = macro_f05(in_cand, gt, ids)
    print(f"=== LOSS ATTRIBUTION ({cand}, tau={tau}) ===")
    print(f"  current                     {base:.4f}")
    print(f"  if no false matches         {no_fp:.4f}   (+{no_fp - base:.4f})   <- precision work")
    print(f"  if no rejected true matches {no_rej:.4f}   (+{no_rej - base:.4f})   <- model/threshold work")
    print(f"  perfect decisions (oracle)  {ceiling:.4f}   (+{ceiling - base:.4f})")
    print(f"  remaining loss from blocking misses: {1 - ceiling:.4f}   <- candidate work")
    print(f"\nfalse matches: {len(wrong):,} of {len(m):,} predicted pairs ({len(wrong) / len(m):.2%})")
    print("  " + wrong["kind"].value_counts().to_string().replace("\n", "\n  "))
    print(f"rejected true matches (in candidates, not assigned): {len(rejected):,}")

    # record-type breakdown
    need = set(wrong["other_id"]) | set(rejected["other_id"]) | set(m["other_id"])
    recs = pd.concat([pd.read_parquet(WORK_DIR / "train" / f"{s}.parquet",
                                      columns=["entity_id", "business_name", "business_address", "addr_n"])
                      for s in ["source2", "source3"]])
    recs = recs[recs["entity_id"].isin(need)].set_index("entity_id")
    def kind(ids_):
        r = recs.reindex(ids_)
        return pd.Series(pd.NA, index=r.index).mask(r["addr_n"].eq(""), "no address") \
                 .fillna(r["business_name"].map(lambda x: "non-Latin name" if isinstance(x, str) and not x.isascii() else "other"))
    print("\nby record type        predicted  false-match rate  rejected-true")
    t_pred = kind(m["other_id"]).value_counts()
    t_wrong = kind(wrong["other_id"]).value_counts()
    t_rej = kind(rejected["other_id"]).value_counts()
    for k in t_pred.index:
        print(f"  {k:18s} {t_pred[k]:>10,}  {t_wrong.get(k, 0) / t_pred[k]:>14.2%}  {t_rej.get(k, 0):>12,}")

    s1 = pd.read_parquet(WORK_DIR / "train" / "source1.parquet", columns=["entity_id", "business_name", "business_address"])
    s1 = s1[s1["entity_id"].isin(set(wrong["s1_id"]) | set(rejected["s1_id"]))].set_index("entity_id")
    p = pred.set_index(["s1_id", "other_id"])["p"]
    for title, df in [("FALSE MATCHES (confusions)", wrong[wrong["kind"].str.startswith("conf")]),
                      ("FALSE MATCHES (traps)", wrong[wrong["kind"].str.startswith("trap")]),
                      ("REJECTED TRUE MATCHES", rejected)]:
        print(f"\n--- {title} ---")
        for _, r in df.sample(min(6, len(df)), random_state=1).iterrows():
            print(f"  p={p.get((r.s1_id, r.other_id), float('nan')):.2f} S1 : {s1.at[r.s1_id, 'business_name']} | {s1.at[r.s1_id, 'business_address']}")
            print(f"         rec: {recs.at[r.other_id, 'business_name']} | {recs.at[r.other_id, 'business_address']}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", default="cand3")
    main(ap.parse_args().cand)
