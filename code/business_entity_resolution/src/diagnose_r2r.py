"""Before building record-to-record linking: can it recover the matches blocking missed?

For a sample of true (S1, S2/S3) matches that are NOT in the candidate set, check:
  1. does the missed record have siblings (other S2/S3 copies of the same business)?
  2. is at least one sibling already found by blocking (so it can act as a bridge)?
  3. is the missed record much more similar to that found sibling than to the S1 record?
Records passing all three are 'rescuable': an upper bound on what linking could add.

  python code/business_entity_resolution/src/diagnose_r2r.py [--cand cand2 --sample 30000]
"""
import argparse
import sys
import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz
from config import WORK_DIR

sys.stdout.reconfigure(encoding="utf-8")
COLS = ["entity_id", "country", "business_name", "business_address", "name_n", "addr_n"]

def text(df):
    return (df["name_n"] + " " + df["addr_n"]).str.strip()

def main(cand_name, sample):
    d = WORK_DIR / "train"
    gt_all = pd.read_parquet(d / "gt_pairs.parquet")
    total_gain = 0.0
    for country in ["US", "India"]:
        f = [("country", "==", country)]
        s1 = pd.read_parquet(d / "source1.parquet", columns=COLS, filters=f).set_index("entity_id")
        oth = pd.concat([pd.read_parquet(d / f"{s}.parquet", columns=COLS, filters=f)
                         for s in ["source2", "source3"]]).set_index("entity_id")
        cand = pd.read_parquet(d / f"{cand_name}_{country}.parquet", columns=["other_id", "s1_id"])
        gt = gt_all[gt_all["s1_id"].isin(s1.index)]
        found = gt.merge(cand, on=["s1_id", "other_id"], how="left", indicator=True)
        found["found"] = found["_merge"] == "both"
        missed = found[~found["found"]]
        print(f"\n=== {country}: {len(gt):,} true pairs, {len(missed):,} missed by blocking "
              f"({len(missed) / len(gt):.1%}) ===")

        m = missed.sample(min(sample, len(missed)), random_state=0)[["s1_id", "other_id"]]
        found_sib = found.loc[found["found"], ["s1_id", "other_id"]].rename(columns={"other_id": "sib_id"})
        all_sib = gt.rename(columns={"other_id": "sib_id"})
        n_sib = m.merge(all_sib, on="s1_id").query("sib_id != other_id").groupby("other_id").size()
        pairs = m.merge(found_sib, on="s1_id")                     # missed record x found siblings
        has_sib = m["other_id"].isin(n_sib.index).mean()
        has_found = m["other_id"].isin(pairs["other_id"]).mean()

        t_o = text(oth)
        q_txt = t_o.reindex(pairs["other_id"]).fillna("").to_numpy()
        sib_txt = t_o.reindex(pairs["sib_id"]).fillna("").to_numpy()
        s1_txt = text(s1).reindex(pairs["s1_id"]).fillna("").to_numpy()
        pairs["sim_sib"] = process.cpdist(q_txt, sib_txt, scorer=fuzz.token_set_ratio, workers=-1)
        pairs["sim_s1"] = process.cpdist(q_txt, s1_txt, scorer=fuzz.token_set_ratio, workers=-1)
        best = pairs.sort_values("sim_sib", ascending=False).drop_duplicates("other_id")
        rescuable = best[(best["sim_sib"] >= 75) & (best["sim_sib"] - best["sim_s1"] >= 10)]
        frac = len(rescuable) / len(m)

        print(f"  missed records with any sibling:            {has_sib:.1%}")
        print(f"  ... with a sibling FOUND by blocking:       {has_found:.1%}")
        print(f"  ... and clearly closer to it than to S1:    {frac:.1%}  (= 'rescuable')")
        print(f"  median similarity to best found sibling {best['sim_sib'].median():.0f} vs to S1 {best['sim_s1'].median():.0f}")
        gain = frac * len(missed) / len(gt)
        print(f"  -> upper bound on recall gain: +{gain:.4f} (recall {len(gt) - len(missed):,}/{len(gt):,} "
              f"= {1 - len(missed) / len(gt):.4f} -> {1 - len(missed) / len(gt) + gain:.4f})")
        total_gain += gain * (0.47 if country == "India" else 0.38)
        ex = rescuable.head(5)
        for _, r in ex.iterrows():
            print(f"    S1 : {s1.at[r.s1_id, 'business_name']} | {s1.at[r.s1_id, 'business_address']}")
            print(f"    miss: {oth.at[r.other_id, 'business_name']} | {oth.at[r.other_id, 'business_address']}")
            print(f"    sib : {oth.at[r.sib_id, 'business_name']} | {oth.at[r.sib_id, 'business_address']}")
        del s1, oth, cand
    print(f"\ntest-weighted recall upside (US+India shares): +{total_gain:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", default="cand2")
    ap.add_argument("--sample", type=int, default=30000)
    a = ap.parse_args()
    main(a.cand, a.sample)
