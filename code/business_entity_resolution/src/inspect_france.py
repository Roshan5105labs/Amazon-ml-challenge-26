"""France inspection (no labels): compare France's behaviour with the US on the test set,
look for vocabulary mismatches between sources, and print samples to read by eye.

  python code/business_entity_resolution/src/inspect_france.py [--cand cand3]
"""
import argparse
import json
import sys
from collections import Counter
import pandas as pd
from config import WORK_DIR
from train_model import model_dir, decide

sys.stdout.reconfigure(encoding="utf-8")
COLS = ["entity_id", "business_name", "business_address", "name_n", "addr_n"]

def load_country(tdir, country):
    f = [("country", "==", country)]
    s1 = pd.read_parquet(tdir / "source1.parquet", columns=COLS, filters=f)
    oth = pd.concat([pd.read_parquet(tdir / f"{s}.parquet", columns=COLS, filters=f).assign(src=s)
                     for s in ["source2", "source3"]], ignore_index=True)
    return s1, oth

def behaviour(tdir, cand_name, country, tau):
    s1, oth = load_country(tdir, country)
    scored = pd.read_parquet(tdir / f"scored_{cand_name}_{country}.parquet")
    best = scored.sort_values("p", ascending=False).drop_duplicates("other_id")
    m = decide(scored, tau)
    per_s1 = m.groupby("s1_id").size().reindex(s1["entity_id"], fill_value=0)
    bins = pd.cut(best["p"], [0, 0.1, 0.5, 0.7, 0.9, 1.0], include_lowest=True).value_counts(normalize=True).sort_index()
    return {
        "records with no candidate": 1 - oth["entity_id"].isin(best["other_id"]).mean(),
        "records assigned": len(m) / len(oth),
        "S1 predicted empty": (per_s1 == 0).mean(),
        "matches per S1": per_s1.mean(),
        **{f"best p in {k}": v for k, v in bins.items()},
    }, s1, oth, scored, best

def top_tokens(series, n=25):
    c = Counter(t for x in series for t in x.split())
    total = sum(c.values())
    return {t: round(v / total, 4) for t, v in c.most_common(n)}

def show(rows, s1, oth, title, n):
    print(f"\n--- {title} ---")
    s1i, oi = s1.set_index("entity_id"), oth.set_index("entity_id")
    for _, r in rows.head(n).iterrows():
        print(f"  p={r.p:.2f}  S1 : {s1i.at[r.s1_id, 'business_name']} | {s1i.at[r.s1_id, 'business_address']}")
        print(f"          {r.other_id[:2]} : {oi.at[r.other_id, 'business_name']} | {oi.at[r.other_id, 'business_address']}")

def main(cand_name):
    tdir = WORK_DIR / "test"
    cfg = json.loads((model_dir(cand_name) / "config.json").read_text())
    us, *_ = behaviour(tdir, cand_name, "US", cfg["tau"])
    fr, s1, oth, scored, best = behaviour(tdir, cand_name, "France", 0.9)

    print("=== 1. BEHAVIOUR ON TEST: France vs US ===")
    print(f"{'':32s}{'US':>10s}{'France':>10s}")
    for k in us:
        print(f"{k:32s}{us[k]:>10.3f}{fr.get(k, 0):>10.3f}")
    print("(training data for reference: ~5.6% of S1 have no match, ~3.5 matches per S1)")

    print("\n=== 2. VOCABULARY: most frequent address / name words by source (France) ===")
    for field in ["addr_n", "name_n"]:
        a = top_tokens(s1[field]); b = top_tokens(oth[field])
        only_s1 = [t for t in a if t not in b]
        only_o = [t for t in b if t not in a]
        print(f"\n{field}  Source 1 top words: {list(a)}")
        print(f"{field}  S2/S3    top words: {list(b)}")
        print(f"  frequent in S1 but not S2/S3: {only_s1}")
        print(f"  frequent in S2/S3 but not S1: {only_o}")

    print("\n=== 3. SAMPLES (France) ===")
    b = best.sample(frac=1.0, random_state=0)
    show(b[(b["p"] >= 0.9) & (b["p"] < 0.95)], s1, oth, "accepted, borderline (0.90-0.95)", 8)
    show(b[(b["p"] >= 0.6) & (b["p"] < 0.9)], s1, oth, "REJECTED but would pass US threshold (0.6-0.9)", 12)
    show(b[(b["p"] >= 0.2) & (b["p"] < 0.6)], s1, oth, "rejected, uncertain (0.2-0.6)", 8)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", default="cand3")
    main(ap.parse_args().cand)
