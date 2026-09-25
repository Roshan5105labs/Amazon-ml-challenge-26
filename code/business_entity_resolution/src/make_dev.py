"""Step 2: build a realistic development slice from the training data.
 
Why a REGIONAL slice instead of a random sample: a random 5% sample thins out every
neighbourhood, so near-duplicate competitors disappear and blocking/matching look far
easier than on the real test set. Taking whole states keeps local density realistic.
 
Unmatched S2/S3 records (no owner in S1, ~26% of the data) are the traps, so they must be
in the slice too. We don't know their state directly, so we learn from matched records
which address components (state names, abbreviations, city names) point almost
exclusively to one state, and use those to place unmatched records. Fully data-driven.
 
Output (WORK_DIR/dev/): source1.parquet, others.parquet (S2+S3), gt_pairs.parquet,
each with a 'split' column (train/val) assigned by Source 1 entity.
 
Run:  python code/business_entity_resolution/src/make_dev.py [--frac 0.05]
"""
import argparse
import numpy as np
import pandas as pd
from config import WORK_DIR, SEED
 
COLS = ["entity_id", "business_name", "business_address", "country", "name_n", "addr_n"]
 
def last_component(addr: pd.Series) -> pd.Series:
    return addr.str.rsplit(",", n=1).str[-1].str.strip().str.lower()
 
def components(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (entity_id, position, lower-cased comma-separated address component)."""
    c = df[["entity_id", "business_address"]].copy()
    c["comp"] = c["business_address"].str.lower().str.split(",")
    c = c.explode("comp")
    c["pos"] = c.groupby(level=0).cumcount()
    c["comp"] = c["comp"].str.strip()
    return c[c["comp"].notna() & (c["comp"] != "")][["entity_id", "pos", "comp"]]
 
def s1_state_keys(s1: pd.DataFrame, min_state_count: int) -> pd.Series:
    """State key per S1 record, robust to reordered addresses.
 
    Source 1 addresses are sometimes reordered ('Rajasthan, Jaipur, ...'), so the last
    component is not always the state. We learn the state vocabulary from the data
    (last components that occur >= min_state_count times per country) and take the
    RIGHTMOST component of each address that belongs to that vocabulary."""
    last = s1["country"] + "|" + last_component(s1["business_address"])
    counts = last.value_counts()
    vocab = set(counts[counts >= min_state_count].index)
    comp = components(s1)
    comp["key"] = comp["entity_id"].map(s1.set_index("entity_id")["country"]) + "|" + comp["comp"]
    comp = comp[comp["key"].isin(vocab)].sort_values("pos")
    rightmost = comp.drop_duplicates("entity_id", keep="last").set_index("entity_id")["key"]
    key = s1["entity_id"].map(rightmost)
    print(f"S1 records with a recognised state: {key.notna().mean():.3f}")
    return key.fillna(last)
 
def main(frac, min_state_count, min_comp_count, purity, val_frac):
    rng = np.random.default_rng(SEED)
    s1 = pd.read_parquet(WORK_DIR / "train" / "source1.parquet", columns=COLS)
    gt = pd.read_parquet(WORK_DIR / "train" / "gt_pairs.parquet")
 
    # 1. choose whole states per country until ~frac of that country's S1 is covered
    s1["key"] = s1_state_keys(s1, min_state_count)
    chosen = set()
    for country, g in s1.groupby("country"):
        counts = g["key"].value_counts()
        counts = counts[counts >= min_state_count]      # ignore malformed 'states'
        keys = counts.index.to_numpy()
        rng.shuffle(keys)
        total = 0
        for k in keys:
            if total >= frac * len(g):
                break
            chosen.add(k)
            total += counts[k]
    print("chosen states:", sorted(chosen))
 
    dev_s1 = s1[s1["key"].isin(chosen)].copy()
    dev_s1["split"] = np.where(rng.random(len(dev_s1)) < val_frac, "val", "train")
    owner_key = s1.set_index("entity_id")["key"]
    split_of = dev_s1.set_index("entity_id")["split"]
    dev_gt = gt[gt["s1_id"].isin(dev_s1["entity_id"])].copy()
    dev_gt["split"] = dev_gt["s1_id"].map(split_of)
    owner_of = gt.set_index("other_id")["s1_id"]
 
    parts = []
    for src in ["source2", "source3"]:
        df = pd.read_parquet(WORK_DIR / "train" / f"{src}.parquet", columns=COLS)
        owned_mask = df["entity_id"].isin(owner_of.index)
 
        # 2a. owned records whose owner is in the slice
        mine = df[df["entity_id"].isin(dev_gt["other_id"])].copy()
        mine["split"] = mine["entity_id"].map(owner_of).map(split_of)
 
        # 2b. learn address component -> state from a sample of owned records
        owned = df[owned_mask]
        sample = owned.sample(min(len(owned), 1_000_000), random_state=SEED)
        comp = components(sample)
        comp["key"] = comp["entity_id"].map(owner_of).map(owner_key)
        cnt = comp.groupby(["comp", "key"]).size().rename("n").reset_index()
        tot = cnt.groupby("comp")["n"].transform("sum")
        cnt = cnt[(tot >= min_comp_count) & (cnt["n"] / tot >= purity)]
        comp_to_key = cnt.set_index("comp")["key"]
 
        # 2c. unmatched records that mention a chosen state/city
        unowned = df[~owned_mask]
        uc = components(unowned)
        uc["key"] = uc["comp"].map(comp_to_key)
        hit_ids = uc.loc[uc["key"].isin(chosen), "entity_id"].unique()
        theirs = unowned[unowned["entity_id"].isin(hit_ids)].copy()
        theirs["split"] = np.where(rng.random(len(theirs)) < val_frac, "val", "train")
 
        found = uc.loc[uc["key"].isin(chosen)].drop_duplicates("entity_id")["key"].value_counts()
        print(f"  {src} unmatched found per chosen state: {found.to_dict()}")
        both = pd.concat([mine, theirs])
        both["source"] = src
        parts.append(both)
        print(f"{src}: {len(mine):,} owned + {len(theirs):,} unmatched")
        del df
 
    others = pd.concat(parts, ignore_index=True)
    out = WORK_DIR / "dev"
    out.mkdir(parents=True, exist_ok=True)
    dev_s1.drop(columns="key").to_parquet(out / "source1.parquet", index=False)
    others.to_parquet(out / "others.parquet", index=False)
    dev_gt.to_parquet(out / "gt_pairs.parquet", index=False)
 
    # realism checks: compare with the full-data numbers from the EDA
    singleton = 1 - dev_s1["entity_id"].isin(dev_gt["s1_id"]).mean()
    unmatched = 1 - others["entity_id"].isin(dev_gt["other_id"]).mean()
    print(f"\nDEV: {len(dev_s1):,} S1 | {len(others):,} S2+S3 | {len(dev_gt):,} true pairs")
    print(dev_s1.groupby(["country", "split"]).size().to_string())
    print(f"singleton rate {singleton:.3f} (full data 0.056) | "
          f"unmatched S2/S3 {unmatched:.3f} (full data ~0.26)")
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--min-state-count", type=int, default=1000)
    ap.add_argument("--min-comp-count", type=int, default=30)
    ap.add_argument("--purity", type=float, default=0.9)
    ap.add_argument("--val-frac", type=float, default=0.2)
    a = ap.parse_args()
    main(a.frac, a.min_state_count, a.min_comp_count, a.purity, a.val_frac)