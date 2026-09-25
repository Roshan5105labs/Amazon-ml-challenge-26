"""Step 1: convert raw TSVs to Parquet with normalized text columns (one file at a time to save RAM).
Run once from the project root:  python code/business_entity_resolution/src/prepare_data.py
"""
import csv
import time
import pandas as pd
from config import DATA_DIR, WORK_DIR, SOURCES
from normalize import normalize_series

def read_tsv(path):
    # QUOTE_NONE: a stray quote character inside a name must not swallow following lines
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)

def main():
    for split in ["train", "test"]:
        out = WORK_DIR / split
        out.mkdir(parents=True, exist_ok=True)
        for src in SOURCES:
            t = time.time()
            df = read_tsv(DATA_DIR / split / f"{split}_{src}.tsv")
            df["name_n"] = normalize_series(df["business_name"])
            df["addr_n"] = normalize_series(df["business_address"])
            df.to_parquet(out / f"{src}.parquet", index=False)
            print(f"{split}/{src}: {len(df):,} rows in {time.time() - t:.0f}s")
            del df

    gt = read_tsv(DATA_DIR / "train" / "train_ground_truth.tsv")
    gt["other_id"] = gt["matched_entity_ids"].str.split(",")
    gt = gt.explode("other_id")
    gt["other_id"] = gt["other_id"].str.strip()
    gt = gt[gt["other_id"].notna() & (gt["other_id"] != "")]
    gt = gt.rename(columns={"source1_entity_id": "s1_id"})[["s1_id", "other_id"]]
    gt.to_parquet(WORK_DIR / "train" / "gt_pairs.parquet", index=False)
    print(f"ground truth: {len(gt):,} matched pairs")

if __name__ == "__main__":
    main()
