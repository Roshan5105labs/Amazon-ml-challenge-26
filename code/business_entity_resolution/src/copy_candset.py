"""Reuse unchanged countries when building a new candidate set.

Copies <src>_<country>.parquet -> <dst>_<country>.parquet (train and test) for the given
countries, and the cached training features for those countries into the new model folder
(renamed to the new tag), so train_model.py does not recompute them.

  python code/business_entity_resolution/src/copy_candset.py --src cand2 --dst cand3 --countries US France
"""
import argparse
import shutil
from config import WORK_DIR

def main(src, dst, countries):
    for split in ["train", "test"]:
        for c in countries:
            f = WORK_DIR / split / f"{src}_{c}.parquet"
            if f.exists():
                shutil.copy2(f, WORK_DIR / split / f"{dst}_{c}.parquet")
                print(f"copied {split}/{f.name} -> {dst}_{c}.parquet")
    m_src, m_dst = WORK_DIR / f"model_{src}", WORK_DIR / f"model_{dst}"
    m_dst.mkdir(parents=True, exist_ok=True)
    for c in countries:
        for f in m_src.glob(f"*_{c}_*_{src}.parquet"):
            new = f.name.replace(f"_{src}.parquet", f"_{dst}.parquet")
            shutil.copy2(f, m_dst / new)
            print(f"copied cached features {f.name} -> {new}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="cand2")
    ap.add_argument("--dst", default="cand3")
    ap.add_argument("--countries", nargs="+", default=["US", "France"])
    a = ap.parse_args()
    main(a.src, a.dst, a.countries)
