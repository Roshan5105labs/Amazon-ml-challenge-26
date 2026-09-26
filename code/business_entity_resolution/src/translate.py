"""Learned script translation for Indian-language names and address components.

Many Source 2/3 records write Indian business names in Tamil, Devanagari or Kannada
('एसएस फूड प्राइवेट लिमिटेड') while Source 1 uses Latin script ('Ss Food Private Limited').
Character similarity between the two is zero, so blocking and name features fail.

We learn a translation dictionary ONLY from the training ground truth:
- names: in matched pairs with the same number of tokens, tokens align position by
  position (प्राइवेट <-> private). We count these alignments and keep a mapping when it is
  seen >= MIN_COUNT times and is the majority translation.
- address components: a native-script component (தமிழ்நாடு) is mapped to the Latin
  component of the matched Source 1 address it co-occurs with most (tamil nadu).
Pairs belonging to the validation entities used by train_model.py are EXCLUDED, so the
offline score is not inflated. No external data or transliteration service is used.

Output: a separate work folder (default <root>/work_t) with translated name_n / addr_n.
Unchanged candidate files (US, France) and cached US features are copied so only India
is recomputed. Run the rest of the pipeline with ER_WORK_DIR pointing at that folder.

  python code/business_entity_resolution/src/translate.py [--out work_t]
"""
import argparse
import re
import shutil
from collections import Counter, defaultdict
import pandas as pd
from config import WORK_DIR, ROOT, SOURCES
from normalize import normalize_series
from train_model import bucket

INDIC_PAT = "[\u0900-\u0DFF]"     # real characters (not an escape): works in re AND pyarrow
INDIC = re.compile(INDIC_PAT)
MIN_COUNT, MIN_SHARE = 2, 0.5
VAL_FRAC = 0.05   # must match train_model.py's --val-frac default

def _best(counts):
    out = {}
    for a, c in counts.items():
        b, n = c.most_common(1)[0]
        if n >= MIN_COUNT and n / sum(c.values()) >= MIN_SHARE:
            out[a] = b
    return out

def _components(addr: pd.Series) -> pd.Series:
    comp = addr.str.split(",").explode().str.strip()
    comp = comp[comp.notna() & (comp != "")]
    return pd.Series(normalize_series(comp).to_numpy(), index=comp.index)

def learn(tdir):
    s1 = pd.read_parquet(tdir / "source1.parquet", columns=["entity_id", "name_n", "business_address"])
    gt = pd.read_parquet(tdir / "gt_pairs.parquet")
    gt = gt[bucket(gt["s1_id"]) >= VAL_FRAC * 1000]          # exclude validation entities
    others = pd.concat([pd.read_parquet(tdir / f"{s}.parquet",
                                        columns=["entity_id", "name_n", "business_address"])
                        for s in ["source2", "source3"]], ignore_index=True)
    others = others[others["name_n"].str.contains(INDIC_PAT) | others["business_address"].str.contains(INDIC_PAT)]
    pairs = gt.merge(others, left_on="other_id", right_on="entity_id") \
              .merge(s1, left_on="s1_id", right_on="entity_id", suffixes=("_o", "_s"))
    print(f"learning from {len(pairs):,} matched pairs with native-script text")

    name_counts = defaultdict(Counter)
    for a, b in zip(pairs["name_n_o"], pairs["name_n_s"]):
        ta, tb = a.split(), b.split()
        if len(ta) == len(tb):
            for x, y in zip(ta, tb):
                if INDIC.search(x) and not INDIC.search(y):
                    name_counts[x][y] += 1
    names = _best(name_counts)

    comp_o = _components(pairs["business_address_o"])
    comp_o = comp_o[comp_o.str.contains(INDIC_PAT)]
    comp_s = _components(pairs["business_address_s"])
    s_by_row = comp_s.groupby(level=0).agg(list)
    # Dice association between a native component c and a Latin S1 component s:
    # 2*n(c,s) / (n(c) + n(s)), counted over pairs. 'tamil nadu' appears with 'தமிழ்நாடு'
    # in nearly every pair and rarely without it; a city or street does not.
    comp_counts, n_c, n_s = defaultdict(Counter), Counter(), Counter()
    rows_with_native = set(comp_o.index)
    for row, lst in s_by_row.items():
        if row in rows_with_native:
            n_s.update(set(lst))
    for row, cs in comp_o.groupby(level=0):
        latin = {s for s in s_by_row.get(row, []) if not INDIC.search(s)}
        for c in set(cs):
            n_c[c] += 1
            for s in latin:
                comp_counts[c][s] += 1
    comps = {}
    for c, cnt in comp_counts.items():
        s, n = max(cnt.items(), key=lambda kv: (2 * kv[1] / (n_c[c] + n_s[kv[0]]), kv[1]))
        if n >= MIN_COUNT and 2 * n / (n_c[c] + n_s[s]) >= MIN_SHARE:
            comps[c] = s
    print(f"learned {len(names):,} name-token and {len(comps):,} address-component translations")
    return names, comps

def translate_df(df, names, comps):
    """Rewrite name_n / addr_n for records containing native script; others unchanged."""
    m = df["name_n"].str.contains(INDIC_PAT) | df["business_address"].str.contains(INDIC_PAT)
    if not m.any():
        return df, 0.0
    sub = df.loc[m]
    toks = [t for x in sub["name_n"] for t in x.split() if INDIC.search(t)]
    cover = sum(t in names for t in toks) / max(len(toks), 1)
    df.loc[m, "name_n"] = [" ".join(names.get(t, t) for t in x.split()) for x in sub["name_n"]]
    comp = _components(sub["business_address"])
    comp = comp.map(lambda c: comps.get(c, c))
    joined = comp.groupby(level=0).agg(" ".join)
    df.loc[m, "addr_n"] = joined.reindex(sub.index).fillna("").to_numpy()
    return df, cover

def main(out_name):
    out = ROOT / out_name
    names, comps = learn(WORK_DIR / "train")
    sample = [(k, v) for k, v in names.items()][:12]
    print("sample name translations:", sample)
    print("sample address translations:", list(comps.items())[:8])
    for split in ["train", "test"]:
        src, dst = WORK_DIR / split, out / split
        dst.mkdir(parents=True, exist_ok=True)
        for s in SOURCES:
            df = pd.read_parquet(src / f"{s}.parquet")
            df, cover = translate_df(df, names, comps)
            df.to_parquet(dst / f"{s}.parquet", index=False)
            print(f"{split}/{s}: native-script name tokens covered by dictionary: {cover:.1%}")
        for f in list(src.glob("gt_pairs.parquet")) + list(src.glob("cand2_US.parquet")) \
                + list(src.glob("cand2_France.parquet")):
            shutil.copy2(f, dst / f.name)
    # cached US features are unaffected (US records contain no Indian scripts)
    mdir_src, mdir_dst = WORK_DIR / "model_cand2", out / "model_cand2"
    mdir_dst.mkdir(parents=True, exist_ok=True)
    for f in list(mdir_src.glob("*_US_*.parquet")):
        shutil.copy2(f, mdir_dst / f.name)
    print(f"\ndone -> {out}. Next: run blocking for India with ER_WORK_DIR={out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work_t")
    main(ap.parse_args().out)
