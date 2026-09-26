"""Harmonize French S2/S3 text toward Source 1 conventions, learned from the data itself.
 
Why: France has no training labels, and its sources disagree systematically:
  - Source 1 writes the REGION ('Hauts-de-France'), S2/S3 often the DEPARTMENT ('Nord')
  - S2/S3 abbreviate street words ('R.' for 'Rue', 'BD.' for 'Boulevard', 'ST-' for 'Saint')
  - dotted legal forms 'S.A.S' normalize to 's a s' instead of 'sas'
How (no external data, no hand-written lists):
  - take this country's CONFIDENT test matches (model probability >= --min-p) as pseudo-labels
  - learn address-component and address-token correspondences between the differing parts
    of those pairs, with the same Dice association used by translate.py
  - join runs of single letters in names ('s a s' -> 'sas')
The rewritten records replace this country's rows in <work>/test/source2|3.parquet (originals
kept once as *.orig.parquet). The model is NOT retrained: features are country-agnostic.
It also prepares candidate set --dst: copies other countries' candidates, the model and their
cached test scores, so only this country is re-blocked and re-scored.
 
  python code/business_entity_resolution/src/france_fix.py [--src cand3 --dst cand4]
"""
import argparse
import re
import shutil
from collections import Counter, defaultdict
import pandas as pd
from config import WORK_DIR
from normalize import normalize_series
from train_model import model_dir
 
MIN_COUNT = 20
_SINGLE_RUN = re.compile(r"\b(?:[a-z] )+[a-z]\b")
 
def join_letters(s: pd.Series) -> pd.Series:
    """'coeur france patrimoine s a s' -> 'coeur france patrimoine sas'"""
    return s.str.replace(_SINGLE_RUN, lambda m: m.group(0).replace(" ", ""), regex=True)
 
def _comps(addr: pd.Series) -> pd.Series:
    c = addr.str.split(",").explode().str.strip()
    c = c[c.notna() & (c != "")]
    return pd.Series(normalize_series(c).to_numpy(), index=c.index)
 
def _dice_map(pairs_a, pairs_b, min_share):
    """pairs_a/pairs_b: aligned lists of sets (differing parts of each confident pair)."""
    co, na, nb = defaultdict(Counter), Counter(), Counter()
    for a, b in zip(pairs_a, pairs_b):
        na.update(a); nb.update(b)
        for x in a:
            for y in b:
                co[x][y] += 1
    out = {}
    for x, cnt in co.items():
        y, n = max(cnt.items(), key=lambda kv: (2 * kv[1] / (na[x] + nb[kv[0]]), kv[1]))
        if n >= MIN_COUNT and 2 * n / (na[x] + nb[y]) >= min_share:
            out[x] = y
    return out
 
def main(country, src, dst, min_p, k_note):
    tdir = WORK_DIR / "test"
    f = [("country", "==", country)]
    s1 = pd.read_parquet(tdir / "source1.parquet", filters=f)
    scored = pd.read_parquet(tdir / f"scored_{src}_{country}.parquet")
    best = scored.sort_values("p", ascending=False).drop_duplicates("other_id")
    conf = best[best["p"] >= min_p]
    # ALWAYS learn from the untouched originals: after a first run the current files are
    # already harmonized, and learning from them would yield only leftovers (e.g. bis -> b)
    frames = {s: pd.read_parquet(tdir / f"{s}.orig.parquet") if (tdir / f"{s}.orig.parquet").exists()
              else pd.read_parquet(tdir / f"{s}.parquet") for s in ["source2", "source3"]}
    oth = pd.concat([df[df["country"] == country] for df in frames.values()])
    print(f"{country}: learning from {len(conf):,} confident matches (p >= {min_p})")
 
    pairs = conf.merge(oth[["entity_id", "business_address", "addr_n"]], left_on="other_id", right_on="entity_id") \
                .merge(s1[["entity_id", "business_address", "addr_n"]], left_on="s1_id", right_on="entity_id",
                       suffixes=("_o", "_s")).reset_index(drop=True)
    # 1. address components (department -> region), only the components that differ
    co_ = _comps(pairs["business_address_o"]).groupby(level=0).agg(set)
    cs_ = _comps(pairs["business_address_s"]).groupby(level=0).agg(set)
    idx = co_.index.intersection(cs_.index)
    comp_map = _dice_map([co_[i] - cs_[i] for i in idx], [cs_[i] - co_[i] for i in idx], 0.3)
    # keep only place-level mappings (department -> region, 'st nazaire' -> 'saint nazaire').
    # Street-level ones learned from a few big businesses can INVENT house numbers
    # ('rue de lorraine' -> '4 rue de lorraine'), which would create false matches.
    comp_map = {c: s for c, s in comp_map.items() if not re.search(r"\d", c) and not re.search(r"\d", s)}
    # 2. short address tokens (abbreviations), only tokens that differ
    to = pairs["addr_n_o"].str.split().map(set)
    ts = pairs["addr_n_s"].str.split().map(set)
    tok_map = _dice_map([{t for t in a - b if len(t) <= 3 and not t.isdigit()} for a, b in zip(to, ts)],
                        [{t for t in b - a if not t.isdigit()} for a, b in zip(to, ts)], 0.5)
    print(f"learned {len(comp_map)} address-component and {len(tok_map)} abbreviation mappings")
    print("  components:", comp_map)
    print("  abbreviations:", tok_map)
 
    def rewrite(df):
        m = df["country"] == country
        sub = df.loc[m]
        comps = _comps(sub["business_address"]).map(lambda c: comp_map.get(c, c))
        addr = comps.groupby(level=0).agg(" ".join).reindex(sub.index).fillna("")
        addr = addr.map(lambda a: " ".join(tok_map.get(t, t) for t in a.split()))
        df.loc[m, "addr_n"] = addr.to_numpy()
        df.loc[m, "name_n"] = join_letters(sub["name_n"]).to_numpy()
        return df
 
    for s, df in frames.items():
        orig = tdir / f"{s}.orig.parquet"
        if not orig.exists():
            shutil.copy2(tdir / f"{s}.parquet", orig)
        df = pd.read_parquet(orig)                      # always rewrite from the original
        rewrite(df).to_parquet(tdir / f"{s}.parquet", index=False)
    s1_all = pd.read_parquet(tdir / "source1.parquet")
    m = s1_all["country"] == country
    s1_all.loc[m, "name_n"] = join_letters(s1_all.loc[m, "name_n"]).to_numpy()
    s1_all.to_parquet(tdir / "source1.parquet", index=False)
    print(f"rewrote {country} text in test source files")
 
    # prepare candidate set dst: reuse other countries' candidates, the model, and their scores
    mdir_src, mdir_dst = model_dir(src), model_dir(dst)
    mdir_dst.mkdir(parents=True, exist_ok=True)
    for fn in ["lgbm.txt", "config.json", "decision.json"]:
        if (mdir_src / fn).exists():
            shutil.copy2(mdir_src / fn, mdir_dst / fn)
    for c in s1_all["country"].unique():
        if c != country:
            shutil.copy2(tdir / f"{src}_{c}.parquet", tdir / f"{dst}_{c}.parquet")
            shutil.copy2(tdir / f"scored_{src}_{c}.parquet", tdir / f"scored_{dst}_{c}.parquet")
    print(f"prepared '{dst}': next, re-block {country} with --name {dst} --countries {country} --k {k_note}")
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="France")
    ap.add_argument("--src", default="cand3")
    ap.add_argument("--dst", default="cand4")
    ap.add_argument("--min-p", type=float, default=0.95)
    ap.add_argument("--k", type=int, default=6)
    a = ap.parse_args()
    main(a.country, a.src, a.dst, a.min_p, a.k)
 