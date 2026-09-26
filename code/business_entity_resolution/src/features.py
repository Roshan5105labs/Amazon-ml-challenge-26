"""Pair features for the matching model.
 
Two groups:
- blocking features: computed on the FULL candidate set of a country (rank, margins,
  'support' = how many S2/S3 records point at an S1). They must be computed before any
  subsampling so training rows see the same distribution as test rows.
- text features: string similarities for each (S2/S3 record, S1 candidate) pair.
 
No feature uses the country label, so the model applies unchanged to France.
"""
import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz
from rapidfuzz.distance import JaroWinkler
 
FEATURES = [
    # blocking / context
    "score", "dice", "rank", "dice_gap_best", "q_margin", "q_ncand", "s1_support", "s1_ncand", "is_s2",
    # name
    "name_jw", "name_tset", "name_tsort", "name_partial", "name_ratio",
    "name_ns_ratio", "name_ns_partial", "name_len_o", "name_len_s", "o_name_ascii", "s_name_ascii",
    # address
    "addr_tset", "addr_partial", "addr_ratio", "o_addr_empty", "addr_len_o", "addr_len_s",
    # numbers in address (house numbers, PIN / ZIP / postal codes)
    "num_shared", "num_o", "num_s", "num_conflict", "first_num_eq",
]
 
# extra features available with blocker-v2 candidates (cand2_*): all three blocker scores
FEATURES_V2 = FEATURES + ["dice_addr", "dice_name", "gap_addr", "gap_name", "max_dice", "gap_max"]
 
def feature_list(cand_columns) -> list:
    return FEATURES_V2 if "dice_addr" in cand_columns else FEATURES
 
def _to_f32(df: pd.DataFrame) -> pd.DataFrame:
    """Store float columns as float32: halves memory, irrelevant for similarity features."""
    cols = df.select_dtypes(include="float64").columns
    if len(cols):
        df[cols] = df[cols].astype(np.float32)
    return df
 
def block_features(cand: pd.DataFrame) -> pd.DataFrame:
    """Context features from the full candidate set of one country.
    Adds columns IN PLACE (no copy of the 50M-row table) and returns the same frame."""
    c = cand
    if "dice_addr" in c.columns:
        c["max_dice"] = c[["dice_tok", "dice_addr", "dice_name"]].max(axis=1)
        for col, gap in [("dice_addr", "gap_addr"), ("dice_name", "gap_name"), ("max_dice", "gap_max")]:
            c[gap] = c.groupby("other_id")[col].transform("max") - c[col]
    best = c.groupby("other_id")["dice"].transform("max")
    second = c.loc[c["rank"] == 1].set_index("other_id")["dice"]
    c["dice_gap_best"] = best - c["dice"]
    c["q_margin"] = best - c["other_id"].map(second).fillna(0)
    c["q_ncand"] = c.groupby("other_id")["dice"].transform("size")
    c["s1_support"] = c["s1_id"].map(c.loc[c["rank"] == 0, "s1_id"].value_counts()).fillna(0)
    c["s1_ncand"] = c.groupby("s1_id")["dice"].transform("size")
    c["is_s2"] = c["other_id"].str.startswith("S2-").astype(np.int8)
    return _to_f32(c)
 
def _pairwise(a, b, scorer, **kw):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw)
 
def text_features(pairs: pd.DataFrame, s1: pd.DataFrame, others: pd.DataFrame) -> pd.DataFrame:
    """String-similarity features. s1 / others: records with entity_id, business_name, name_n, addr_n."""
    si = pd.Index(s1["entity_id"]).get_indexer(pairs["s1_id"])
    oi = pd.Index(others["entity_id"]).get_indexer(pairs["other_id"])
    sn, on_ = s1["name_n"].to_numpy()[si], others["name_n"].to_numpy()[oi]
    sa, oa = s1["addr_n"].to_numpy()[si], others["addr_n"].to_numpy()[oi]
    f = pairs.copy()
 
    f["name_jw"] = _pairwise(on_, sn, JaroWinkler.normalized_similarity)
    f["name_tset"] = _pairwise(on_, sn, fuzz.token_set_ratio)
    f["name_tsort"] = _pairwise(on_, sn, fuzz.token_sort_ratio)
    f["name_partial"] = _pairwise(on_, sn, fuzz.partial_ratio)
    f["name_ratio"] = _pairwise(on_, sn, fuzz.ratio)
    # spaces removed: 'historicalcommittee com' vs 'historical committee inc'
    ns_o = np.array([x.replace(" ", "") for x in on_], dtype=object)
    ns_s = np.array([x.replace(" ", "") for x in sn], dtype=object)
    f["name_ns_ratio"] = _pairwise(ns_o, ns_s, fuzz.ratio)
    f["name_ns_partial"] = _pairwise(ns_o, ns_s, fuzz.partial_ratio)
    f["name_len_o"] = np.fromiter((len(x) for x in on_), np.int32, len(on_))
    f["name_len_s"] = np.fromiter((len(x) for x in sn), np.int32, len(sn))
    raw_o = others["business_name"].to_numpy()[oi]
    raw_s = s1["business_name"].to_numpy()[si]
    f["o_name_ascii"] = np.fromiter((x.isascii() for x in raw_o), np.int8, len(raw_o))
    f["s_name_ascii"] = np.fromiter((x.isascii() for x in raw_s), np.int8, len(raw_s))
 
    f["addr_tset"] = _pairwise(oa, sa, fuzz.token_set_ratio)
    f["addr_partial"] = _pairwise(oa, sa, fuzz.partial_ratio)
    f["addr_ratio"] = _pairwise(oa, sa, fuzz.ratio)
    f["o_addr_empty"] = (oa == "").astype(np.int8)
    f["addr_len_o"] = np.fromiter((len(x) for x in oa), np.int32, len(oa))
    f["addr_len_s"] = np.fromiter((len(x) for x in sa), np.int32, len(sa))
 
    # numbers: per-record sets computed once, then compared per pair
    s_nums = s1["addr_n"].str.findall(r"\d+").map(frozenset).to_numpy()[si]
    o_nums = others["addr_n"].str.findall(r"\d+").map(frozenset).to_numpy()[oi]
    shared = np.fromiter((len(a & b) for a, b in zip(o_nums, s_nums)), np.int16, len(si))
    n_o = np.fromiter((len(a) for a in o_nums), np.int16, len(si))
    n_s = np.fromiter((len(b) for b in s_nums), np.int16, len(si))
    f["num_shared"], f["num_o"], f["num_s"] = shared, n_o, n_s
    f["num_conflict"] = ((n_o > 0) & (n_s > 0) & (shared == 0)).astype(np.int8)
    s_first = s1["addr_n"].str.extract(r"(\d+)", expand=False).to_numpy()[si]
    o_first = others["addr_n"].str.extract(r"(\d+)", expand=False).to_numpy()[oi]
    f["first_num_eq"] = np.where(pd.isna(o_first) | pd.isna(s_first), -1,
                                 (o_first == s_first).astype(np.int8)).astype(np.int8)
    return _to_f32(f)
 