"""Blocker v1: IDF-weighted shared-token blocking, queried from the S2/S3 side.
 
Each S2/S3 record belongs to at most one S1 entity (verified in EDA), so every S2/S3
record asks: "which k S1 records am I most likely to belong to?". Candidates are
scored by the summed IDF of shared tokens (name + address + house-number/street
composites). Tokens that are too common in S1 (e.g. 'ltd', 'road', 'street') are
dropped: they carry little signal and would make the sparse product explode.
 
Scale notes:
- max_df scales with the country's S1 size, so dev (small) and test (large) prune alike
- queries are tokenized chunk by chunk, so memory stays bounded on ~5M-record countries
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
 
_NUM_STREET = r"\b(\d+) ([^\W\d_][\w\u0900-\u0DFF]*)"   # '3315 fremont' -> '#3315_fremont'
 
def block_tokens(df: pd.DataFrame) -> list:
    """Set of blocking tokens per record, from normalized name + address."""
    words = (df["name_n"] + " " + df["addr_n"]).str.split()
    combos = df["addr_n"].str.findall(_NUM_STREET)
    return [list(set(w) | {f"#{n}_{s}" for n, s in c}) for w, c in zip(words, combos)]
 
def _identity(toks):
    return toks
 
def _topk_rows(R: sp.csr_matrix, k: int):
    """Row-wise top-k of a sparse score matrix -> (row, col, score, rank) arrays."""
    R.sum_duplicates()
    rows = np.repeat(np.arange(R.shape[0]), np.diff(R.indptr))
    order = np.lexsort((-R.data, rows))                   # rows stay grouped, scores descending
    rank = np.arange(len(order)) - R.indptr[rows[order]]
    sel = rank < k
    keep = order[sel]
    return rows[keep], R.indices[keep], R.data[keep], rank[sel]
 
def block_country(s1: pd.DataFrame, others: pd.DataFrame, k=10,
                  max_df_floor=500, max_df_frac=0.0025, chunk=2000) -> pd.DataFrame:
    """Top-k S1 candidates for every S2/S3 record of one country.
    Returns [other_id, s1_id, score, dice, rank]; rank 0 = best S1 for that record.
    dice = 2*shared / (own weight of record + own weight of S1), in [0, 1]."""
    max_df = max(max_df_floor, int(max_df_frac * len(s1)))
    cv = CountVectorizer(analyzer=_identity, binary=True, dtype=np.float32)
    S = cv.fit_transform(block_tokens(s1)).tocsc()
    df = np.asarray(S.sum(axis=0)).ravel()
    keep = np.flatnonzero(df <= max_df)
    idf = np.log(S.shape[0] / df[keep]).astype(np.float32)
    Sw = S[:, keep] @ sp.diags(idf)
    s1_self = np.asarray(Sw.sum(axis=1)).ravel()
    ST = Sw.T.tocsr()                                       # vocab x n_s1, IDF-weighted
    del S, Sw
 
    s1_ids = s1["entity_id"].to_numpy()
    o_ids = others["entity_id"].to_numpy()
    out = []
    for start in range(0, len(others), chunk):
        part = others.iloc[start:start + chunk]
        Q = cv.transform(block_tokens(part))[:, keep].tocsr()
        q_self = Q @ idf
        r, c, v, rk = _topk_rows((Q @ ST).tocsr(), k)
        dice = 2 * v / (q_self[r] + s1_self[c])
        out.append(pd.DataFrame({"other_id": o_ids[start + r], "s1_id": s1_ids[c],
                                 "score": v, "dice": dice.astype(np.float32),
                                 "rank": rk.astype(np.int16)}))
    if not out:
        return pd.DataFrame(columns=["other_id", "s1_id", "score", "dice", "rank"])
    return pd.concat(out, ignore_index=True)
 
def block_all(s1: pd.DataFrame, others: pd.DataFrame, **kw) -> pd.DataFrame:
    """Block within each country (EDA: 100% of true matches share the country label).
    Countries are taken from the data, so unseen ones (France) are handled automatically."""
    parts = []
    for country in sorted(set(s1["country"]) | set(others["country"])):
        a, b = s1[s1["country"] == country], others[others["country"] == country]
        if len(a) and len(b):
            parts.append(block_country(a, b, **kw))
    return pd.concat(parts, ignore_index=True)
 