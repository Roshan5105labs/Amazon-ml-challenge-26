"""Blocker v1: IDF-weighted shared-token blocking, queried from the S2/S3 side.

Each S2/S3 record belongs to at most one S1 entity (verified in EDA), so every S2/S3
record asks: "which k S1 records am I most likely to belong to?". Candidates are
scored by the summed IDF of shared tokens (name + address + house-number/street
composites). Tokens that are too common in S1 (e.g. 'ltd', 'road', 'street') are
dropped: they carry little signal and would make the sparse product explode.
Runs per country, in chunks, so memory stays bounded.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

_NUM_STREET = r"\b(\d+) ([^\W\d_][\w\u0900-\u0DFF]*)"   # '3315 fremont' -> '#3315_fremont'

def block_tokens(df: pd.DataFrame) -> pd.Series:
    """Set of blocking tokens per record, from normalized name + address."""
    words = (df["name_n"] + " " + df["addr_n"]).str.split()
    combos = df["addr_n"].str.findall(_NUM_STREET)
    return pd.Series([list(set(w) | {f"#{n}_{s}" for n, s in c}) for w, c in zip(words, combos)],
                     index=df.index)

def _topk_rows(R: sp.csr_matrix, k: int):
    """Row-wise top-k of a sparse score matrix -> (row, col, score, rank) arrays."""
    R.sum_duplicates()
    rows = np.repeat(np.arange(R.shape[0]), np.diff(R.indptr))
    order = np.lexsort((-R.data, rows))                   # rows stay grouped, scores descending
    rank = np.arange(len(order)) - R.indptr[rows[order]]
    keep = order[rank < k]
    return rows[keep], R.indices[keep], R.data[keep], rank[rank < k]

def block_country(s1: pd.DataFrame, others: pd.DataFrame, k=10, max_df=500, chunk=5000):
    """Returns DataFrame [other_id, s1_id, score, rank] (rank 0 = best S1 for that record)."""
    cv = CountVectorizer(analyzer=lambda toks: toks, binary=True, dtype=np.float32)
    S = cv.fit_transform(block_tokens(s1)).tocsc()
    df = np.asarray(S.sum(axis=0)).ravel()
    keep = np.flatnonzero(df <= max_df)
    idf = np.log(S.shape[0] / df[keep]).astype(np.float32)
    ST = (S[:, keep] @ sp.diags(idf)).T.tocsr()           # vocab x n_s1, IDF-weighted
    Q = cv.transform(block_tokens(others))[:, keep].tocsr()

    out = []
    s1_ids = s1["entity_id"].to_numpy()
    o_ids = others["entity_id"].to_numpy()
    for start in range(0, Q.shape[0], chunk):
        r, c, v, rk = _topk_rows((Q[start:start + chunk] @ ST).tocsr(), k)
        out.append(pd.DataFrame({"other_id": o_ids[start + r], "s1_id": s1_ids[c],
                                 "score": v, "rank": rk.astype(np.int16)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["other_id", "s1_id", "score", "rank"])

def block_all(s1: pd.DataFrame, others: pd.DataFrame, **kw) -> pd.DataFrame:
    """Block within each country (EDA: 100% of true matches share the country label).
    Countries are taken from the data, so unseen ones (France) are handled automatically."""
    parts = []
    for country in sorted(set(s1["country"]) | set(others["country"])):
        a, b = s1[s1["country"] == country], others[others["country"] == country]
        if len(a) and len(b):
            parts.append(block_country(a, b, **kw))
    return pd.concat(parts, ignore_index=True)
