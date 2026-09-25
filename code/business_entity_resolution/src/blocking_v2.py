"""Blocker v2: union of three complementary blockers, run in parallel.

  tok   - IDF-weighted name+address tokens (= blocker v1)
  addr  - address tokens only: lets a record whose NAME is in another script (Tamil,
          Devanagari...) or is a trade name ('Dréxkor') find its owner through the address
  name  - character 4-grams of the name with spaces removed: catches domain-style names
          ('historicalcommittee.com') and typos, including records with no address

Each blocker proposes its top-k S1 records for every S2/S3 record; the union becomes the
candidate set. Then ALL three similarity scores are computed for EVERY candidate pair
(a pair found by one blocker still gets the other two scores), and become model features.

Parallelism: the S1 index is shared with worker processes through joblib memory-mapping,
so each worker does not hold its own copy (keeps 16 GB RAM workable).
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import CountVectorizer
from blocking import block_tokens, _topk_rows, _identity, _NUM_STREET

def addr_tokens(df: pd.DataFrame) -> list:
    words = df["addr_n"].str.split()
    combos = df["addr_n"].str.findall(_NUM_STREET)
    return [list(set(w) | {f"#{n}_{s}" for n, s in c}) for w, c in zip(words, combos)]

def name_grams(df: pd.DataFrame, n: int = 4) -> list:
    out = []
    for x in df["name_n"].str.replace(" ", "", regex=False):
        out.append(list({x[i:i + n] for i in range(len(x) - n + 1)} or ({x} if x else set())))
    return out

BLOCKERS = {"tok": block_tokens, "addr": addr_tokens, "name": name_grams}

class Index:
    """IDF-weighted sparse index over Source 1 for one blocker."""
    def __init__(self, s1, tok_fn, max_df_floor=500, max_df_frac=0.0025):
        self.tok_fn = tok_fn
        self.cv = CountVectorizer(analyzer=_identity, binary=True, dtype=np.float32)
        S = self.cv.fit_transform(tok_fn(s1)).tocsc()
        df = np.asarray(S.sum(axis=0)).ravel()
        max_df = max(max_df_floor, int(max_df_frac * S.shape[0]))
        self.keep = np.flatnonzero(df <= max_df)
        self.idf = np.log(S.shape[0] / df[self.keep]).astype(np.float32)
        self.Sw = (S[:, self.keep] @ sp.diags(self.idf)).tocsr().astype(np.float32)
        self.s_self = np.asarray(self.Sw.sum(axis=1)).ravel()
        self.ST = self.Sw.T.tocsr()

    def transform(self, others, chunk=500_000):
        parts = [self.cv.transform(self.tok_fn(others.iloc[i:i + chunk]))[:, self.keep].tocsr()
                 for i in range(0, len(others), chunk)]
        Q = sp.vstack(parts).tocsr().astype(np.float32)
        return Q, Q @ self.idf

def _topk_chunk(Q, ST, k, offset):
    r, c, v, _ = _topk_rows((Q @ ST).tocsr(), k)
    return (r + offset).astype(np.int32), c.astype(np.int32)

def _pair_scores(Q, Sw, qi, si, chunk=1_000_000):
    """Exact score for arbitrary (query, S1) pairs: row-wise dot product, chunked."""
    out = np.empty(len(qi), dtype=np.float32)
    for i in range(0, len(qi), chunk):
        a, b = Q[qi[i:i + chunk]], Sw[si[i:i + chunk]]
        out[i:i + chunk] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out

def block_country_v2(s1, others, k=3, n_jobs=4, chunk=5000, log=print):
    """Candidates for one country: DataFrame [other_id, s1_id, dice, score, rank,
    dice_tok, dice_addr, dice_name]. 'dice'/'score' = token blocker (v1-compatible)."""
    idx, Qs, qself = {}, {}, {}
    pairs = []
    for name, fn in BLOCKERS.items():
        idx[name] = Index(s1, fn)
        Qs[name], qself[name] = idx[name].transform(others)
        Q, ST = Qs[name], idx[name].ST
        res = Parallel(n_jobs=n_jobs, max_nbytes="1M")(
            delayed(_topk_chunk)(Q[i:i + chunk], ST, k, i) for i in range(0, Q.shape[0], chunk))
        qi = np.concatenate([r for r, _ in res]) if res else np.empty(0, np.int32)
        si = np.concatenate([c for _, c in res]) if res else np.empty(0, np.int32)
        pairs.append(pd.DataFrame({"qi": qi, "si": si}))
        log(f"    blocker '{name}': {len(qi):,} proposals")
    u = pd.concat(pairs, ignore_index=True).drop_duplicates()
    qi, si = u["qi"].to_numpy(), u["si"].to_numpy()

    out = pd.DataFrame({"other_id": others["entity_id"].to_numpy()[qi],
                        "s1_id": s1["entity_id"].to_numpy()[si]})
    for name in BLOCKERS:
        sc = _pair_scores(Qs[name], idx[name].Sw, qi, si)
        denom = qself[name][qi] + idx[name].s_self[si]
        out[f"dice_{name}"] = np.where(denom > 0, 2 * sc / np.maximum(denom, 1e-9), 0).astype(np.float32)
        if name == "tok":
            out["score"] = sc
    out["dice"] = out["dice_tok"]
    out = out.sort_values(["other_id", "dice"], ascending=[True, False], kind="stable")
    out["rank"] = out.groupby("other_id").cumcount().astype(np.int16)
    return out.reset_index(drop=True)
