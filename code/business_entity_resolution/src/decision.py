"""F0.5-aware decision layer.

The metric scores each Source 1 entity as a whole: F0.5 of its predicted match set, and
1.0 for a correctly empty prediction. A fixed per-record threshold ignores that. Here,
for every S1 entity we compare, using the model's probabilities:
  - predicting its top-k records, expected F0.5 ~ 1.25*S_k / (0.25*(S_all + beta) + k)
      S_k   = sum of the k highest probabilities pointing at this entity
      S_all = sum of all probabilities pointing at it (expected number of true matches)
      beta  = allowance for true matches that blocking missed (tuned)
  - predicting nothing, expected score = P(no true match) = prod(1 - p_i)
and keep whichever is larger. Each S2/S3 record still goes to at most one entity (its
highest-probability candidate); records below tau_min are never used.
"""
import numpy as np
import pandas as pd

def decide_ef(pred: pd.DataFrame, tau_min: float = 0.2, beta: float = 0.0) -> pd.DataFrame:
    best = pred.sort_values("p", ascending=False).drop_duplicates("other_id")
    best = best[best["p"] >= tau_min]
    if best.empty:
        return best[["s1_id", "other_id"]]
    best = best.sort_values(["s1_id", "p"], ascending=[True, False], kind="stable")
    g = best.groupby("s1_id", sort=False)
    p = best["p"].to_numpy()
    k = g.cumcount().to_numpy() + 1
    s_k = g["p"].cumsum().to_numpy()
    s_all = g["p"].transform("sum").to_numpy()
    f_k = 1.25 * s_k / (0.25 * (s_all + beta) + k)
    log_empty = np.log1p(-np.clip(p, 0, 1 - 1e-9))
    p_empty = np.exp(pd.Series(log_empty, index=best.index).groupby(best["s1_id"]).transform("sum").to_numpy())
    best = best.assign(f_k=f_k, p_empty=p_empty, k=k)
    best_f = best.groupby("s1_id")["f_k"].transform("max")
    k_star = best.loc[best["f_k"] == best_f].groupby("s1_id")["k"].min()
    keep = (best["k"] <= best["s1_id"].map(k_star)) & (best_f > best["p_empty"])
    return best.loc[keep, ["s1_id", "other_id"]]
