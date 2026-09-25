"""Competition metric: macro-averaged F0.5 over Source 1 entities (singletons included)."""
import numpy as np
import pandas as pd

def macro_f05(pred_pairs: pd.DataFrame, true_pairs: pd.DataFrame, s1_ids) -> float:
    """pred_pairs / true_pairs: DataFrames with columns [s1_id, other_id].
    s1_ids: every Source 1 entity in the evaluation set (entities absent from both
    frames are singletons predicted empty -> score 1.0)."""
    ids = pd.Index(pd.unique(pd.Series(list(s1_ids))))
    pred = pred_pairs[["s1_id", "other_id"]].drop_duplicates()
    true = true_pairs[["s1_id", "other_id"]].drop_duplicates()
    n_pred = pred.groupby("s1_id").size().reindex(ids, fill_value=0).to_numpy(float)
    n_true = true.groupby("s1_id").size().reindex(ids, fill_value=0).to_numpy(float)
    tp = pred.merge(true, on=["s1_id", "other_id"]).groupby("s1_id").size().reindex(ids, fill_value=0).to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p, r = tp / n_pred, tp / n_true
        f = np.where(tp > 0, 1.25 * p * r / (0.25 * p + r), 0.0)
    f[(n_pred == 0) & (n_true == 0)] = 1.0
    return float(f.mean())

if __name__ == "__main__":
    # sanity check against the worked example in the problem statement (expects 0.714)
    pred = pd.DataFrame({"s1_id": ["A"] * 3, "other_id": ["S2-47", "S2-193", "S3-812"]})
    true = pd.DataFrame({"s1_id": ["A"] * 2, "other_id": ["S2-47", "S3-812"]})
    print(round(macro_f05(pred, true, ["A"]), 3))
