"""Write matching_results.tsv and candidate_pairs.tsv in the exact competition format."""
import csv
import pandas as pd

def _grouped(all_s1_ids, pairs: pd.DataFrame, col: str) -> pd.DataFrame:
    """One row per S1 id (every id present, in input order), comma-joined unique other ids."""
    p = pairs[["s1_id", "other_id"]].drop_duplicates()
    lists = p.groupby("s1_id", sort=False)["other_id"].agg(",".join)
    out = pd.DataFrame({"source1_entity_id": pd.Series(all_s1_ids, dtype=object)})
    out[col] = out["source1_entity_id"].map(lists).fillna("")
    return out

def write_outputs(all_s1_ids, matches: pd.DataFrame, candidates: pd.DataFrame, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    kw = dict(sep="\t", index=False, quoting=csv.QUOTE_NONE, lineterminator="\n")
    _grouped(all_s1_ids, matches, "matched_entity_ids").to_csv(out_dir / "matching_results.tsv", **kw)
    _grouped(all_s1_ids, candidates, "candidate_entity_ids").to_csv(out_dir / "candidate_pairs.tsv", **kw)
    print(f"wrote {out_dir / 'matching_results.tsv'} and {out_dir / 'candidate_pairs.tsv'}")
