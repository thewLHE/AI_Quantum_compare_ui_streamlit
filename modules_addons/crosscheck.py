"""
Utilities to cross-check AI pLDDT vs Quantum local energy.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class CrossCheckResult:
    corr: float
    df: pd.DataFrame

def merge_plddt_and_scores(plddt_series, residue_scores_csv: str) -> CrossCheckResult:
    """
    pLDDT from PDB (indexed by (chain,resi)) and residue_scores CSV.
    plddt_series: pd.Series indexed by tuples (chain, resi) -> plddt float
    residue_scores_csv: path to CSV with columns: chain,resi,score

    Returns Pearson correlation between (1 - norm_plddt) and score.
    """
    rs = pd.read_csv(residue_scores_csv)
    rs["key"] = list(zip(rs["chain"].astype(str), rs["resi"].astype(int)))
    df = pd.DataFrame({"key": list(plddt_series.index), "plddt": plddt_series.values})
    merged = df.merge(rs[["key","score"]], on="key", how="inner")
    if merged.empty:
        return CrossCheckResult(corr=float("nan"), df=merged)
    # normalize pLDDT to [0,1]
    p = merged["plddt"].clip(0,100)/100.0
    target = 1.0 - p  # lower pLDDT -> higher instability
    corr = float(pd.Series(target).corr(merged["score"]))
    merged["instability"]=target
    return CrossCheckResult(corr=corr, df=merged)
