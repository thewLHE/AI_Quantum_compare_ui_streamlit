from __future__ import annotations
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd

def parse_header_remark(pdb_text: str, max_lines: int = 40) -> str:
    lines = []
    for l in pdb_text.splitlines():
        if l.startswith(("HEADER","TITLE","REMARK 1")):
            lines.append(l)
        if len(lines) >= max_lines:
            break
    return "\n".join(lines)

def parse_plddt_series(pdb_text: str):
    """
    Returns:
        plddt_arr: np.ndarray of pLDDT values (per atom)
        per_residue: pd.Series indexed by (chain,resi) -> mean pLDDT 
    """
    atoms = []
    res_keys = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM","HETATM")):
            try:
                b = float(line[60:66])           # B-factor slot -> pLDDT
                chain = line[21].strip() or "A"  # chain ID at col 22 (0-based 21)
                resi = int(line[22:26])
            except Exception:
                continue
            atoms.append(b)
            res_keys.append((chain, resi))
    if not atoms:
        return None, None
    arr = np.array(atoms, dtype=float)
    df = pd.DataFrame({"key": res_keys, "plddt": arr})
    per_res = df.groupby("key")["plddt"].mean().sort_index()
    per_res.index = pd.Index(per_res.index, name="key")
    return arr, per_res

def render_pdb_bfactor(pdb_text: str, width=720, height=560):
    import py3Dmol, streamlit.components.v1 as components
    v = py3Dmol.view(width=width, height=height)
    v.addModel(pdb_text, 'pdb')
    v.setStyle({'cartoon': {'colorscheme': 'bfactor'}})  # color by B-factor (pLDDT)
    v.zoomTo()
    components.html(v._make_html(), height=height, scrolling=False)

def render_overlay_by_residue(pdb_text: str, residue_scores: Dict[str, float], width=720, height=560):
    """
    residue_scores: mapping like {"A:123": 0.75, ...} in [0,1]
    """
    import py3Dmol, streamlit.components.v1 as components
    v = py3Dmol.view(width=width, height=height)
    v.addModel(pdb_text,'pdb')
    # base
    v.setStyle({'cartoon': {'color': 'white'}})
    # overlay
    for k, val in residue_scores.items():
        try:
            chain, resi = k.split(":")
            resi = int(resi)
        except Exception:
            continue
        color = 'yellow'
        if val > 0.7:
            color = 'red'
        elif val > 0.4:
            color = 'orange'
        v.setStyle({'chain': chain, 'resi': resi}, {'cartoon': {'color': color}})
    v.zoomTo()
    components.html(v._make_html(), height=height, scrolling=False)