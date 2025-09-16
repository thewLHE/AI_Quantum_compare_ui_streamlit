from __future__ import annotations
import time, math, re
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd

@dataclass
class VQEResult:
    series: List[float]
    emin: float
    backend: str

def _exp_convergence(steps:int=80, noise:float=0.02, true_E:float=-1.234, seed:int=0):
    rng = np.random.default_rng(seed)
    vals = []
    for t in range(steps):
        e = true_E + (1.0 + 0.3*math.cos(t/7)) * math.exp(-t/12) + rng.normal(0, noise)
        vals.append(float(e))
        time.sleep(0.01)
    return vals

def run_spin_chain(steps:int=80, noise:float=0.02, seed:int=0) -> VQEResult:
    # Placeholder mock; hook up Qiskit VQE here later
    vals = _exp_convergence(steps=steps, noise=noise, seed=seed)
    return VQEResult(series=vals, emin=float(np.min(vals)), backend="mock:spin")

def parse_pauli_text(text:str) -> List[Tuple[float,str]]:
    """
    Lines like:
      -1.0 ZI
      +0.5 XX
    Returns list of (coeff, pauli_str)
    """
    terms = []
    for line in text.splitlines():
        line = line.strip()
        if (not line) or line.startswith("#"): 
            continue
        m = re.match(r"([+\-]?\d+(\.\d+)?([eE][+\-]?\d+)?)\s+([IXYZ]+)$", line)
        if not m: 
            continue
        coeff = float(m.group(1))
        pstr = m.group(4)
        terms.append((coeff, pstr))
    return terms

def run_custom_pauli(terms:List[Tuple[float,str]], steps:int=80, noise:float=0.02, seed:int=0) -> VQEResult:
    # Placeholder: just uses different "true_E" based on hash of terms
    thash = sum(int(abs(c)*1000) + len(p) for c,p in terms) if terms else 0
    true_E = -1.0 - 0.0005*thash
    vals = _exp_convergence(steps=steps, noise=noise, seed=seed, true_E=true_E)
    return VQEResult(series=vals, emin=float(np.min(vals)), backend="mock:custom")