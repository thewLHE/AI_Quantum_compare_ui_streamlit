from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class ConvMetrics:
    delta_e: float
    r2_exp: float
    var_last: float

def _safe(arr):
    return np.array(arr, dtype=float)

def convergence_metrics(energies, last_n:int=10) -> ConvMetrics:
    y = _safe(energies)
    if y.size < 2:
        return ConvMetrics(delta_e=float("nan"), r2_exp=float("nan"), var_last=float("nan"))
    delta_e = float(y[0] - y.min())
    # crude exponential-fit R^2 via log transform around min
    eps = max(1e-6, 1e-3 * max(1.0, abs(y).max()))
    ymin = y.min()
    yy = np.log(np.clip(y - ymin + eps, eps, None))
    x = np.arange(y.size, dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    a,b = np.linalg.lstsq(A, yy, rcond=None)[0]
    yfit = a*x + b
    ss_res = float(np.sum((yy - yfit)**2))
    ss_tot = float(np.sum((yy - yy.mean())**2))
    r2 = float(1 - ss_res/ss_tot) if ss_tot>0 else float("nan")
    v_last = float(np.var(y[-min(last_n, y.size):]))
    return ConvMetrics(delta_e=delta_e, r2_exp=r2, var_last=v_last)