"""Portfolio Engine — allocation methods on observed return series."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from core.risk import RiskEngine


class PortfolioEngine:
    def __init__(self, method: str = "inverse_vol", max_weight: float = 0.50, min_weight: float = 0.0, no_short: bool = True, ann: int = 12):
        if max_weight <= 0 or min_weight < 0 or min_weight > max_weight or ann <= 0:
            raise ValueError("invalid portfolio constraints")
        self.method = method
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.no_short = no_short
        self.ann = ann

    def _project_constraints(self, w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=float)
        if not np.isfinite(w).all():
            raise ValueError("non-finite portfolio weights")
        if self.no_short:
            w = np.maximum(w, 0.0)
        if w.sum() <= 0:
            w = np.ones_like(w)
        w = w / w.sum()
        for _ in range(100):
            old = w.copy()
            w = np.clip(w, self.min_weight, self.max_weight)
            w = w / w.sum()
            if np.max(np.abs(w - old)) < 1e-10:
                break
        if np.any(w < self.min_weight - 1e-8) or np.any(w > self.max_weight + 1e-8):
            raise ValueError("incompatible portfolio weight constraints")
        return w

    def weights(self, returns: pd.DataFrame) -> pd.Series:
        r = returns.apply(pd.to_numeric, errors="coerce").dropna(how="any")
        if len(r) < 24 or r.shape[1] < 2:
            raise ValueError("AMOSTRA INSUFICIENTE para portfolio")
        cov = r.cov().to_numpy() * self.ann
        cov = (cov + cov.T) / 2.0
        eig_min = np.linalg.eigvalsh(cov).min()
        if eig_min < 0:
            cov += np.eye(len(cov)) * (-eig_min + 1e-10)
        vol = np.sqrt(np.maximum(np.diag(cov), 1e-12))
        n = r.shape[1]

        if self.method == "equal":
            w = np.ones(n) / n
        elif self.method == "inverse_vol":
            inv = 1.0 / vol
            w = inv / inv.sum()
        elif self.method == "min_var":
            inv_cov = np.linalg.pinv(cov, rcond=1e-10)
            ones = np.ones(n)
            raw = inv_cov @ ones
            w = raw / raw.sum() if abs(raw.sum()) > 1e-12 else ones / n
        elif self.method == "risk_parity":
            w = 1.0 / vol
            w = w / w.sum()
            for _ in range(200):
                mrc = cov @ w
                rc = w * mrc
                target = rc.mean()
                if target <= 0:
                    break
                updated = w * target / np.maximum(rc, 1e-12)
                updated = updated / updated.sum()
                if np.max(np.abs(updated - w)) < 1e-8:
                    w = updated
                    break
                w = updated
        else:
            raise ValueError(f"unknown portfolio method: {self.method}")

        w = self._project_constraints(w)
        return pd.Series(w, index=r.columns, name="weight")

    def evaluate(self, returns: pd.DataFrame) -> Dict:
        r = returns.apply(pd.to_numeric, errors="coerce").dropna(how="any")
        w = self.weights(r)
        port = (r @ w).rename("portfolio")
        cov = r.cov() * self.ann
        rc = (w * (cov @ w)).rename("rc")
        return {
            "weights": w,
            "returns": port,
            "vol": float(port.std(ddof=1) * np.sqrt(self.ann)),
            "sharpe": RiskEngine.sharpe(port, self.ann),
            "max_dd": RiskEngine.max_drawdown(port),
            "equity": (1.0 + port).cumprod(),
            "risk_contribution": rc,
            "annualization": self.ann,
        }
