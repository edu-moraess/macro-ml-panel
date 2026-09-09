"""Portfolio Engine V2 — allocation methods on real returns, constraints."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from core.risk import RiskEngine


class PortfolioEngine:
    def __init__(
        self,
        method: str = "inverse_vol",
        max_weight: float = 0.50,
        min_weight: float = 0.0,
        no_short: bool = True,
    ):
        self.method = method
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.no_short = no_short

    def _apply_constraints(self, w: np.ndarray) -> np.ndarray:
        if self.no_short:
            w = np.maximum(w, 0.0)
        w = np.clip(w, self.min_weight, self.max_weight)
        s = w.sum()
        if s <= 0:
            w = np.ones_like(w) / len(w)
        else:
            w = w / s
        w = np.clip(w, self.min_weight, self.max_weight)
        w = w / w.sum()
        return w

    def weights(self, returns: pd.DataFrame) -> pd.Series:
        r = returns.dropna()
        if len(r) < 24 or r.shape[1] < 2:
            raise ValueError("AMOSTRA INSUFICIENTE para portfolio.")
        cov = r.cov().values * 12
        vol = np.sqrt(np.diag(cov))
        n = r.shape[1]
        if self.method == "equal":
            w = np.ones(n) / n
        elif self.method == "inverse_vol":
            inv = 1.0 / np.maximum(vol, 1e-8)
            w = inv / inv.sum()
        elif self.method == "min_var":
            try:
                inv_cov = np.linalg.pinv(cov)
                ones = np.ones(n)
                w = inv_cov @ ones
                w = w / w.sum() if w.sum() != 0 else ones / n
            except Exception:
                w = np.ones(n) / n
        elif self.method == "risk_parity":
            w = 1.0 / np.maximum(vol, 1e-8)
            w = w / w.sum()
            for _ in range(20):
                marg = cov @ w
                rc = w * marg
                w = w * (rc.mean() / np.maximum(rc, 1e-12))
                w = w / w.sum()
        else:
            w = np.ones(n) / n
        w = self._apply_constraints(w)
        return pd.Series(w, index=r.columns, name="weight")

    def evaluate(self, returns: pd.DataFrame) -> Dict:
        w = self.weights(returns)
        port = (returns.dropna() @ w).rename("portfolio")
        return {
            "weights": w,
            "returns": port,
            "vol": float(port.std() * np.sqrt(12)),
            "sharpe": RiskEngine.sharpe(port),
            "max_dd": RiskEngine.max_drawdown(port),
            "equity": (1 + port).cumprod(),
            "risk_contribution": (w * (returns.dropna().cov() * 12 @ w)).rename("rc"),
        }
