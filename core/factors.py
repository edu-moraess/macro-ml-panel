"""Factor Engine V2 — robust macro factors with direction, weight, contribution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from data_utils import rolling_zscore


@dataclass
class FactorSpec:
    name: str
    direction: float
    weight: float


DEFAULT_SPECS = [
    FactorSpec("Growth", +1.0, 1.0),
    FactorSpec("Inflation", -1.0, 1.0),
    FactorSpec("Rates", -1.0, 1.0),
    FactorSpec("FX", +1.0, 1.0),
    FactorSpec("Liquidity", +1.0, 1.0),
    FactorSpec("Momentum", +1.0, 1.0),
]


class FactorEngine:
    def __init__(self, window: int = 60, specs: Optional[list] = None):
        self.window = window
        self.specs = specs or DEFAULT_SPECS
        self._weights = self._normalize_weights()

    def _normalize_weights(self) -> np.ndarray:
        w = np.array([s.weight * s.direction for s in self.specs], dtype=float)
        denom = np.abs(w).sum()
        return w / denom if denom > 0 else w

    def build(
        self,
        growth_raw: pd.Series,
        inflation_raw: pd.Series,
        rates_raw: pd.Series,
        fx_raw: pd.Series,
        liquidity_raw: pd.Series,
        momentum_raw: pd.Series,
    ) -> pd.DataFrame:
        z = lambda s: rolling_zscore(s, self.window)
        f = pd.DataFrame(
            {
                "Growth": z(growth_raw),
                "Inflation": z(inflation_raw),
                "Rates": z(rates_raw),
                "FX": z(fx_raw),
                "Liquidity": z(liquidity_raw),
                "Momentum": z(momentum_raw),
            }
        )
        return f.replace([np.inf, -np.inf], np.nan).dropna()

    def composite(self, factors: pd.DataFrame) -> pd.Series:
        cols = [s.name for s in self.specs]
        present = [c for c in cols if c in factors.columns]
        if not present:
            return pd.Series(dtype=float)
        wmap = {s.name: s.weight * s.direction for s in self.specs}
        ww = np.array([wmap[c] for c in present])
        ww = ww / np.abs(ww).sum()
        return (factors[present] * ww).sum(axis=1).rename("macro_score")

    def latest_state(self, factors: pd.DataFrame) -> Dict:
        if factors.empty:
            return {"score": np.nan, "regime": "N/A", "confidence": 0.0, "contributions": {}}
        latest = factors.iloc[-1]
        score = float(self.composite(factors).iloc[-1])
        regime_score = float(np.tanh(score / 1.5))
        if regime_score >= 0.30:
            regime = "RISK-ON"
        elif regime_score <= -0.30:
            regime = "RISK-OFF"
        else:
            regime = "NEUTRAL"
        conf = min(99.0, 50.0 + 50.0 * abs(regime_score))
        wmap = {s.name: s.weight * s.direction for s in self.specs}
        present = [c for c in wmap if c in latest.index]
        ww = np.array([wmap[c] for c in present])
        ww = ww / np.abs(ww).sum() if np.abs(ww).sum() > 0 else ww
        contrib = {c: float(latest[c] * ww[i]) for i, c in enumerate(present)}
        hist = self.composite(factors)
        pct = float((hist <= score).mean() * 100) if len(hist) > 5 else np.nan
        return {
            "score": score,
            "regime_score": regime_score,
            "regime": regime,
            "confidence": conf,
            "contributions": contrib,
            "percentile": pct,
            "z_latest": latest.to_dict(),
            "n_obs": len(factors),
            "last_date": factors.index.max(),
        }
