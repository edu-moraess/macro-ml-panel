"""Macro Signal Engine — aggregate factor score with documented confidence.

Score scale: approximately [-100, +100] via 50 * tanh(raw_composite).
Contributions sum to the composite before scaling (documented).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.factors import FactorEngine, DEFAULT_SPECS


class SignalEngine:
    """Build MACRO SCORE from factor panel. No synthetic inputs."""

    def __init__(self, factor_engine: Optional[FactorEngine] = None):
        self.fe = factor_engine or FactorEngine()

    def contributions(self, factors: pd.DataFrame) -> pd.DataFrame:
        if factors.empty:
            return pd.DataFrame()
        wmap = {s.name: s.weight * s.direction for s in self.fe.specs}
        present = [c for c in factors.columns if c in wmap]
        ww = np.array([wmap[c] for c in present], dtype=float)
        denom = np.abs(ww).sum()
        if denom <= 0:
            return pd.DataFrame(index=factors.index)
        ww = ww / denom
        contrib = factors[present].multiply(ww, axis=1)
        contrib["composite"] = contrib.sum(axis=1)
        return contrib

    def score_series(self, factors: pd.DataFrame) -> pd.Series:
        c = self.contributions(factors)
        if c.empty or "composite" not in c.columns:
            return pd.Series(dtype=float, name="macro_score")
        return (50.0 * np.tanh(c["composite"])).rename("macro_score")

    def confidence(
        self,
        factors: pd.DataFrame,
        regime_prob: float = 50.0,
        data_ok_ratio: float = 1.0,
    ) -> float:
        if factors.empty:
            return 0.0
        latest = factors.iloc[-1]
        expected = [s.name for s in self.fe.specs]
        present = [c for c in expected if c in latest.index and pd.notna(latest[c])]
        coverage = len(present) / max(len(expected), 1)
        if len(present) >= 2:
            vals = latest[present].astype(float)
            dispersion = float(vals.std())
            agreement = float(np.clip(1.0 - dispersion / 3.0, 0.0, 1.0))
        else:
            agreement = 0.3
        regime_c = float(np.clip(regime_prob / 100.0, 0.0, 1.0))
        data_c = float(np.clip(data_ok_ratio, 0.0, 1.0))
        conf = 100.0 * (0.30 * coverage + 0.30 * agreement + 0.25 * regime_c + 0.15 * data_c)
        return float(np.clip(conf, 0.0, 99.0))

    def latest(self, factors: pd.DataFrame, regime_prob: float = 50.0, data_ok_ratio: float = 1.0) -> Dict[str, Any]:
        if factors.empty:
            return {
                "macro_score": np.nan,
                "confidence": 0.0,
                "contributions": {},
                "z_latest": {},
                "n_factors": 0,
            }
        state = self.fe.latest_state(factors)
        score_s = self.score_series(factors)
        score = float(score_s.iloc[-1]) if len(score_s) else np.nan
        conf = self.confidence(factors, regime_prob=regime_prob, data_ok_ratio=data_ok_ratio)
        contrib = state.get("contributions", {})
        return {
            "macro_score": score,
            "confidence": conf,
            "contributions": contrib,
            "z_latest": state.get("z_latest", {}),
            "n_factors": len(contrib),
            "score_series": score_s,
            "regime_hint": state.get("regime"),
            "methodology": (
                "score = 50*tanh(sum_i w_i * z_i); "
                "w signed+normalized; confidence = 0.3*coverage + 0.3*agreement + 0.25*regime_p + 0.15*data_ok"
            ),
        }
