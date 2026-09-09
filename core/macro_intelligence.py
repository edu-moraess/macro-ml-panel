"""Macro Intelligence orchestrator — Features → Factors → Curve → Regime → Signal → Validation."""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from core.factors import FactorEngine
from core.model_cache import fit_regime_engine
from core.signals import SignalEngine
from core.historical_validation import HistoricalValidationEngine
from core.lineage import build_lineage
from core.yield_curve import load_history, historical_spreads, curve_regime_history, curve_regime_persistence


class MacroIntelligence:
    def __init__(self, window: int = 60):
        self.window = window
        self.fe = FactorEngine(window=window)
        self.se = SignalEngine(self.fe)
        self.hve = HistoricalValidationEngine(horizons=(1, 3, 6))

    def run(self, growth: pd.Series, inflation: pd.Series, rates: pd.Series, fx: pd.Series,
            liquidity: pd.Series, momentum: pd.Series, asset_returns: pd.Series,
            curve_raw: Optional[pd.Series] = None, data_ok_ratio: float = 1.0) -> Dict[str, Any]:
        factors = self.fe.build(growth, inflation, rates, fx, liquidity, momentum, curve_raw=curve_raw)
        if factors.empty or len(factors) < self.window // 2:
            raise ValueError("AMOSTRA INSUFICIENTE para Macro Intelligence")

        regime_info: Dict[str, Any] = {"regime": "N/A", "probability": 0.0, "method": "none", "probabilities": {}}
        hist_labeled = None
        try:
            cols = [c for c in ["Growth", "Inflation", "Liquidity", "Momentum"] if c in factors.columns]
            re = fit_regime_engine(factors[cols].dropna(), n_regimes=3, random_state=42)
            regime_info = re.current()
            hist_labeled = re.history_labeled()
        except Exception as exc:
            regime_info["error"] = str(exc)

        sig = self.se.latest(factors, regime_prob=float(regime_info.get("probability", 50.0)), data_ok_ratio=data_ok_ratio)

        curve_pers = {"regime": "DATA UNAVAILABLE", "duration": 0, "previous": "—"}
        try:
            hist = load_history()
            spreads = historical_spreads(hist)
            curve_pers = curve_regime_persistence(curve_regime_history(spreads))
        except Exception:
            pass

        validation: Dict[str, Any] = {"status": "SKIPPED"}
        try:
            panel = self.hve.build_forward_panel(sig["score_series"], asset_returns)
            validation = {
                "status": "OK",
                "n_panel": len(panel),
                "event_risk_off": self.hve.event_study(panel, "lt", threshold=-40.0),
                "event_risk_on": self.hve.event_study(panel, "gt", threshold=40.0),
            }
            if hist_labeled is not None and len(hist_labeled) > 20:
                validation["transition_off_to_on"] = self.hve.regime_transition_study(hist_labeled, asset_returns, "RISK-OFF", "RISK-ON")
                validation["transition_on_to_off"] = self.hve.regime_transition_study(hist_labeled, asset_returns, "RISK-ON", "RISK-OFF")
        except Exception as exc:
            validation = {"status": "ERROR", "detail": str(exc)}

        lineage = build_lineage(
            inputs=[
                {"name": "growth", "source": "BCB/SGS", "series_id": "24363+24369"},
                {"name": "inflation", "source": "BCB/SGS", "series_id": "433"},
                {"name": "rates", "source": "BCB/SGS", "series_id": "432"},
                {"name": "fx", "source": "BCB/SGS", "series_id": "1"},
                {"name": "liquidity", "source": "FRED", "series_id": "BAMLH0A0HYM2+VIXCLS"},
                {"name": "momentum", "source": "FRED", "series_id": "SP500"},
                {"name": "curve", "source": "FRED", "series_id": "DGS3MO+DGS2+DGS10+DGS30"},
            ], factors=factors, regime=regime_info, signal=sig,
            validation=validation, curve_persistence=curve_pers,
        )
        return {"factors": factors, "regime": regime_info, "signal": sig,
                "curve_persistence": curve_pers, "validation": validation,
                "history_labeled": hist_labeled, "lineage": lineage}
