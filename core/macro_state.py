"""Canonical macro state pipeline used by the dashboard modules.

BCB/FRED observations -> raw macro features -> FactorEngine -> SignalEngine -> RegimeEngine.
All consumers receive the same factor, signal and regime definitions.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from data_utils import FRED, SGS, align, get_bcb, get_fred
from core.factors import FactorEngine
from core.model_cache import fit_regime_engine
from core.signals import SignalEngine


@st.cache_data(ttl=900, show_spinner=False)
def load_macro_panel() -> pd.DataFrame:
    """Load and cache the real BCB/FRED monthly panel for 15 minutes."""
    return align(
        get_bcb(SGS["ibc_br"]).resample("MS").mean().rename("IBC"),
        get_bcb(SGS["desemprego_pnad"]).resample("MS").mean().rename("U"),
        get_bcb(SGS["ipca_mensal"]).resample("MS").mean().rename("IPCA"),
        get_bcb(SGS["selic_meta"]).resample("MS").mean().rename("Selic"),
        get_bcb(SGS["cambio_ptax"]).resample("MS").mean().rename("FX"),
        get_fred(FRED["t10y3m"]).resample("MS").mean().rename("TermSpread"),
        get_fred(FRED["vix"]).resample("MS").mean().rename("VIX"),
        get_fred(FRED["hy_spread"]).resample("MS").mean().rename("HY"),
        get_fred(FRED["sp500"]).resample("MS").last().rename("SP500"),
    )


def build_macro_state(panel: pd.DataFrame, window: int = 60) -> dict[str, Any]:
    """Build canonical factors, signal and statistical regime from one panel."""
    if panel is None or panel.empty:
        raise ValueError("DADOS INDISPONÍVEIS para Macro State")

    raw = pd.DataFrame(index=panel.index)
    raw["growth"] = panel["IBC"].pct_change(3) * 100 - panel["U"].diff(3)
    raw["inflation"] = panel["IPCA"].diff(3)
    raw["rates"] = panel["Selic"].diff(3)
    raw["fx"] = -panel["FX"].pct_change(3) * 100
    raw["liquidity"] = -(panel["HY"] + panel["VIX"])
    raw["momentum"] = panel["SP500"].pct_change(12) * 100
    raw["curve"] = panel["TermSpread"]

    fe = FactorEngine(window=window)
    factors = fe.build(
        raw["growth"], raw["inflation"], raw["rates"], raw["fx"],
        raw["liquidity"], raw["momentum"], curve_raw=raw["curve"]
    )
    if len(factors) < max(24, window // 2):
        raise ValueError("AMOSTRA INSUFICIENTE para Macro State")

    se = SignalEngine(fe)
    regime_info: dict[str, Any] = {
        "regime": "N/A", "probability": 50.0, "method": "none", "probabilities": {}
    }
    regime_cols = [c for c in ["Growth", "Inflation", "Liquidity", "Momentum", "Curve"] if c in factors]
    try:
        regime = fit_regime_engine(factors[regime_cols].dropna(), n_regimes=3, random_state=42)
        regime_info = regime.current()
    except ValueError:
        raise
    except Exception as exc:
        regime_info["error"] = str(exc)

    signal = se.latest(
        factors,
        regime_prob=float(regime_info.get("probability", 50.0)),
        data_ok_ratio=1.0,
    )
    return {
        "panel": panel,
        "raw": raw,
        "factors": factors,
        "signal": signal,
        "regime": regime_info,
        "score_series": signal["score_series"],
    }
