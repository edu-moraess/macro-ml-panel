"""Macro Intelligence tests — signal, regime persistence and validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.signals import SignalEngine
from core.factors import FactorEngine
from core.regimes import RegimeEngine
from core.historical_validation import HistoricalValidationEngine, forward_return
from core.yield_curve import rolling_spread_stats, curve_regime_persistence


def _factor_panel(n=80, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="MS")
    raw = [pd.Series(rng.normal(0, 1, n), index=idx) for _ in range(6)]
    fe = FactorEngine(window=36)
    return fe.build(*raw)


def test_signal_score_and_confidence():
    f = _factor_panel()
    out = SignalEngine().latest(f, regime_prob=70.0, data_ok_ratio=1.0)
    assert "macro_score" in out
    assert 0 <= out["confidence"] <= 99
    assert out["n_factors"] >= 1


def test_contribution_keys():
    f = _factor_panel()
    c = SignalEngine().contributions(f)
    assert "composite" in c.columns
    fac_cols = [c for c in c.columns if c != "composite"]
    assert (c[fac_cols].sum(axis=1) - c["composite"]).abs().max() < 1e-8


def test_regime_proba_sum():
    f = _factor_panel(n=100)
    re = RegimeEngine(n_regimes=3)
    re.fit(f[["Growth", "Inflation", "Liquidity", "Momentum"]].dropna())
    st = re.current()
    assert abs(sum(st["probabilities"].values()) - 100.0) < 1.5
    assert st["method"] in ("HMM", "GMM")
    assert st["duration_months"] >= 1
    assert "transition" in st


def test_regime_insufficient():
    with pytest.raises(ValueError):
        RegimeEngine(n_regimes=3).fit(pd.DataFrame(np.random.randn(10, 2)))


def test_forward_return_lag():
    idx = pd.date_range("2020-01-01", periods=12, freq="MS")
    valid = forward_return(pd.Series(0.01, index=idx), horizon=3).dropna()
    assert len(valid) >= 1
    assert abs(valid.iloc[0] - (1.01 ** 3 - 1)) < 1e-9


def test_event_study_insufficient():
    panel = pd.DataFrame({"signal":[0.0]*3,"fwd_1m":[0.01]*3,"fwd_3m":[0.02]*3,"fwd_6m":[0.03]*3,"mdd_1m":[-0.01]*3,"mdd_3m":[-0.01]*3,"mdd_6m":[-0.01]*3})
    assert HistoricalValidationEngine().event_study(panel, "lt", threshold=-40)["status"] == "AMOSTRA INSUFICIENTE"


def test_event_study_ok():
    rng = np.random.default_rng(1)
    n = 80
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    panel = HistoricalValidationEngine().build_forward_panel(pd.Series(rng.normal(0, 50, n), index=idx), pd.Series(rng.normal(0.005, 0.04, n), index=idx))
    out = HistoricalValidationEngine().event_study(panel, "lt", threshold=0.0)
    assert out["n"] >= 5 or out["status"] == "AMOSTRA INSUFICIENTE"
    if out["status"] == "OK":
        assert "+1M" in out["horizons"]


def test_curve_rolling_and_persistence():
    idx = pd.date_range("2018-01-01", periods=40, freq="MS")
    spreads = pd.DataFrame({"2s10s_bps": np.linspace(-50, 100, 40), "10s30s_bps": np.linspace(20, -30, 40)}, index=idx)
    assert any("z" in c for c in rolling_spread_stats(spreads, window=12).columns)
    regimes = pd.Series(["NORMAL"] * 10 + ["STEEPING"] * 5, index=idx[:15])
    p = curve_regime_persistence(regimes)
    assert p["regime"] == "STEEPING"
    assert p["duration"] == 5
