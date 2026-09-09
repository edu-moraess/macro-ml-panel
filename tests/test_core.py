"""Tests for core quant engines — real logic, no synthetic market data required."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.factors import FactorEngine
from core.risk import RiskEngine
from core.backtest import BacktestEngine
from core.portfolio import PortfolioEngine
from core.regimes import RegimeEngine
from data_utils import rolling_zscore


def _synth_returns(n=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="MS")
    return pd.Series(rng.normal(0.005, 0.04, n), index=idx)


def test_rolling_zscore_finite():
    s = pd.Series(np.arange(100, dtype=float), index=pd.date_range("2010-01-01", periods=100, freq="MS"))
    z = rolling_zscore(s, 24)
    assert z.notna().sum() > 50


def test_factor_engine_shapes():
    n = 80
    idx = pd.date_range("2018-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(1)
    raw = {k: pd.Series(rng.normal(0, 1, n), index=idx) for k in range(6)}
    fe = FactorEngine(window=36)
    f = fe.build(raw[0], raw[1], raw[2], raw[3], raw[4], raw[5])
    assert not f.empty
    state = fe.latest_state(f)
    assert "score" in state and "regime" in state


def test_risk_summary():
    r = _synth_returns(100)
    s = RiskEngine.summary(r)
    assert s["status"] == "OK"
    assert s["var_95"] >= 0
    assert s["max_dd"] <= 0


def test_backtest_lag_enforced():
    with pytest.raises(ValueError):
        BacktestEngine(lag=0)
    sig = _synth_returns(80)
    ret = _synth_returns(80, seed=2)
    bt = BacktestEngine(lag=1, threshold=0.1)
    out = bt.run(sig, ret)
    assert out["lag"] == 1
    assert "equity" in out


def test_portfolio_weights_sum():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2016-01-01", periods=60, freq="MS")
    assets = pd.DataFrame(rng.normal(0.005, 0.03, (60, 3)), index=idx, columns=["A", "B", "C"])
    for m in ["equal", "inverse_vol", "min_var", "risk_parity"]:
        pe = PortfolioEngine(method=m, max_weight=0.7, min_weight=0.05)
        w = pe.weights(assets)
        assert abs(w.sum() - 1.0) < 1e-6
        assert (w >= -1e-9).all()


def test_regime_insufficient_sample():
    X = pd.DataFrame(np.random.randn(10, 2))
    re = RegimeEngine(n_regimes=3)
    with pytest.raises(ValueError):
        re.fit(X)
