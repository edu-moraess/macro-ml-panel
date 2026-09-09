"""Unit tests for core quant engines using deterministic fixtures."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.factors import FactorEngine
from core.risk import RiskEngine
from core.backtest import BacktestEngine
from core.portfolio import PortfolioEngine
from core.regimes import RegimeEngine
from core.historical_validation import forward_return, max_drawdown_forward
from data_utils import rolling_zscore


def _returns(n=120, seed=0):
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
    r = _returns(100)
    s = RiskEngine.summary(r)
    assert s["status"] == "OK"
    assert s["var_95"] >= 0
    assert s["max_dd"] <= 0
    assert s["annualization"] == 12


def test_sortino_uses_downside_deviation():
    r = pd.Series([0.10, -0.05, 0.02, -0.01, 0.03, 0.01], index=pd.date_range("2020-01-01", periods=6, freq="MS"))
    value = RiskEngine.sortino(r, ann=12, mar=0.0)
    downside = np.sqrt(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2))
    expected = r.mean() / downside * np.sqrt(12)
    assert value == pytest.approx(expected)


def test_backtest_lag_enforced():
    with pytest.raises(ValueError):
        BacktestEngine(lag=0)
    sig = _returns(80)
    ret = _returns(80, seed=2)
    out = BacktestEngine(lag=1, threshold=0.1).run(sig, ret)
    assert out["lag"] == 1
    assert "equity" in out
    assert out["exposed_periods"] <= len(out["positions"])


def test_backtest_does_not_convert_missing_returns_to_zero():
    idx = pd.date_range("2020-01-01", periods=6, freq="MS")
    sig = pd.Series([1, 1, 1, 1, 1, 1], index=idx, dtype=float)
    ret = pd.Series([0.01, np.nan, 0.02, 0.01, 0.01, 0.01], index=idx)
    out = BacktestEngine(lag=1, threshold=0.0).run(sig, ret)
    assert out["strat_returns"].notna().all()
    assert len(out["strat_returns"]) == 4


def test_forward_return_requires_complete_window():
    idx = pd.date_range("2020-01-01", periods=6, freq="MS")
    r = pd.Series([0.01, 0.02, np.nan, 0.01, 0.02, 0.01], index=idx)
    fwd = forward_return(r, 2)
    assert pd.isna(fwd.iloc[0])
    assert fwd.iloc[3] == pytest.approx((1.01 * 1.02) - 1.0)


def test_forward_drawdown_requires_complete_window():
    idx = pd.date_range("2020-01-01", periods=5, freq="MS")
    r = pd.Series([0.01, -0.02, np.nan, 0.03, 0.01], index=idx)
    mdd = max_drawdown_forward(r, 2)
    assert pd.isna(mdd.iloc[0])


def test_portfolio_weights_sum():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2016-01-01", periods=60, freq="MS")
    assets = pd.DataFrame(rng.normal(0.005, 0.03, (60, 3)), index=idx, columns=["A", "B", "C"])
    for m in ["equal", "inverse_vol", "min_var", "risk_parity"]:
        pe = PortfolioEngine(method=m, max_weight=0.7, min_weight=0.05)
        w = pe.weights(assets)
        assert abs(w.sum() - 1.0) < 1e-6
        assert (w >= -1e-9).all()
        assert (w <= 0.7 + 1e-9).all()


def test_regime_insufficient_sample():
    X = pd.DataFrame(np.random.randn(10, 2))
    re = RegimeEngine(n_regimes=3)
    with pytest.raises(ValueError):
        re.fit(X)
