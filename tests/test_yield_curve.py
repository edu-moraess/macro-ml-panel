"""Yield Curve Intelligence unit tests — no live FRED required for pure logic."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.yield_curve import curve_factors, classify_curve_regime, historical_spreads


def _sample_curve():
    return pd.Series(
        {0.25: 5.2, 2.0: 4.8, 10.0: 4.3, 30.0: 4.5},
        name="yield_pct",
    )


def test_curve_factors_level_slope_curvature():
    f = curve_factors(_sample_curve())
    assert f["n_vertices"] == 4
    assert abs(f["slope_10y2y_bps"] - (4.3 - 4.8) * 100) < 1e-6
    assert abs(f["curvature_bps"] - (4.8 - 2 * 4.3 + 4.5) * 100) < 1e-6
    assert f["level"] > 0


def test_curve_factors_missing_vertex():
    c = pd.Series({2.0: 4.0, 10.0: 4.5})
    f = curve_factors(c)
    assert np.isnan(f["curvature_bps"])
    assert f["slope_10y2y_bps"] == pytest.approx(50.0)


def test_regime_inverted():
    f = {"slope_10y2y_bps": -40.0, "slope_30y10y_bps": 10.0, "n_vertices": 6}
    r = classify_curve_regime(f)
    assert r["regime"] == "INVERTED"
    assert r["confidence"] > 50


def test_regime_steepening():
    f = {"slope_10y2y_bps": 200.0, "slope_30y10y_bps": 30.0, "n_vertices": 8}
    r = classify_curve_regime(f)
    assert r["regime"] == "STEEPING"


def test_regime_long_end_pressure():
    f = {"slope_10y2y_bps": 80.0, "slope_30y10y_bps": -25.0, "n_vertices": 8}
    r = classify_curve_regime(f)
    assert r["regime"] == "LONG-END PRESSURE"


def test_historical_spreads():
    idx = pd.date_range("2020-01-01", periods=12, freq="MS")
    hist = pd.DataFrame({2.0: np.linspace(1, 2, 12), 10.0: np.linspace(1.5, 2.2, 12), 0.25: 0.5}, index=idx)
    sp = historical_spreads(hist)
    assert "2s10s_bps" in sp.columns
    assert sp["2s10s_bps"].iloc[-1] == pytest.approx((2.2 - 2.0) * 100)


def test_insufficient_curve():
    f = curve_factors(pd.Series(dtype=float))
    assert f["n_vertices"] == 0
