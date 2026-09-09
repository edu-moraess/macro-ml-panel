"""Yield Curve Intelligence Engine — real FRED Treasury vertices only.

Gaussian Process is used for curve construction/smoothing, NOT as a causal
macro forecast model.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from data_utils import get_fred

FRED_TREASURY: Dict[float, str] = {
    0.25: "DGS3MO",
    0.5: "DGS6MO",
    1.0: "DGS1",
    2.0: "DGS2",
    3.0: "DGS3",
    5.0: "DGS5",
    7.0: "DGS7",
    10.0: "DGS10",
    20.0: "DGS20",
    30.0: "DGS30",
}


def load_latest_curve() -> pd.Series:
    points = {}
    for mat, code in FRED_TREASURY.items():
        s = get_fred(code).dropna()
        if s.empty:
            continue
        points[mat] = float(s.iloc[-1])
    if len(points) < 3:
        raise ValueError("AMOSTRA INSUFICIENTE — menos de 3 vértices Treasury disponíveis")
    curva = pd.Series(points, name="yield_pct").sort_index()
    curva.index.name = "maturity_years"
    return curva


def load_history(codes: Optional[Dict[float, str]] = None) -> pd.DataFrame:
    codes = codes or {2.0: "DGS2", 10.0: "DGS10", 30.0: "DGS30", 0.25: "DGS3MO"}
    frames = {}
    for mat, code in codes.items():
        s = get_fred(code).dropna()
        if not s.empty:
            frames[mat] = s.resample("MS").last()
    if not frames:
        raise RuntimeError("DADOS INDISPONÍVEIS — nenhum vértice histórico carregado")
    return pd.DataFrame(frames).dropna(how="all")


def curve_factors(curve: pd.Series) -> Dict[str, float]:
    def y(m: float):
        return float(curve[m]) if m in curve.index and pd.notna(curve[m]) else None

    y2, y10, y30, y3m = y(2.0), y(10.0), y(30.0), y(0.25)
    vals = [v for v in curve.values if pd.notna(v)]
    level = float(np.mean(vals)) if vals else np.nan
    slope_10_2 = (y10 - y2) * 100 if y10 is not None and y2 is not None else np.nan
    slope_10_3m = (y10 - y3m) * 100 if y10 is not None and y3m is not None else np.nan
    slope_30_10 = (y30 - y10) * 100 if y30 is not None and y10 is not None else np.nan
    curvature = (y2 - 2 * y10 + y30) * 100 if None not in (y2, y10, y30) else np.nan
    return {
        "level": level,
        "slope_10y2y_bps": slope_10_2,
        "slope_10y3m_bps": slope_10_3m,
        "slope_30y10y_bps": slope_30_10,
        "curvature_bps": curvature,
        "n_vertices": len(vals),
    }


def classify_curve_regime(factors: Dict[str, float]) -> Dict[str, Any]:
    s = factors.get("slope_10y2y_bps")
    long_end = factors.get("slope_30y10y_bps")
    n = factors.get("n_vertices", 0)
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return {"regime": "DATA UNAVAILABLE", "confidence": 0.0, "primary_spread_bps": np.nan}
    if s < 0:
        regime, conf = "INVERTED", min(95.0, 55 + abs(s) / 2)
    elif s < 50:
        regime, conf = "FLATTENING", min(90.0, 50 + (50 - s) / 2)
    elif s < 150:
        regime, conf = "NORMAL", min(85.0, 55 + (150 - s) / 5)
    else:
        regime, conf = "STEEPING", min(95.0, 55 + (s - 150) / 3)
    if long_end is not None and not (isinstance(long_end, float) and np.isnan(long_end)) and long_end < -10:
        regime = "LONG-END PRESSURE"
        conf = min(95.0, conf + 5)
    if n < 5:
        conf = max(40.0, conf - 15)
    return {
        "regime": regime,
        "confidence": float(conf),
        "primary_spread_bps": float(s),
        "long_end_bps": float(long_end) if long_end is not None and not np.isnan(long_end) else np.nan,
    }


def fit_gp(curve: pd.Series) -> Dict[str, Any]:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

    X = curve.index.values.reshape(-1, 1).astype(float)
    y = curve.values.astype(float)
    if len(y) < 3:
        raise ValueError("AMOSTRA INSUFICIENTE para GP (mínimo 3 vértices)")
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=5.0, length_scale_bounds=(0.1, 50.0)) + WhiteKernel(
        noise_level=0.05, noise_level_bounds=(1e-5, 1.0)
    )
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=True, random_state=0)
    gp.fit(X, y)
    grid = np.linspace(float(X.min()), float(X.max()), 80).reshape(-1, 1)
    mean, std = gp.predict(grid, return_std=True)
    y_hat = gp.predict(X)
    resid = y - y_hat
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    length_scale = noise = np.nan
    try:
        k = gp.kernel_
        length_scale = float(k.k1.k2.length_scale)
        noise = float(k.k2.noise_level)
    except Exception:
        k = gp.kernel_
    return {
        "gp": gp,
        "grid_maturity": grid.ravel(),
        "mean": mean,
        "std": std,
        "y_hat": y_hat,
        "residuals": resid,
        "rmse": rmse,
        "mae": mae,
        "kernel_str": str(gp.kernel_),
        "length_scale": length_scale,
        "noise": noise,
        "observed_X": X.ravel(),
        "observed_y": y,
        "n": len(y),
    }


def historical_spreads(hist: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=hist.index)
    if 10.0 in hist.columns and 2.0 in hist.columns:
        out["2s10s_bps"] = (hist[10.0] - hist[2.0]) * 100
    if 10.0 in hist.columns and 0.25 in hist.columns:
        out["3m10y_bps"] = (hist[10.0] - hist[0.25]) * 100
    if 30.0 in hist.columns and 10.0 in hist.columns:
        out["10s30s_bps"] = (hist[30.0] - hist[10.0]) * 100
    if 10.0 in hist.columns:
        out["level_10y"] = hist[10.0]
    return out.dropna(how="all")


def curve_factor_z(hist_spreads: pd.DataFrame, window: int = 60) -> pd.Series:
    from data_utils import rolling_zscore
    if "2s10s_bps" not in hist_spreads.columns:
        return pd.Series(dtype=float)
    return rolling_zscore(hist_spreads["2s10s_bps"], window).rename("Curve")
