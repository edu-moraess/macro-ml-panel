"""Yield Curve Intelligence Engine — real FRED Treasury vertices only.

Gaussian Process is used for curve construction/smoothing, NOT as a causal
macro forecast model.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

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
    """Load the latest available real Treasury yield at each FRED vertex."""
    points: Dict[float, float] = {}
    for mat, code in FRED_TREASURY.items():
        series = get_fred(code).dropna()
        if not series.empty:
            points[mat] = float(series.iloc[-1])
    if len(points) < 3:
        raise ValueError("AMOSTRA INSUFICIENTE — menos de 3 vértices Treasury disponíveis")
    curve = pd.Series(points, name="yield_pct").sort_index()
    curve.index.name = "maturity_years"
    return curve


def load_history(codes: Optional[Dict[float, str]] = None) -> pd.DataFrame:
    """Load monthly historical Treasury vertices from real FRED observations."""
    codes = codes or {2.0: "DGS2", 10.0: "DGS10", 30.0: "DGS30", 0.25: "DGS3MO"}
    frames: Dict[float, pd.Series] = {}
    for mat, code in codes.items():
        series = get_fred(code).dropna()
        if not series.empty:
            frames[mat] = series.resample("MS").last()
    if not frames:
        raise RuntimeError("DADOS INDISPONÍVEIS — nenhum vértice histórico carregado")
    return pd.DataFrame(frames).dropna(how="all")


def curve_factors(curve: pd.Series) -> Dict[str, float]:
    """Calculate level, slope and curvature factors from observed curve vertices."""
    def y(maturity: float) -> Optional[float]:
        return float(curve[maturity]) if maturity in curve.index and pd.notna(curve[maturity]) else None

    y2, y10, y30, y3m = y(2.0), y(10.0), y(30.0), y(0.25)
    vals = [float(v) for v in curve.values if pd.notna(v)]
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
    """Classify the observed curve using documented spread thresholds."""
    slope = factors.get("slope_10y2y_bps")
    long_end = factors.get("slope_30y10y_bps")
    n = int(factors.get("n_vertices", 0))
    if slope is None or (isinstance(slope, float) and np.isnan(slope)):
        return {"regime": "DATA UNAVAILABLE", "confidence": 0.0, "primary_spread_bps": np.nan}
    if slope < 0:
        regime, confidence = "INVERTED", min(95.0, 55 + abs(slope) / 2)
    elif slope < 50:
        regime, confidence = "FLATTENING", min(90.0, 50 + (50 - slope) / 2)
    elif slope < 150:
        regime, confidence = "NORMAL", min(85.0, 55 + (150 - slope) / 5)
    else:
        regime, confidence = "STEEPENING", min(95.0, 55 + (slope - 150) / 3)
    if long_end is not None and not (isinstance(long_end, float) and np.isnan(long_end)) and long_end < -10:
        regime = "LONG-END PRESSURE"
        confidence = min(95.0, confidence + 5)
    if n < 5:
        confidence = max(40.0, confidence - 15)
    return {
        "regime": regime,
        "confidence": float(confidence),
        "primary_spread_bps": float(slope),
        "long_end_bps": float(long_end) if long_end is not None and not np.isnan(long_end) else np.nan,
    }


def fit_gp(curve: pd.Series) -> Dict[str, Any]:
    """Fit a Gaussian Process to observed maturity/yield points and report diagnostics."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

    X = curve.index.values.reshape(-1, 1).astype(float)
    y_values = curve.values.astype(float)
    if len(y_values) < 3:
        raise ValueError("AMOSTRA INSUFICIENTE para GP (mínimo 3 vértices)")
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=5.0, length_scale_bounds=(0.1, 50.0)) + WhiteKernel(
        noise_level=0.05, noise_level_bounds=(1e-5, 1.0)
    )
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=True, random_state=0)
    gp.fit(X, y_values)
    grid = np.linspace(float(X.min()), float(X.max()), 80).reshape(-1, 1)
    mean, std = gp.predict(grid, return_std=True)
    fitted = gp.predict(X)
    residuals = y_values - fitted
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    length_scale = noise = np.nan
    try:
        optimized = gp.kernel_
        length_scale = float(optimized.k1.k2.length_scale)
        noise = float(optimized.k2.noise_level)
    except Exception:
        pass
    return {
        "gp": gp,
        "grid_maturity": grid.ravel(),
        "mean": mean,
        "std": std,
        "y_hat": fitted,
        "residuals": residuals,
        "rmse": rmse,
        "mae": mae,
        "kernel_str": str(gp.kernel_),
        "length_scale": length_scale,
        "noise": noise,
        "observed_X": X.ravel(),
        "observed_y": y_values,
        "n": len(y_values),
    }


def historical_spreads(hist: pd.DataFrame) -> pd.DataFrame:
    """Build historical Treasury spread series from observed monthly vertices."""
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


def rolling_spread_stats(spreads: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Calculate rolling mean, volatility and z-score for available spread columns."""
    if window < 2:
        raise ValueError("window deve ser >= 2")
    result = pd.DataFrame(index=spreads.index)
    for column in spreads.columns:
        series = pd.to_numeric(spreads[column], errors="coerce")
        mean = series.rolling(window, min_periods=window).mean()
        std = series.rolling(window, min_periods=window).std(ddof=0)
        result[f"{column}_mean"] = mean
        result[f"{column}_vol"] = std
        result[f"{column}_z"] = (series - mean) / std.replace(0.0, np.nan)
    return result


def curve_regime_history(spreads: pd.DataFrame) -> pd.Series:
    """Classify each historical observed spread row using the same curve rules."""
    if spreads.empty:
        return pd.Series(dtype="object", index=spreads.index, name="curve_regime")

    def classify_row(row: pd.Series) -> str:
        slope = row.get("2s10s_bps", np.nan)
        long_end = row.get("10s30s_bps", np.nan)
        if pd.isna(slope):
            return "DATA UNAVAILABLE"
        factors = {
            "slope_10y2y_bps": float(slope),
            "slope_30y10y_bps": float(long_end) if pd.notna(long_end) else np.nan,
            "n_vertices": 5,
        }
        return str(classify_curve_regime(factors)["regime"])

    return spreads.apply(classify_row, axis=1).rename("curve_regime")


def curve_regime_persistence(regimes: pd.Series) -> Dict[str, Any]:
    """Return current curve regime, duration and previous regime."""
    clean = regimes.dropna()
    if clean.empty:
        return {"regime": "DATA UNAVAILABLE", "duration": 0, "previous": "—"}
    current = str(clean.iloc[-1])
    duration = 0
    for value in reversed(clean.tolist()):
        if str(value) != current:
            break
        duration += 1
    previous = "—"
    if len(clean) > duration:
        previous = str(clean.iloc[-duration - 1])
    return {"regime": current, "duration": int(duration), "previous": previous}


def curve_factor_z(hist_spreads: pd.DataFrame, window: int = 60) -> pd.Series:
    """Return rolling z-score of the observed 2s10s spread."""
    from data_utils import rolling_zscore

    if "2s10s_bps" not in hist_spreads.columns:
        return pd.Series(dtype=float)
    return rolling_zscore(hist_spreads["2s10s_bps"], window).rename("Curve")
