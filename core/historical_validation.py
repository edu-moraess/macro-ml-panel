"""Historical Validation Engine — forward outcomes with mandatory lag.

Signal(t) is paired only with outcomes measured from t+1 onward.
Missing observations are preserved as missing; they are never treated as zero.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def _forward_compound_returns(r: pd.Series, horizon: int) -> pd.Series:
    """Compound the next ``horizon`` observations only when all are present."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    values = pd.to_numeric(r, errors="coerce")
    out = pd.Series(np.nan, index=values.index, dtype=float)
    for i in range(len(values) - horizon):
        window = values.iloc[i + 1 : i + 1 + horizon]
        if len(window) == horizon and window.notna().all():
            out.iloc[i] = float(np.prod(1.0 + window.to_numpy(dtype=float)) - 1.0)
    return out


def forward_return(prices_or_rets: pd.Series, horizon: int, is_return: bool = True) -> pd.Series:
    """Return from t+1 through t+h, excluding incomplete forward windows."""
    r = pd.to_numeric(prices_or_rets, errors="coerce")
    if not is_return:
        r = r.pct_change()
    out = _forward_compound_returns(r, horizon)
    out.name = f"fwd_{horizon}"
    return out


def max_drawdown_forward(r: pd.Series, horizon: int) -> pd.Series:
    """Maximum drawdown over the next h observations; incomplete windows are NaN."""
    values = pd.to_numeric(r, errors="coerce")
    out = pd.Series(np.nan, index=values.index, dtype=float)
    for i in range(len(values) - horizon):
        window = values.iloc[i + 1 : i + 1 + horizon]
        if len(window) != horizon or window.isna().any():
            continue
        wealth = np.cumprod(1.0 + window.to_numpy(dtype=float))
        peak = np.maximum.accumulate(wealth)
        out.iloc[i] = float(np.min(wealth / peak - 1.0))
    return out


class HistoricalValidationEngine:
    def __init__(self, horizons: Tuple[int, ...] = (1, 3, 6)):
        if not horizons or any(h < 1 for h in horizons):
            raise ValueError("horizons must contain positive integers")
        self.horizons = tuple(horizons)

    def build_forward_panel(self, signal: pd.Series, asset_returns: pd.Series) -> pd.DataFrame:
        sig = pd.to_numeric(signal, errors="coerce").dropna()
        ret = pd.to_numeric(asset_returns, errors="coerce").reindex(sig.index)
        panel = pd.DataFrame({"signal": sig, "asset_return": ret})
        for h in self.horizons:
            panel[f"fwd_{h}m"] = forward_return(ret, h, is_return=True)
            panel[f"mdd_{h}m"] = max_drawdown_forward(ret, h)
        required = [f"fwd_{h}m" for h in self.horizons]
        return panel.dropna(subset=required, how="any")

    def event_study(self, panel: pd.DataFrame, condition: str, threshold: float = -40.0) -> Dict[str, Any]:
        if panel.empty or "signal" not in panel.columns:
            return {"status": "AMOSTRA INSUFICIENTE", "n": 0}
        s = panel["signal"]
        if condition == "lt":
            mask = s < threshold
            label = f"MACRO SCORE < {threshold}"
        elif condition == "gt":
            mask = s > threshold
            label = f"MACRO SCORE > {threshold}"
        else:
            mask = s.abs() > abs(threshold)
            label = f"|MACRO SCORE| > {abs(threshold)}"
        sub = panel.loc[mask]
        n = len(sub)
        if n < 5:
            return {"status": "AMOSTRA INSUFICIENTE", "n": n, "label": label}
        result: Dict[str, Any] = {"status": "OK", "n": n, "label": label, "horizons": {}}
        for h in self.horizons:
            x = sub[f"fwd_{h}m"].dropna()
            result["horizons"][f"+{h}M"] = {
                "avg": float(x.mean()) if len(x) else np.nan,
                "median": float(x.median()) if len(x) else np.nan,
                "hit_ratio": float((x > 0).mean()) if len(x) else np.nan,
                "vol": float(x.std()) if len(x) > 1 else np.nan,
                "mdd_avg": float(sub[f"mdd_{h}m"].mean()) if f"mdd_{h}m" in sub else np.nan,
            }
        return result

    def regime_transition_study(self, regime_labels: pd.Series, asset_returns: pd.Series, from_state: str, to_state: str) -> Dict[str, Any]:
        labels = regime_labels.dropna()
        prev = labels.shift(1)
        transitions = (prev == from_state) & (labels == to_state)
        dates = labels.index[transitions.fillna(False)]
        if len(dates) < 3:
            return {"status": "AMOSTRA INSUFICIENTE", "n": len(dates), "label": f"{from_state} → {to_state}"}
        ret = pd.to_numeric(asset_returns, errors="coerce").reindex(labels.index)
        rows = []
        for t in dates:
            loc = ret.index.get_indexer([t])[0]
            if loc < 0:
                continue
            row = {"date": t}
            for h in self.horizons:
                window = ret.iloc[loc + 1 : loc + 1 + h]
                row[f"fwd_{h}m"] = (
                    float(np.prod(1.0 + window.to_numpy(dtype=float)) - 1.0)
                    if len(window) == h and window.notna().all()
                    else np.nan
                )
            rows.append(row)
        df = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
        out: Dict[str, Any] = {"status": "OK" if len(df) >= 3 else "AMOSTRA INSUFICIENTE", "n": len(df), "label": f"{from_state} → {to_state}", "horizons": {}}
        for h in self.horizons:
            x = df.get(f"fwd_{h}m", pd.Series(dtype=float)).dropna()
            out["horizons"][f"+{h}M"] = {
                "avg": float(x.mean()) if len(x) else np.nan,
                "median": float(x.median()) if len(x) else np.nan,
                "hit_ratio": float((x > 0).mean()) if len(x) else np.nan,
            }
        return out
