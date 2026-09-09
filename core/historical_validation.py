"""Historical Validation Engine — forward outcomes with mandatory lag (anti-lookahead).

CRITICAL: signal(t) is always paired with outcome measured from t+1 onward.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def forward_return(prices_or_rets: pd.Series, horizon: int, is_return: bool = True) -> pd.Series:
    r = prices_or_rets if is_return else prices_or_rets.pct_change()
    one = (1 + r.fillna(0))
    inv = one.iloc[::-1]
    roll = inv.rolling(horizon, min_periods=horizon).apply(np.prod, raw=True)
    fwd = roll.iloc[::-1].shift(-1) - 1.0
    fwd.name = f"fwd_{horizon}"
    return fwd


def max_drawdown_forward(r: pd.Series, horizon: int) -> pd.Series:
    out = pd.Series(index=r.index, dtype=float)
    vals = r.values
    n = len(vals)
    for i in range(n - horizon):
        window = vals[i + 1 : i + 1 + horizon]
        wealth = np.cumprod(1 + np.nan_to_num(window, nan=0.0))
        if len(wealth) == 0:
            out.iloc[i] = np.nan
            continue
        peak = np.maximum.accumulate(wealth)
        dd = (wealth / peak - 1).min()
        out.iloc[i] = dd
    return out


class HistoricalValidationEngine:
    def __init__(self, horizons: Tuple[int, ...] = (1, 3, 6)):
        self.horizons = horizons

    def build_forward_panel(self, signal: pd.Series, asset_returns: pd.Series) -> pd.DataFrame:
        sig = signal.dropna()
        ret = asset_returns.reindex(sig.index).dropna()
        common = sig.index.intersection(ret.index)
        sig = sig.reindex(common)
        ret = ret.reindex(common)
        panel = pd.DataFrame({"signal": sig})
        for h in self.horizons:
            panel[f"fwd_{h}m"] = forward_return(ret, h, is_return=True)
            panel[f"mdd_{h}m"] = max_drawdown_forward(ret, h)
        panel = panel.dropna(subset=[f"fwd_{h}m" for h in self.horizons], how="any")
        return panel

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
            col = f"fwd_{h}m"
            x = sub[col].dropna()
            hit = float((x > 0).mean()) if len(x) else np.nan
            result["horizons"][f"+{h}M"] = {
                "avg": float(x.mean()) if len(x) else np.nan,
                "median": float(x.median()) if len(x) else np.nan,
                "hit_ratio": hit,
                "vol": float(x.std()) if len(x) else np.nan,
                "mdd_avg": float(sub[f"mdd_{h}m"].mean()) if f"mdd_{h}m" in sub else np.nan,
            }
        return result

    def regime_transition_study(
        self,
        regime_labels: pd.Series,
        asset_returns: pd.Series,
        from_state: str,
        to_state: str,
    ) -> Dict[str, Any]:
        labels = regime_labels.dropna()
        prev = labels.shift(1)
        transitions = (prev == from_state) & (labels == to_state)
        dates = labels.index[transitions.fillna(False)]
        if len(dates) < 3:
            return {"status": "AMOSTRA INSUFICIENTE", "n": len(dates), "label": f"{from_state} → {to_state}"}
        panel_rows = []
        ret = asset_returns.reindex(labels.index)
        for t in dates:
            row = {"date": t}
            for h in self.horizons:
                loc = ret.index.get_loc(t)
                if isinstance(loc, slice):
                    continue
                if loc + h >= len(ret):
                    row[f"fwd_{h}m"] = np.nan
                    continue
                window = ret.iloc[loc + 1 : loc + 1 + h]
                row[f"fwd_{h}m"] = float(np.prod(1 + window.fillna(0)) - 1)
            panel_rows.append(row)
        df = pd.DataFrame(panel_rows).dropna(how="all")
        n = len(df)
        out: Dict[str, Any] = {
            "status": "OK" if n >= 3 else "AMOSTRA INSUFICIENTE",
            "n": n,
            "label": f"{from_state} → {to_state}",
            "horizons": {},
        }
        for h in self.horizons:
            col = f"fwd_{h}m"
            if col not in df.columns:
                continue
            x = df[col].dropna()
            out["horizons"][f"+{h}M"] = {
                "avg": float(x.mean()) if len(x) else np.nan,
                "median": float(x.median()) if len(x) else np.nan,
                "hit_ratio": float((x > 0).mean()) if len(x) else np.nan,
            }
        return out
