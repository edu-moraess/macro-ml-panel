"""Risk Engine V2 — realized metrics on real returns only."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


class RiskEngine:
    @staticmethod
    def realized_vol(r: pd.Series, window: int = 20, ann: int = 12) -> pd.Series:
        return r.rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(ann)

    @staticmethod
    def ewma_vol(r: pd.Series, lam: float = 0.94, ann: int = 12) -> pd.Series:
        return r.pow(2).ewm(alpha=1 - lam, adjust=False).mean().pow(0.5) * np.sqrt(ann)

    @staticmethod
    def historical_var(r: pd.Series, alpha: float = 0.05) -> float:
        if r.dropna().empty:
            return np.nan
        return float(-r.quantile(alpha))

    @staticmethod
    def expected_shortfall(r: pd.Series, alpha: float = 0.05) -> float:
        if r.dropna().empty:
            return np.nan
        q = r.quantile(alpha)
        tail = r[r <= q]
        return float(-tail.mean()) if len(tail) else np.nan

    @staticmethod
    def max_drawdown(r: pd.Series) -> float:
        wealth = (1 + r.fillna(0)).cumprod()
        if wealth.empty:
            return np.nan
        return float((wealth / wealth.cummax() - 1).min())

    @staticmethod
    def sharpe(r: pd.Series, ann: int = 12) -> float:
        s = r.dropna()
        if len(s) < 5 or s.std() == 0:
            return np.nan
        return float(s.mean() / s.std() * np.sqrt(ann))

    @staticmethod
    def sortino(r: pd.Series, ann: int = 12) -> float:
        s = r.dropna()
        downside = s[s < 0]
        if len(downside) < 5 or downside.std() == 0:
            return np.nan
        return float(s.mean() / downside.std() * np.sqrt(ann))

    @staticmethod
    def calmar(r: pd.Series, ann: int = 12) -> float:
        mdd = RiskEngine.max_drawdown(r)
        if mdd >= 0 or np.isnan(mdd):
            return np.nan
        return float(r.dropna().mean() * ann / abs(mdd))

    @staticmethod
    def rolling_sharpe(r: pd.Series, window: int = 36, ann: int = 12) -> pd.Series:
        def _sh(x):
            if len(x) < 10 or x.std() == 0:
                return np.nan
            return x.mean() / x.std() * np.sqrt(ann)
        return r.rolling(window, min_periods=max(12, window // 2)).apply(_sh, raw=True)

    @staticmethod
    def beta(asset: pd.Series, market: pd.Series) -> float:
        df = pd.concat([asset, market], axis=1).dropna()
        if len(df) < 20:
            return np.nan
        cov = np.cov(df.iloc[:, 0], df.iloc[:, 1])
        return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else np.nan

    @staticmethod
    def rolling_corr(a: pd.Series, b: pd.Series, window: int = 36) -> pd.Series:
        return a.rolling(window).corr(b)

    @classmethod
    def summary(cls, r: pd.Series, lam: float = 0.94) -> Dict:
        r = r.dropna()
        if len(r) < 24:
            return {"status": "AMOSTRA INSUFICIENTE", "n": len(r)}
        vol = cls.realized_vol(r)
        ewma = cls.ewma_vol(r, lam)
        return {
            "status": "OK",
            "n": len(r),
            "vol_20": float(vol.iloc[-1]) if len(vol) else np.nan,
            "ewma_vol": float(ewma.iloc[-1]) if len(ewma) else np.nan,
            "var_95": cls.historical_var(r, 0.05),
            "var_99": cls.historical_var(r, 0.01),
            "es_95": cls.expected_shortfall(r, 0.05),
            "es_99": cls.expected_shortfall(r, 0.01),
            "max_dd": cls.max_drawdown(r),
            "sharpe": cls.sharpe(r),
            "sortino": cls.sortino(r),
            "calmar": cls.calmar(r),
            "vol_series": vol,
            "ewma_series": ewma,
            "dd_series": (1 + r).cumprod() / (1 + r).cumprod().cummax() - 1,
        }
