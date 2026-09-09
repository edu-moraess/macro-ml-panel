"""Risk Engine — realized metrics for observed return series."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


class RiskEngine:
    @staticmethod
    def _clean(r: pd.Series) -> pd.Series:
        s = pd.to_numeric(r, errors="coerce").dropna()
        if not isinstance(s.index, pd.DatetimeIndex):
            raise ValueError("RiskEngine requires a DatetimeIndex")
        return s

    @staticmethod
    def realized_vol(r: pd.Series, window: int = 20, ann: int = 12) -> pd.Series:
        if window < 2 or ann <= 0:
            raise ValueError("window must be >= 2 and ann must be positive")
        return pd.to_numeric(r, errors="coerce").rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(ann)

    @staticmethod
    def ewma_vol(r: pd.Series, lam: float = 0.94, ann: int = 12) -> pd.Series:
        if not 0 < lam < 1 or ann <= 0:
            raise ValueError("lam must be in (0, 1) and ann must be positive")
        s = pd.to_numeric(r, errors="coerce")
        return s.pow(2).ewm(alpha=1 - lam, adjust=False, min_periods=2).mean().pow(0.5) * np.sqrt(ann)

    @staticmethod
    def historical_var(r: pd.Series, alpha: float = 0.05) -> float:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        s = RiskEngine._clean(r)
        return float(-s.quantile(alpha)) if len(s) else np.nan

    @staticmethod
    def expected_shortfall(r: pd.Series, alpha: float = 0.05) -> float:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        s = RiskEngine._clean(r)
        if not len(s):
            return np.nan
        q = s.quantile(alpha)
        tail = s[s <= q]
        return float(-tail.mean()) if len(tail) else np.nan

    @staticmethod
    def max_drawdown(r: pd.Series) -> float:
        s = RiskEngine._clean(r)
        if not len(s):
            return np.nan
        wealth = (1.0 + s).cumprod()
        return float((wealth / wealth.cummax() - 1.0).min())

    @staticmethod
    def sharpe(r: pd.Series, ann: int = 12, risk_free: float = 0.0) -> float:
        if ann <= 0:
            raise ValueError("ann must be positive")
        s = RiskEngine._clean(r) - risk_free
        if len(s) < 5 or s.std(ddof=1) == 0:
            return np.nan
        return float(s.mean() / s.std(ddof=1) * np.sqrt(ann))

    @staticmethod
    def sortino(r: pd.Series, ann: int = 12, mar: float = 0.0) -> float:
        if ann <= 0:
            raise ValueError("ann must be positive")
        s = RiskEngine._clean(r)
        excess = s - mar
        downside = np.minimum(excess, 0.0)
        downside_deviation = float(np.sqrt(np.mean(downside ** 2)))
        if len(s) < 5 or downside_deviation == 0:
            return np.nan
        return float(excess.mean() / downside_deviation * np.sqrt(ann))

    @staticmethod
    def calmar(r: pd.Series, ann: int = 12) -> float:
        if ann <= 0:
            raise ValueError("ann must be positive")
        s = RiskEngine._clean(r)
        mdd = RiskEngine.max_drawdown(s)
        if len(s) < 2 or mdd >= 0 or np.isnan(mdd):
            return np.nan
        years = (s.index[-1] - s.index[0]).days / 365.25
        if years <= 0:
            return np.nan
        wealth = float((1.0 + s).prod())
        if wealth <= 0:
            return np.nan
        cagr = wealth ** (1.0 / years) - 1.0
        return float(cagr / abs(mdd))

    @staticmethod
    def rolling_sharpe(r: pd.Series, window: int = 36, ann: int = 12) -> pd.Series:
        if window < 10 or ann <= 0:
            raise ValueError("window must be >= 10 and ann must be positive")
        s = pd.to_numeric(r, errors="coerce")
        def _sh(x: np.ndarray) -> float:
            x = x[np.isfinite(x)]
            if len(x) < 10 or np.std(x, ddof=1) == 0:
                return np.nan
            return float(np.mean(x) / np.std(x, ddof=1) * np.sqrt(ann))
        return s.rolling(window, min_periods=max(10, window // 2)).apply(_sh, raw=True)

    @staticmethod
    def beta(asset: pd.Series, market: pd.Series) -> float:
        df = pd.concat([pd.to_numeric(asset, errors="coerce"), pd.to_numeric(market, errors="coerce")], axis=1).dropna()
        if len(df) < 20:
            return np.nan
        market_var = df.iloc[:, 1].var(ddof=1)
        return float(df.iloc[:, 0].cov(df.iloc[:, 1]) / market_var) if market_var > 0 else np.nan

    @staticmethod
    def rolling_corr(a: pd.Series, b: pd.Series, window: int = 36) -> pd.Series:
        return pd.to_numeric(a, errors="coerce").rolling(window).corr(pd.to_numeric(b, errors="coerce"))

    @classmethod
    def summary(cls, r: pd.Series, lam: float = 0.94, ann: int = 12, mar: float = 0.0) -> Dict:
        s = cls._clean(r)
        if len(s) < 24:
            return {"status": "AMOSTRA INSUFICIENTE", "n": len(s)}
        vol = cls.realized_vol(s, ann=ann)
        ewma = cls.ewma_vol(s, lam, ann=ann)
        wealth = (1.0 + s).cumprod()
        return {
            "status": "OK", "n": len(s), "annualization": ann, "mar": mar,
            "vol_20": float(vol.iloc[-1]) if pd.notna(vol.iloc[-1]) else np.nan,
            "ewma_vol": float(ewma.iloc[-1]) if pd.notna(ewma.iloc[-1]) else np.nan,
            "var_95": cls.historical_var(s, 0.05), "var_99": cls.historical_var(s, 0.01),
            "es_95": cls.expected_shortfall(s, 0.05), "es_99": cls.expected_shortfall(s, 0.01),
            "max_dd": cls.max_drawdown(s), "sharpe": cls.sharpe(s, ann),
            "sortino": cls.sortino(s, ann, mar), "calmar": cls.calmar(s, ann),
            "vol_series": vol, "ewma_series": ewma,
            "dd_series": wealth / wealth.cummax() - 1.0,
        }
