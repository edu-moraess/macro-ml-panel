"""Feature Engine — centralized quantitative transformations on real series only."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from data_utils import rolling_zscore


class FeatureEngine:
    """Statistically justified transforms. No lookahead. No synthetic fill."""

    @staticmethod
    def yoy(s: pd.Series, periods: int = 12) -> pd.Series:
        return s.pct_change(periods) * 100

    @staticmethod
    def mom(s: pd.Series) -> pd.Series:
        return s.pct_change(1) * 100

    @staticmethod
    def growth_3m_ann(s: pd.Series) -> pd.Series:
        return ((1 + s.pct_change(3)) ** 4 - 1) * 100

    @staticmethod
    def diff(s: pd.Series, periods: int = 1) -> pd.Series:
        return s.diff(periods)

    @staticmethod
    def acceleration(s: pd.Series, periods: int = 3) -> pd.Series:
        return s.diff(periods).diff(periods)

    @staticmethod
    def rolling_vol(s: pd.Series, window: int = 12, ann: int = 12) -> pd.Series:
        r = s.pct_change() if s.abs().max() > 5 else s
        return r.rolling(window, min_periods=max(6, window // 2)).std() * np.sqrt(ann)

    @staticmethod
    def momentum(s: pd.Series, periods: int) -> pd.Series:
        return s.pct_change(periods) * 100

    @staticmethod
    def drawdown(s: pd.Series) -> pd.Series:
        wealth = (1 + s.pct_change().fillna(0)).cumprod()
        return wealth / wealth.cummax() - 1

    @staticmethod
    def z(s: pd.Series, window: int = 60) -> pd.Series:
        return rolling_zscore(s, window)

    @staticmethod
    def real_rate(nominal: pd.Series, inflation_yoy: pd.Series) -> pd.Series:
        aligned = pd.concat([nominal, inflation_yoy], axis=1).dropna()
        if aligned.empty:
            return pd.Series(dtype=float)
        return (aligned.iloc[:, 0] - aligned.iloc[:, 1]).rename("real_rate")

    @classmethod
    def build_macro_features(
        cls,
        ibc: pd.Series,
        unemp: pd.Series,
        ipca: pd.Series,
        selic: pd.Series,
        fx: pd.Series,
        vix: Optional[pd.Series] = None,
        hy: Optional[pd.Series] = None,
        spx: Optional[pd.Series] = None,
        spread: Optional[pd.Series] = None,
        window: int = 60,
    ) -> pd.DataFrame:
        feats = pd.DataFrame(index=ibc.index)
        feats["growth_yoy"] = cls.yoy(ibc)
        feats["growth_3m"] = cls.growth_3m_ann(ibc)
        feats["growth_accel"] = cls.acceleration(ibc, 3)
        feats["unemp_chg3"] = cls.diff(unemp, 3)
        feats["inf_yoy"] = cls.yoy(ipca.cumsum() if ipca.abs().max() < 5 else ipca)
        feats["inf_mom"] = ipca.rolling(3).sum()
        feats["inf_accel"] = cls.diff(ipca, 3)
        feats["rates_level"] = selic
        feats["rates_d1"] = cls.diff(selic, 1)
        feats["rates_d3"] = cls.diff(selic, 3)
        feats["fx_ret3"] = cls.momentum(fx, 3)
        feats["fx_ret12"] = cls.momentum(fx, 12)
        if vix is not None:
            feats["vix"] = vix.reindex(feats.index)
        if hy is not None:
            feats["hy"] = hy.reindex(feats.index)
        if spx is not None:
            feats["spx_mom3"] = cls.momentum(spx, 3)
            feats["spx_mom12"] = cls.momentum(spx, 12)
        if spread is not None:
            feats["term_spread"] = spread.reindex(feats.index)
        return feats.replace([np.inf, -np.inf], np.nan).dropna(how="all")
