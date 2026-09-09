"""Backtest Engine — lag-controlled, leakage-safe, observed returns only."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from core.risk import RiskEngine


class BacktestEngine:
    """Signal → lag → position → observed returns → performance metrics."""

    def __init__(self, lag: int = 1, threshold: float = 0.15, cost_bps: float = 0.0, ann: int = 12):
        if lag < 1:
            raise ValueError("lag must be >= 1 to prevent look-ahead bias")
        if threshold < 0 or cost_bps < 0 or ann <= 0:
            raise ValueError("threshold, cost_bps and ann must be non-negative/positive")
        self.lag = lag
        self.threshold = threshold
        self.cost_bps = cost_bps
        self.ann = ann

    def run(self, signal: pd.Series, returns: pd.Series) -> Dict:
        sig, ret = signal.align(returns, join="inner")
        sig = pd.to_numeric(sig, errors="coerce")
        ret = pd.to_numeric(ret, errors="coerce")
        sig = sig.shift(self.lag)
        valid = sig.notna() & ret.notna()
        sig, ret = sig.loc[valid], ret.loc[valid]
        if len(ret) < 2:
            raise ValueError("AMOSTRA INSUFICIENTE para backtest")

        pos = pd.Series(0.0, index=sig.index)
        pos.loc[sig > self.threshold] = 1.0
        pos.loc[sig < -self.threshold] = -1.0
        turnover = pos.diff().abs()
        turnover.iloc[0] = abs(pos.iloc[0])
        cost = turnover * (self.cost_bps / 10000.0)
        strat = pos * ret - cost
        eq = (1.0 + strat).cumprod()
        bh = (1.0 + ret).cumprod()

        exposed = pos != 0
        direction_correct = ((pos * ret) > 0) & exposed
        hit = float(direction_correct.sum() / exposed.sum()) if exposed.any() else np.nan
        wins = strat[exposed & (strat > 0)]
        losses = strat[exposed & (strat < 0)]
        pf = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else np.nan
        n_years = (eq.index[-1] - eq.index[0]).days / 365.25
        cagr = float(eq.iloc[-1] ** (1.0 / n_years) - 1.0) if n_years > 0 and eq.iloc[-1] > 0 else np.nan

        return {
            "equity": eq, "benchmark": bh, "positions": pos, "strat_returns": strat,
            "cagr": cagr, "sharpe": RiskEngine.sharpe(strat, self.ann),
            "sortino": RiskEngine.sortino(strat, self.ann), "calmar": RiskEngine.calmar(strat, self.ann),
            "max_dd": RiskEngine.max_drawdown(strat), "hit_ratio": hit,
            "exposed_periods": int(exposed.sum()), "n_trades": int((turnover > 0).sum()),
            "turnover": float(turnover.sum()), "avg_win": float(wins.mean()) if len(wins) else np.nan,
            "avg_loss": float(losses.mean()) if len(losses) else np.nan, "profit_factor": pf,
            "rolling_sharpe": RiskEngine.rolling_sharpe(strat, ann=self.ann),
            "lag": self.lag, "threshold": self.threshold, "annualization": self.ann,
        }
