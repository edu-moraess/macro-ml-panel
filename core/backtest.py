"""Backtest Engine V2 — lag-controlled, no leakage, real returns only."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from core.risk import RiskEngine


class BacktestEngine:
    """Signal → lag → position → returns → metrics. Mandatory lag >= 1."""

    def __init__(self, lag: int = 1, threshold: float = 0.15, cost_bps: float = 0.0):
        if lag < 1:
            raise ValueError("lag must be >= 1 to prevent look-ahead bias")
        self.lag = lag
        self.threshold = threshold
        self.cost_bps = cost_bps

    def run(self, signal: pd.Series, returns: pd.Series) -> Dict:
        sig = signal.shift(self.lag)
        ret = returns.reindex(sig.index)
        pos = pd.Series(0.0, index=sig.index)
        pos = pos.mask(sig > self.threshold, 1.0)
        pos = pos.mask(sig < -self.threshold, -1.0)
        turnover = pos.diff().abs().fillna(0)
        cost = turnover * (self.cost_bps / 10000.0)
        strat = (pos * ret - cost).fillna(0.0)
        eq = (1 + strat).cumprod()
        bh = (1 + ret.fillna(0)).cumprod()
        n_trades = int((turnover > 0).sum())
        wins = strat[strat > 0]
        losses = strat[strat < 0]
        hit = float((np.sign(strat) == np.sign(ret)).mean()) if len(strat) else np.nan
        avg_win = float(wins.mean()) if len(wins) else np.nan
        avg_loss = float(losses.mean()) if len(losses) else np.nan
        pf = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else np.nan
        n_years = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.01) if len(eq) > 1 else np.nan
        cagr = float(eq.iloc[-1] ** (1 / n_years) - 1) if n_years and eq.iloc[-1] > 0 else np.nan
        return {
            "equity": eq,
            "benchmark": bh,
            "positions": pos,
            "strat_returns": strat,
            "cagr": cagr,
            "sharpe": RiskEngine.sharpe(strat),
            "sortino": RiskEngine.sortino(strat),
            "calmar": RiskEngine.calmar(strat),
            "max_dd": RiskEngine.max_drawdown(strat),
            "hit_ratio": hit,
            "n_trades": n_trades,
            "turnover": float(turnover.sum() / 2),
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": pf,
            "rolling_sharpe": RiskEngine.rolling_sharpe(strat),
            "lag": self.lag,
            "threshold": self.threshold,
        }
