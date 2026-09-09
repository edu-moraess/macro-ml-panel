"""Module 11 — Quant State Engine.

Transforms observed macro/market series into standardized quantitative factors,
a composite regime score and a simple risk state. No synthetic observations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from data_utils import FRED, SGS, align, data_status, get_bcb, get_fred


def _zscore(s: pd.Series, window: int = 60) -> pd.Series:
    mean = s.rolling(window, min_periods=max(24, window // 2)).mean()
    std = s.rolling(window, min_periods=max(24, window // 2)).std()
    return (s - mean) / std.replace(0, np.nan)


def render() -> None:
    st.header("Quant State Engine")
    st.caption(
        "Camada quantitativa que transforma observações macroeconômicas reais em fatores "
        "padronizados, regime, risco e sinais compostos."
    )

    lookback = st.slider("Janela de padronização (meses)", 36, 120, 60, step=12)

    selic = get_bcb(SGS["selic_meta"]).resample("MS").mean().rename("selic")
    ipca = get_bcb(SGS["ipca_mensal"]).resample("MS").mean().rename("ipca")
    cambio = get_bcb(SGS["cambio_ptax"]).resample("MS").mean().rename("cambio")
    desemprego = get_bcb(SGS["desemprego_pnad"]).resample("MS").mean().rename("desemprego")
    spread = get_fred(FRED["t10y3m"]).resample("MS").mean().rename("term_spread")

    df = align(selic, ipca, cambio, desemprego, spread)
    df["fx_mom"] = df["cambio"].pct_change(3) * 100
    df["inflation_mom"] = df["ipca"].diff(3)
    df["rates_mom"] = df["selic"].diff(3)
    df["unemployment_mom"] = df["desemprego"].diff(3)
    df = df.dropna()

    if len(df) < max(60, lookback):
        raise ValueError(f"Apenas {len(df)} observações mensais disponíveis; amostra insuficiente para a janela escolhida.")

    # Economic sign convention: higher growth/risk appetite is positive.
    factors = pd.DataFrame(index=df.index)
    factors["Growth"] = -_zscore(df["unemployment_mom"], lookback) + _zscore(df["term_spread"], lookback)
    factors["Inflation"] = _zscore(df["inflation_mom"], lookback)
    factors["Rates"] = _zscore(df["rates_mom"], lookback)
    factors["FX"] = -_zscore(df["fx_mom"], lookback)
    factors["Risk"] = factors["Growth"] - factors["Inflation"] - factors["Rates"] + factors["FX"]
    factors = factors.replace([np.inf, -np.inf], np.nan).dropna()

    latest = factors.iloc[-1]
    regime_score = float(np.tanh(latest["Risk"] / 2.0))
    regime = "RISK-ON" if regime_score >= 0.25 else "RISK-OFF" if regime_score <= -0.25 else "NEUTRAL"
    probability = 50.0 + 50.0 * abs(regime_score)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("REGIME", regime)
    c2.metric("REGIME SCORE", f"{regime_score:+.2f}")
    c3.metric("CONFIDENCE", f"{probability:.1f}%")
    c4.metric("OBSERVAÇÕES", f"{len(factors):,}")

    st.subheader("Factor State")
    st.dataframe(latest.to_frame("z-score").style.format("{:+.2f}"), use_container_width=True)
    st.line_chart(factors[["Growth", "Inflation", "Rates", "FX"]], height=320)

    st.subheader("Composite Quant Signal")
    st.line_chart(factors["Risk"], height=260)
    st.caption(
        "O score é um indicador quantitativo de estado, não uma recomendação de investimento. "
        "Os fatores são padronizados em janela móvel para reduzir dependência de escala."
    )
    st.caption(data_status("BCB/SGS + FRED", "432 / 433 / 1 / 24369 / T10Y3M", factors.index.max()))
