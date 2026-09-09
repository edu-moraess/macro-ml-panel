"""Quant Factor Engine — real macro/market factor construction."""
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
from data_utils import FRED, SGS, data_status, get_bcb, get_fred

def _z(s: pd.Series, n: int = 60) -> pd.Series:
    r = s.rolling(n, min_periods=max(24, n // 2))
    return (s - r.mean()) / r.std().replace(0, np.nan)

def render() -> None:
    st.header("Factor Engine")
    st.caption("Fatores macro e de mercado em unidades comparáveis, somente com observações públicas.")
    n = st.slider("Lookback", 36, 120, 60, 12)
    s = {
        "Growth": get_bcb(SGS["ibc_br"]).resample("MS").mean(),
        "Unemployment": get_bcb(SGS["desemprego_pnad"]).resample("MS").mean(),
        "Inflation": get_bcb(SGS["ipca_mensal"]).resample("MS").mean(),
        "Rates": get_bcb(SGS["selic_meta"]).resample("MS").mean(),
        "FX": get_bcb(SGS["cambio_ptax"]).resample("MS").mean(),
        "TermSpread": get_fred(FRED["t10y3m"]).resample("MS").mean(),
        "VIX": get_fred(FRED["vix"]).resample("MS").mean(),
        "HYSpread": get_fred(FRED["hy_spread"]).resample("MS").mean(),
    }
    df = pd.concat(s, axis=1).dropna()
    f = pd.DataFrame(index=df.index)
    f["Growth"] = _z(df.Growth.pct_change(3) * 100, n) - _z(df.Unemployment.diff(3), n)
    f["Inflation"] = _z(df.Inflation.diff(3), n)
    f["Rates"] = _z(df.Rates.diff(3), n)
    f["FX"] = -_z(df.FX.pct_change(3) * 100, n)
    f["Liquidity"] = -_z(df.HYSpread, n) - _z(df.VIX, n)
    f["Carry"] = _z(df.Rates - df.Rates.rolling(12).mean(), n)
    f["Momentum"] = _z(df.Growth.pct_change(12) * 100, n)
    f = f.replace([np.inf, -np.inf], np.nan).dropna()
    if len(f) < n:
        raise ValueError(f"Apenas {len(f)} observações fatoriais; amostra insuficiente para lookback {n}.")
    latest = f.iloc[-1]
    cols = st.columns(4)
    for c, (name, val) in zip(cols, latest.items()):
        c.metric(name.upper(), f"{val:+.2f}σ")
    st.subheader("Factor Matrix")
    st.dataframe(latest.sort_values(ascending=False).to_frame("z-score").style.format("{:+.2f}"), use_container_width=True)
    st.subheader("Cross-Factor State")
    st.line_chart(f, height=360)
    st.caption(data_status("BCB/SGS + FRED", "24363 / 24369 / 433 / 432 / 1 / T10Y3M / VIXCLS / BAMLH0A0HYM2", f.index.max()))
