"""Quant State — canonical macro factors, signal and regime."""
from __future__ import annotations

import streamlit as st

from core.macro_state import build_macro_state, load_macro_panel
from data_utils import data_status


def render() -> None:
    st.header("Quant State Engine")
    st.caption(
        "Estado macroeconômico canônico: dados reais → fatores → sinal composto → regime estatístico."
    )
    lookback = st.slider("Janela de padronização (meses)", 36, 120, 60, step=12, key="quant_state_window")

    state = build_macro_state(load_macro_panel(), window=lookback)
    factors = state["factors"]
    signal = state["signal"]
    regime = state["regime"]
    latest = factors.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("REGIME", regime.get("regime", "N/A"))
    c2.metric("MACRO SCORE", f"{signal.get('macro_score', float('nan')):+.1f}")
    c3.metric("SIGNAL CONFIDENCE", f"{signal.get('confidence', 0.0):.1f}%")
    c4.metric("OBSERVAÇÕES", f"{len(factors):,}")

    st.subheader("Factor State")
    st.dataframe(latest.to_frame("z-score").style.format("{:+.2f}"), use_container_width=True)
    st.line_chart(factors, height=320)

    st.subheader("Composite Quant Signal")
    st.line_chart(signal["score_series"].rename("MACRO SCORE"), height=260)
    st.caption(
        "O MACRO SCORE é um indicador quantitativo de estado, não uma recomendação de investimento. "
        "O regime é estimado pelo Regime Engine; a confiança do sinal não representa uma probabilidade estatística de retorno."
    )
    st.caption(data_status("BCB/SGS + FRED", "macro state canonical panel", factors.index.max()))
