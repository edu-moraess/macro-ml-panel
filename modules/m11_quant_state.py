"""Quant State — canonical macro factors, signal and historical state replay."""
from __future__ import annotations
import streamlit as st
from core.macro_state import build_macro_state, load_macro_panel
from data_utils import data_status

def render()->None:
    st.header("Quant State Engine")
    st.caption("Estado macroeconômico canônico: dados reais → fatores → sinal composto → regime estatístico. Inclui replay de datas históricas.")
    lookback=st.slider("Janela de padronização (meses)",36,120,60,step=12,key="quant_state_window")
    state=build_macro_state(load_macro_panel(),window=lookback); factors=state["factors"]; signal=state["signal"]; regime=state["regime"]
    latest=factors.iloc[-1]; c1,c2,c3,c4=st.columns(4); c1.metric("REGIME",regime.get("regime","N/A")); c2.metric("MACRO SCORE",f"{signal.get('macro_score',float('nan')):+.1f}"); c3.metric("SIGNAL CONFIDENCE",f"{signal.get('confidence',0.0):.1f}%"); c4.metric("OBSERVAÇÕES",f"{len(factors):,}")
    st.subheader("Factor State atual"); st.dataframe(latest.to_frame("z-score").style.format("{:+.2f}"),use_container_width=True); st.line_chart(factors,height=320)
    st.subheader("Composite Quant Signal"); st.line_chart(signal["score_series"].rename("MACRO SCORE"),height=260)
    st.subheader("Historical State Replay")
    dates=factors.index; selected=st.select_slider("Data histórica",options=list(dates),value=dates[-1],format_func=lambda x:x.strftime("%Y-%m"),key="quant_state_replay_date"); snap=factors.loc[selected]
    rc1,rc2,rc3=st.columns(3); rc1.metric("DATA",selected.strftime("%Y-%m")); rc2.metric("SCORE",f"{float(signal['score_series'].loc[selected]):+.2f}" if selected in signal['score_series'].index else "—"); rc3.metric("FATORES",f"{snap.notna().sum()}/{len(snap)}")
    st.dataframe(snap.to_frame("z-score").style.format("{:+.2f}"),use_container_width=True)
    st.write("**Contribuição relativa no estado selecionado:**")
    st.bar_chart(snap.sort_values())
    st.caption("Replay preserva as definições canônicas de fatores e score. O regime mostrado no topo é o regime atual; esta versão não reclassifica historicamente o HMM sem um ajuste específico por data.")
    st.caption(data_status("BCB/SGS + FRED","macro state canonical panel",factors.index.max()))
