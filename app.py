"""Macro Quant Research Terminal — macro intelligence and historical validation."""
from __future__ import annotations

import streamlit as st

from modules import (
    m1_recession, m2_regime_hmm, m3_structural_break, m4_phillips_curve,
    m5_nowcasting, m6_clustering, m7_trend_cycle, m8_nlp_sentiment,
    m9_anomaly, m10_gp_yield_curve, m11_quant_state, m12_quant_lab,
)
from modules import data_quality, macro_intelligence

st.set_page_config(page_title="Quant Macro Terminal", page_icon="⌁", layout="wide", initial_sidebar_state="collapsed")

THEMES = {
    "Light": {"bg":"#F5F7FA","panel":"#FFFFFF","ink":"#17202A","muted":"#667085","line":"#D9DEE7","accent":"#174EA6"},
    "Dark": {"bg":"#11151A","panel":"#171C22","ink":"#E6EDF3","muted":"#8B98A8","line":"#2A333D","accent":"#58A6FF"},
}
if "ui_theme" not in st.session_state:
    st.session_state.ui_theme = "Light"
with st.sidebar:
    st.caption("Interface")
    theme_name = st.radio("Theme", list(THEMES), index=list(THEMES).index(st.session_state.ui_theme), label_visibility="collapsed", horizontal=True)
    st.session_state.ui_theme = theme_name

t = THEMES[st.session_state.ui_theme]
st.markdown(f"""
<style>
:root {{--bg:{t['bg']};--panel:{t['panel']};--ink:{t['ink']};--muted:{t['muted']};--line:{t['line']};--accent:{t['accent']};}}
.stApp{{background:var(--bg);color:var(--ink);}} .block-container{{max-width:1540px;padding-top:.8rem;padding-bottom:2rem;}}
header[data-testid="stHeader"]{{background:transparent;}} [data-testid="stMetric"]{{background:var(--panel);border:1px solid var(--line);padding:12px 14px;border-radius:6px;}}
[data-testid="stMetricValue"],[data-testid="stMetricLabel"]{{font-family:"JetBrains Mono","SFMono-Regular",Consolas,monospace;}}
div[data-baseweb="tab-list"]{{gap:2px;border-bottom:1px solid var(--line);overflow-x:auto;}} button[data-baseweb="tab"]{{height:40px;background:transparent;color:var(--muted);border-radius:4px 4px 0 0;font-family:"JetBrains Mono",monospace;font-size:12px;}}
button[data-baseweb="tab"][aria-selected="true"]{{color:var(--ink);background:var(--panel);border-bottom:2px solid var(--accent);}} [data-testid="stDataFrame"]{{border:1px solid var(--line);border-radius:5px;}}
.terminal-header{{display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:12px;margin-bottom:10px;}}
.kicker{{color:var(--accent);font:600 10px/1.2 "JetBrains Mono",monospace;letter-spacing:.14em;text-transform:uppercase;}} .title{{color:var(--ink);font:700 25px/1.15 Inter,system-ui,sans-serif;margin-top:4px;letter-spacing:-.02em;}}
.method-card{{background:var(--panel);border:1px solid var(--line);border-radius:5px;padding:12px 15px;margin:0 0 12px;}} .method-title{{color:var(--accent);font:600 10px/1.2 "JetBrains Mono",monospace;letter-spacing:.12em;text-transform:uppercase;margin-bottom:5px;}} .method-text{{color:var(--muted);font-size:12px;line-height:1.5;}}
.provenance{{color:var(--muted);font:10px/1.5 "JetBrains Mono",monospace;border-top:1px solid var(--line);margin-top:18px;padding-top:8px;}}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="terminal-header"><div><div class="kicker">QUANT RESEARCH</div><div class="title">Macro Terminal</div></div></div>', unsafe_allow_html=True)

MODULES = [
    ("MACRO INTELLIGENCE", macro_intelligence, "Score · Regime · Contributions · Historical Validation.", r"S_t \\to R_t \\to E[r_{t+h}]"),
    ("QUANT STATE", m11_quant_state, "Macro state: regime, score, factors, curve.", r"Q_t=w^\\top Z_t"),
    ("QUANT LAB", m12_quant_lab, "Factors · Signals · Risk · TS · Backtest · Portfolio.", r"S_t=\\sum_i w_i z_{i,t}"),
    ("YIELD CURVE", m10_gp_yield_curve, "Curve factors, GP construction, regime, history.", r"f\\sim GP"),
    ("DATA QUALITY", data_quality, "Provenance e status de todas as séries.", r""),
    ("Recessão", m1_recession, "Elastic Net Logit — probabilidade de recessão.", r"P(y=1|X)=\\sigma(\\beta_0+X\\beta)"),
    ("Regimes HMM", m2_regime_hmm, "HMM monetário legado.", r"P(S_t=j|S_{t-1}=i)=A_{ij}"),
    ("Quebra", m3_structural_break, "Changepoint detection.", r"C(\\tau)"),
    ("Phillips", m4_phillips_curve, "Desemprego × inflação.", r"\\pi_t=f(u_t)"),
    ("Nowcasting", m5_nowcasting, "Atividade corrente.", r"\\widehat{IBC}_t"),
    ("Câmbio", m6_clustering, "Clustering cambial.", r"K-Means"),
    ("Tendência", m7_trend_cycle, "Autoencoder tendência-ciclo.", r"x\\to z\\to\\hat{x}"),
    ("Sentimento", m8_nlp_sentiment, "Hawkish/dovish lexicon.", r"Score"),
    ("Anomalias", m9_anomaly, "Isolation Forest.", r"s(x)"),
]
tabs = st.tabs([x[0] for x in MODULES])
for tab, (name, module, description, formula) in zip(tabs, MODULES):
    with tab:
        st.markdown(f'<div class="method-card"><div class="method-title">{name}</div><div class="method-text">{description}</div></div>', unsafe_allow_html=True)
        if formula:
            with st.expander("Metodologia", expanded=False): st.latex(formula)
        try: module.render()
        except RuntimeError as exc: st.error("DADOS INDISPONÍVEIS — o modelo não foi executado."); st.caption(str(exc))
        except ValueError as exc: st.error("AMOSTRA INSUFICIENTE — o modelo não foi executado."); st.caption(str(exc))
        except Exception as exc: st.error("FALHA CONTROLADA — o módulo não pôde ser calculado."); st.caption(f"Detalhe operacional: {exc}")

st.markdown('<div class="provenance">PUBLIC DATA · BCB/SGS + FRED · SEM DADOS SINTÉTICOS · CACHE 15 MIN</div>', unsafe_allow_html=True)
