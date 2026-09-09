"""Macro-ML Panel — institutional-style research terminal."""
from __future__ import annotations

import streamlit as st

from modules import (
    m1_recession, m2_regime_hmm, m3_structural_break, m4_phillips_curve,
    m5_nowcasting, m6_clustering, m7_trend_cycle, m8_nlp_sentiment,
    m9_anomaly, m10_gp_yield_curve,
)

st.set_page_config(page_title="Macro-ML Research Terminal", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

MODULES = [
    ("01", "Recessão", m1_recession), ("02", "Regimes", m2_regime_hmm),
    ("03", "Quebra", m3_structural_break), ("04", "Phillips", m4_phillips_curve),
    ("05", "Nowcasting", m5_nowcasting), ("06", "Câmbio", m6_clustering),
    ("07", "Tendência", m7_trend_cycle), ("08", "Sentimento", m8_nlp_sentiment),
    ("09", "Anomalias", m9_anomaly), ("10", "Yield Curve", m10_gp_yield_curve),
]

st.markdown("""
<style>
:root { --ink:#dce4ee; --muted:#8290a3; --line:#263244; --panel:#101722; --bg:#080d14; --accent:#58a6ff; }
.stApp { background:var(--bg); color:var(--ink); }
.block-container { max-width:1500px; padding-top:1.5rem; padding-bottom:2rem; }
section[data-testid="stSidebar"] { display:none; }
header[data-testid="stHeader"] { background:transparent; }
[data-testid="stMetric"] { background:var(--panel); border:1px solid var(--line); padding:12px 14px; border-radius:8px; }
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] { font-family:"JetBrains Mono","SFMono-Regular",Consolas,monospace; }
div[data-baseweb="tab-list"] { gap:4px; border-bottom:1px solid var(--line); }
button[data-baseweb="tab"] { height:42px; background:transparent; color:var(--muted); border-radius:7px 7px 0 0; }
button[data-baseweb="tab"][aria-selected="true"] { color:var(--ink); background:#111c2a; border-bottom:2px solid var(--accent); }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:7px; }
hr { border-color:var(--line); }
.research-header { border-bottom:1px solid var(--line); padding-bottom:14px; margin-bottom:12px; }
.kicker { color:var(--accent); font:600 11px/1.2 "JetBrains Mono",monospace; letter-spacing:.14em; text-transform:uppercase; }
.title { font:700 28px/1.15 Inter,system-ui,sans-serif; margin:5px 0; }
.subtitle { color:var(--muted); font-size:13px; }
.provenance { color:var(--muted); font:11px/1.5 "JetBrains Mono",monospace; border-top:1px solid var(--line); margin-top:20px; padding-top:9px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="research-header">
  <div class="kicker">QUANTITATIVE MACRO RESEARCH · LIVE PUBLIC DATA</div>
  <div class="title">Macro-ML Panel</div>
  <div class="subtitle">Macroeconometria + Machine Learning · 10 modelos · BCB/SGS + FRED · sem dados sintéticos</div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([f"{num} · {name}" for num, name, _ in MODULES])
for tab, (_, _, module) in zip(tabs, MODULES):
    with tab:
        try:
            module.render()
        except RuntimeError as exc:
            st.error("DADOS INDISPONÍVEIS — o modelo não foi executado.")
            st.caption(str(exc))
        except ValueError as exc:
            st.error("AMOSTRA INSUFICIENTE — o modelo não foi executado.")
            st.caption(str(exc))
        except Exception as exc:
            st.error("FALHA CONTROLADA — o módulo não pôde ser calculado.")
            st.caption(f"Detalhe operacional: {exc}")

st.markdown('<div class="provenance">DATA POLICY · Somente observações publicadas pelos provedores são aceitas. Falhas de API nunca são substituídas por dados sintéticos. Cache de dados: 15 min.</div>', unsafe_allow_html=True)
