"""Macro-ML Panel — institutional quantitative macro research terminal."""
from __future__ import annotations
import streamlit as st
from modules import (m1_recession,m2_regime_hmm,m3_structural_break,m4_phillips_curve,m5_nowcasting,m6_clustering,m7_trend_cycle,m8_nlp_sentiment,m9_anomaly,m10_gp_yield_curve,m11_quant_state,m12_quant_lab)

st.set_page_config(page_title="Quantitative Macro Research Terminal",page_icon="⌁",layout="wide",initial_sidebar_state="collapsed")

THEMES={
    "Light — Research": {"bg":"#f5f7fa","panel":"#ffffff","ink":"#17202a","muted":"#667085","line":"#d9dee7","accent":"#174ea6","accent2":"#0f766e"},
    "Dark — Terminal": {"bg":"#11151a","panel":"#171c22","ink":"#e6edf3","muted":"#8b98a8","line":"#2a333d","accent":"#58a6ff","accent2":"#3fb950"},
}
if "ui_theme" not in st.session_state: st.session_state.ui_theme="Light — Research"
ctrl1,ctrl2=st.columns([8,2])
with ctrl2:
    theme_name=st.selectbox("Interface",list(THEMES),index=list(THEMES).index(st.session_state.ui_theme),label_visibility="collapsed")
st.session_state.ui_theme=theme_name
t=THEMES[theme_name]

st.markdown(f"""
<style>
:root{{--bg:{t['bg']};--panel:{t['panel']};--ink:{t['ink']};--muted:{t['muted']};--line:{t['line']};--accent:{t['accent']};--accent2:{t['accent2']};}}
.stApp{{background:var(--bg);color:var(--ink);}}
.block-container{{max-width:1540px;padding-top:1rem;padding-bottom:2rem;}}
header[data-testid="stHeader"]{{background:transparent;}}
[data-testid="stMetric"]{{background:var(--panel);border:1px solid var(--line);padding:12px 14px;border-radius:6px;box-shadow:none;}}
[data-testid="stMetricValue"],[data-testid="stMetricLabel"]{{font-family:"JetBrains Mono","SFMono-Regular",Consolas,monospace;}}
div[data-baseweb="tab-list"]{{gap:2px;border-bottom:1px solid var(--line);overflow-x:auto;}}
button[data-baseweb="tab"]{{height:40px;background:transparent;color:var(--muted);border-radius:4px 4px 0 0;font-family:"JetBrains Mono",monospace;font-size:12px;}}
button[data-baseweb="tab"][aria-selected="true"]{{color:var(--ink);background:var(--panel);border-bottom:2px solid var(--accent);}}
[data-testid="stDataFrame"]{{border:1px solid var(--line);border-radius:5px;}}
hr{{border-color:var(--line);}}
.research-header{{border-bottom:1px solid var(--line);padding-bottom:13px;margin-bottom:10px;}}
.kicker{{color:var(--accent);font:600 10px/1.2 "JetBrains Mono",monospace;letter-spacing:.16em;text-transform:uppercase;}}
.title{{color:var(--ink);font:700 27px/1.15 Inter,system-ui,sans-serif;margin:5px 0;letter-spacing:-.02em;}}
.subtitle{{color:var(--muted);font:12px/1.5 "JetBrains Mono",monospace;}}
.method-card{{background:var(--panel);border:1px solid var(--line);border-radius:5px;padding:12px 15px;margin:0 0 12px;}}
.method-title{{color:var(--accent);font:600 10px/1.2 "JetBrains Mono",monospace;letter-spacing:.12em;text-transform:uppercase;margin-bottom:5px;}}
.method-text{{color:var(--muted);font-size:12px;line-height:1.5;}}
.provenance{{color:var(--muted);font:10px/1.5 "JetBrains Mono",monospace;border-top:1px solid var(--line);margin-top:18px;padding-top:8px;}}
</style>""",unsafe_allow_html=True)

st.markdown(f'''<div class="research-header"><div class="kicker">QUANTITATIVE MACRO RESEARCH · LIVE PUBLIC DATA</div><div class="title">Quantitative Macro Research Terminal</div><div class="subtitle">FACTOR ENGINE · REGIME ENGINE · SIGNAL ENGINE · RISK ENGINE · TIME-SERIES · BACKTEST · PORTFOLIO · 17 ENGINES · BCB/SGS + FRED · NO SYNTHETIC DATA</div></div>''',unsafe_allow_html=True)

MODULES=[
("Quant State",m11_quant_state,"Macro state agregado: fatores padronizados, regime, risco e sinal composto.",r"Q_t=w^\top Z_t,\qquad R_t=\tanh(Q_t/2)"),
("Quant Lab",m12_quant_lab,"Suite integrada de fatores, sinais, risco, séries temporais, backtest e alocação.",r"S_t=\sum_i w_i z_{i,t},\qquad w_i\geq0,\quad\sum_iw_i=1"),
("Recessão",m1_recession,"Probabilidade de recessão nos EUA via spread de juros e Elastic Net.",r"P(y=1|X)=\sigma(\beta_0+X\beta)"),
("Regimes",m2_regime_hmm,"Regimes monetários latentes via Hidden Markov Model.",r"P(S_t=j|S_{t-1}=i)=A_{ij}"),
("Quebra Estrutural",m3_structural_break,"Detecção de mudanças na estrutura estatística das séries.",r"C(\tau)=\sum_k\sum_{t\in I_k}(x_t-\bar{x}_{I_k})^2"),
("Phillips",m4_phillips_curve,"Relação não linear entre desemprego e inflação.",r"\pi_t=f(u_t,u_{t-12},\pi_{t-12})+\varepsilon_t"),
("Nowcasting",m5_nowcasting,"Estimativa corrente da atividade econômica antes da divulgação oficial.",r"\widehat{IBC}_t=f(FX_t,Selic_t,IPCA_t,U_t)"),
("Câmbio",m6_clustering,"Clustering quantitativo de retornos e volatilidade cambial.",r"\min_C\sum_j\sum_{x_i\in C_j}\lVert x_i-\mu_j\rVert^2"),
("Tendência",m7_trend_cycle,"Decomposição de tendência/ciclo por representação latente.",r"x\rightarrow z\rightarrow\hat{x}"),
("Sentimento",m8_nlp_sentiment,"Classificação transparente de linguagem hawkish/dovish.",r"Score=1000(N_h-N_d)/N_{words}"),
("Anomalias",m9_anomaly,"Detecção multivariada de observações macro incomuns.",r"s(x)=-Score_{IF}(x)"),
("Yield Curve",m10_gp_yield_curve,"Curva Treasury com Gaussian Process e incerteza posterior.",r"f(x)\sim\mathcal{GP}(m(x),k(x,x'))"),
]

tabs=st.tabs([x[0] for x in MODULES])
for tab,(name,module,description,formula) in zip(tabs,MODULES):
    with tab:
        st.markdown(f'<div class="method-card"><div class="method-title">QUANT ENGINE · {name}</div><div class="method-text">{description}</div></div>',unsafe_allow_html=True)
        with st.expander("Fórmula e metodologia",expanded=False): st.latex(formula)
        try: module.render()
        except RuntimeError as exc: st.error("DADOS INDISPONÍVEIS — o modelo não foi executado."); st.caption(str(exc))
        except ValueError as exc: st.error("AMOSTRA INSUFICIENTE — o modelo não foi executado."); st.caption(str(exc))
        except Exception as exc: st.error("FALHA CONTROLADA — o módulo não pôde ser calculado."); st.caption(f"Detalhe operacional: {exc}")

st.markdown('<div class="provenance">DATA POLICY · Somente observações publicadas pelos provedores são aceitas. Falhas de API nunca são substituídas por dados sintéticos. Cache: 15 min · UI: Light/Dark selecionável.</div>',unsafe_allow_html=True)
