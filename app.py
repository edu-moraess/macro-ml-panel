"""Macro-ML Panel — institutional quantitative macro research terminal."""
from __future__ import annotations
import streamlit as st
from modules import (m1_recession,m2_regime_hmm,m3_structural_break,m4_phillips_curve,m5_nowcasting,m6_clustering,m7_trend_cycle,m8_nlp_sentiment,m9_anomaly,m10_gp_yield_curve,m11_quant_state,m12_quant_lab)

st.set_page_config(page_title="Quant Macro Terminal",page_icon="⌁",layout="wide",initial_sidebar_state="collapsed")

THEMES={"Light — Research":{"bg":"#f5f7fa","panel":"#ffffff","ink":"#17202a","muted":"#667085","line":"#d9dee7","accent":"#174ea6"},"Dark — Terminal":{"bg":"#11151a","panel":"#171c22","ink":"#e6edf3","muted":"#8b98a8","line":"#2a333d","accent":"#58a6ff"}}
if "ui_theme" not in st.session_state: st.session_state.ui_theme="Light — Research"
theme_name=st.selectbox("Interface",list(THEMES),index=list(THEMES).index(st.session_state.ui_theme),label_visibility="collapsed")
st.session_state.ui_theme=theme_name
t=THEMES[theme_name]

st.markdown(f"""
<style>
:root{{--bg:{t['bg']};--panel:{t['panel']};--ink:{t['ink']};--muted:{t['muted']};--line:{t['line']};--accent:{t['accent']};}}
.stApp{{background:var(--bg);color:var(--ink);}}
.block-container{{max-width:1540px;padding-top:.8rem;padding-bottom:2rem;}}
header[data-testid="stHeader"]{{background:transparent;}}
[data-testid="stMetric"]{{background:var(--panel);border:1px solid var(--line);padding:12px 14px;border-radius:6px;box-shadow:none;}}
[data-testid="stMetricValue"],[data-testid="stMetricLabel"]{{font-family:"JetBrains Mono","SFMono-Regular",Consolas,monospace;}}
div[data-baseweb="tab-list"]{{gap:2px;border-bottom:1px solid var(--line);overflow-x:auto;}}
button[data-baseweb="tab"]{{height:40px;background:transparent;color:var(--muted);border-radius:4px 4px 0 0;font-family:"JetBrains Mono",monospace;font-size:12px;}}
button[data-baseweb="tab"][aria-selected="true"]{{color:var(--ink);background:var(--panel);border-bottom:2px solid var(--accent);}}
[data-testid="stDataFrame"]{{border:1px solid var(--line);border-radius:5px;}}
hr{{border-color:var(--line);}}
.terminal-header{{display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:12px;margin-bottom:10px;}}
.kicker{{color:var(--accent);font:600 10px/1.2 "JetBrains Mono",monospace;letter-spacing:.14em;text-transform:uppercase;}}
.title{{color:var(--ink);font:700 25px/1.15 Inter,system-ui,sans-serif;margin-top:4px;letter-spacing:-.02em;}}
.method-card{{background:var(--panel);border:1px solid var(--line);border-radius:5px;padding:12px 15px;margin:0 0 12px;}}
.method-title{{color:var(--accent);font:600 10px/1.2 "JetBrains Mono",monospace;letter-spacing:.12em;text-transform:uppercase;margin-bottom:5px;}}
.method-text{{color:var(--muted);font-size:12px;line-height:1.5;}}
.provenance{{color:var(--muted);font:10px/1.5 "JetBrains Mono",monospace;border-top:1px solid var(--line);margin-top:18px;padding-top:8px;}}
</style>""",unsafe_allow_html=True)

st.markdown('<div class="terminal-header"><div><div class="kicker">QUANT RESEARCH</div><div class="title">Macro Terminal</div></div></div>',unsafe_allow_html=True)

MODULES=[
("Quant State",m11_quant_state,"Macro state agregado: fatores, regime, risco e sinal composto.",r"Q_t=w^\top Z_t"),
("Quant Lab",m12_quant_lab,"Fatores, sinais, risco, séries temporais, backtest e alocação.",r"S_t=\sum_i w_i z_{i,t}"),
("Recessão",m1_recession,"Probabilidade de recessão nos EUA.",r"P(y=1|X)=\sigma(\beta_0+X\beta)"),
("Regimes",m2_regime_hmm,"Regimes monetários latentes via HMM.",r"P(S_t=j|S_{t-1}=i)=A_{ij}"),
("Quebra Estrutural",m3_structural_break,"Mudanças na estrutura estatística.",r"C(\tau)=\sum_k\sum_{t\in I_k}(x_t-\bar{x}_{I_k})^2"),
("Phillips",m4_phillips_curve,"Relação entre desemprego e inflação.",r"\pi_t=f(u_t,\pi_{t-12})+\varepsilon_t"),
("Nowcasting",m5_nowcasting,"Estimativa corrente da atividade.",r"\widehat{IBC}_t=f(X_t)"),
("Câmbio",m6_clustering,"Clustering de regimes cambiais.",r"\min_C\sum_j\sum_{x_i\in C_j}\lVert x_i-\mu_j\rVert^2"),
("Tendência",m7_trend_cycle,"Decomposição de tendência e ciclo.",r"x\rightarrow z\rightarrow\hat{x}"),
("Sentimento",m8_nlp_sentiment,"Sinal hawkish/dovish em documentos.",r"Score=1000(N_h-N_d)/N_{words}"),
("Anomalias",m9_anomaly,"Detecção de observações incomuns.",r"s(x)=-Score_{IF}(x)"),
("Yield Curve",m10_gp_yield_curve,"Curva Treasury probabilística.",r"f(x)\sim\mathcal{GP}(m(x),k(x,x'))"),
]

tabs=st.tabs([x[0] for x in MODULES])
for tab,(name,module,description,formula) in zip(tabs,MODULES):
    with tab:
        st.markdown(f'<div class="method-card"><div class="method-title">{name}</div><div class="method-text">{description}</div></div>',unsafe_allow_html=True)
        with st.expander("Metodologia",expanded=False): st.latex(formula)
        try: module.render()
        except RuntimeError as exc: st.error("DADOS INDISPONÍVEIS — o modelo não foi executado."); st.caption(str(exc))
        except ValueError as exc: st.error("AMOSTRA INSUFICIENTE — o modelo não foi executado."); st.caption(str(exc))
        except Exception as exc: st.error("FALHA CONTROLADA — o módulo não pôde ser calculado."); st.caption(f"Detalhe operacional: {exc}")

st.markdown('<div class="provenance">PUBLIC DATA · BCB/SGS + FRED · SEM DADOS SINTÉTICOS · CACHE 15 MIN</div>',unsafe_allow_html=True)
