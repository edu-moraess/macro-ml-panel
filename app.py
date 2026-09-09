"""Macro-ML Panel — quantitative macro research terminal."""
from __future__ import annotations

import streamlit as st

from modules import (
    m1_recession, m2_regime_hmm, m3_structural_break, m4_phillips_curve,
    m5_nowcasting, m6_clustering, m7_trend_cycle, m8_nlp_sentiment,
    m9_anomaly, m10_gp_yield_curve, m11_quant_state,
)

st.set_page_config(page_title="Quantitative Macro Research Terminal", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

MODULES = [
    ("Quant State", m11_quant_state, "Estado quantitativo agregado: fatores macro padronizados, regime, risco e sinal composto.", r"Q_t=w^\top Z_t,\qquad R_t=\tanh(Q_t/2)", "Os dados reais são transformados em fatores z-score com janela móvel. O composite combina crescimento, inflação, juros e câmbio em um estado quantitativo interpretável."),
    ("Recessão", m1_recession, "Probabilidade de recessão nos EUA usando spread de juros e Logit com regularização Elastic Net.", r"P(y_{t+h}=1\mid X_t)=\sigma(\beta_0+X_t\beta)", "Elastic Net combina penalização L1 e L2. O modelo usa spread 10Y–3M, dinâmica recente e referência NBER."),
    ("Regimes", m2_regime_hmm, "Identificação não supervisionada de regimes monetários por Hidden Markov Model.", r"P(S_t=j\mid S_{t-1}=i)=A_{ij}", "O HMM estima estados latentes, emissões e matriz de transição sem impor rótulos econômicos previamente."),
    ("Quebra Estrutural", m3_structural_break, "Detecção de pontos de mudança na estrutura estatística das séries.", r"C(\tau)=\sum_k\sum_{t\in I_k}(x_t-\bar{x}_{I_k})^2", "A segmentação procura changepoints que reduzem o erro intrassegmento."),
    ("Phillips", m4_phillips_curve, "Relação não linear entre desemprego e inflação nos EUA.", r"\pi_t=f(u_t,u_{t-12},\pi_{t-12})+\varepsilon_t", "Gradient Boosting estima uma função preditiva não linear e sua importância por permutação."),
    ("Nowcasting", m5_nowcasting, "Estimativa corrente da atividade econômica antes da divulgação oficial.", r"\widehat{IBC}_t=f(FX_t,Selic_t,IPCA_{t-1},U_{t-1})", "Gradient Boosting combina indicadores observáveis e avalia o desempenho fora da amostra."),
    ("Câmbio", m6_clustering, "Agrupamento quantitativo de países por retorno e volatilidade cambial.", r"\min_C\sum_j\sum_{x_i\in C_j}\lVert x_i-\mu_j\rVert^2", "K-Means agrupa observações por distância aos centroides."),
    ("Tendência", m7_trend_cycle, "Decomposição não linear entre tendência e ciclo via representação latente.", r"x\rightarrow z\rightarrow\hat{x},\qquad \min\sum_t(x_t-\hat{x}_t)^2", "Autoencoder aprende uma representação de baixa dimensão em janelas móveis."),
    ("Sentimento", m8_nlp_sentiment, "Classificação transparente de linguagem hawkish/dovish em documentos reais.", r"Score_t=1000\frac{N_{hawkish}-N_{dovish}}{N_{palavras}}", "Dicionário lexical explícito e normalização por mil palavras."),
    ("Anomalias", m9_anomaly, "Detecção multivariada de observações macroeconomicamente incomuns.", r"s(x)=-Score_{IF}(x)", "Isolation Forest identifica observações isoladas após padronização das variáveis."),
    ("Yield Curve", m10_gp_yield_curve, "Modelagem probabilística da curva de juros Treasury com Gaussian Process.", r"f(x)\sim\mathcal{GP}(m(x),k(x,x'))", "A média posterior estima a curva e o desvio posterior fornece a incerteza."),
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
div[data-baseweb="tab-list"] { gap:4px; border-bottom:1px solid var(--line); overflow-x:auto; }
button[data-baseweb="tab"] { height:42px; background:transparent; color:var(--muted); border-radius:7px 7px 0 0; }
button[data-baseweb="tab"][aria-selected="true"] { color:var(--ink); background:#111c2a; border-bottom:2px solid var(--accent); }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:7px; }
hr { border-color:var(--line); }
.research-header { border-bottom:1px solid var(--line); padding-bottom:14px; margin-bottom:12px; }
.kicker { color:var(--accent); font:600 11px/1.2 "JetBrains Mono",monospace; letter-spacing:.14em; text-transform:uppercase; }
.title { font:700 28px/1.15 Inter,system-ui,sans-serif; margin:5px 0; }
.subtitle { color:var(--muted); font-size:13px; }
.provenance { color:var(--muted); font:11px/1.5 "JetBrains Mono",monospace; border-top:1px solid var(--line); margin-top:20px; padding-top:9px; }
.method-card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px 16px; margin:0 0 14px 0; }
.method-title { color:var(--accent); font:600 11px/1.2 "JetBrains Mono",monospace; letter-spacing:.1em; text-transform:uppercase; margin-bottom:6px; }
.method-text { color:var(--muted); font-size:13px; line-height:1.55; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="research-header">
  <div class="kicker">QUANTITATIVE MACRO RESEARCH · LIVE PUBLIC DATA</div>
  <div class="title">Quantitative Macro Research Terminal</div>
  <div class="subtitle">Factor Engine · Regime Engine · Signal Engine · Risk Analytics · 11 modelos · BCB/SGS + FRED · sem dados sintéticos</div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([name for name, _, _, _, _ in MODULES])
for tab, (name, module, description, formula, methodology) in zip(tabs, MODULES):
    with tab:
        st.markdown(f"""
        <div class="method-card">
          <div class="method-title">QUANT ENGINE · {name}</div>
          <div class="method-text">{description}</div>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("Fórmula e metodologia", expanded=True):
            st.latex(formula)
            st.markdown(f"**Metodologia:** {methodology}")
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
