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
    ("Recessão", m1_recession, "Probabilidade de recessão nos EUA usando o spread de juros e um Logit com regularização Elastic Net.", r"P(y_{t+h}=1\mid X_t)=\sigma(\beta_0+X_t\beta),\quad \sigma(z)=\frac{1}{1+e^{-z}}", "Elastic Net combina L1 e L2 na estimação dos coeficientes. O modelo usa spread 10Y–3M, variação de 3 meses e mínimo móvel de 6 meses para prever a recessão NBER no horizonte escolhido."),
    ("Regimes", m2_regime_hmm, "Identificação não supervisionada de regimes de política monetária a partir da dinâmica dos juros.", r"P(S_t=j\mid S_{t-1}=i)=A_{ij},\quad \Delta i_t\mid S_t=j\sim\mathcal{N}(\mu_j,\sigma_j^2)", "O Hidden Markov Model estima estados latentes, distribuição da variação dos juros em cada estado e matriz de transição. Os regimes não recebem rótulos econômicos previamente impostos."),
    ("Quebra Estrutural", m3_structural_break, "Detecção de pontos em que a estrutura estatística da série muda de forma relevante.", r"C(\tau)=\sum_{k=1}^{K}\sum_{t\in I_k}(x_t-\bar{x}_{I_k})^2", "A segmentação binária procura pontos de mudança que reduzem o custo L2 dentro dos segmentos. As datas encontradas são changepoints estatísticos, não necessariamente eventos causais."),
    ("Phillips", m4_phillips_curve, "Estimativa não linear da relação entre desemprego, inflação passada e inflação corrente nos EUA.", r"\pi_t=f(u_t,u_{t-12},\pi_{t-12})+\varepsilon_t", "Gradient Boosting constrói uma função preditiva por sucessivas árvores de decisão. A importância por permutação mede quanto o desempenho cai quando uma variável é embaralhada."),
    ("Nowcasting", m5_nowcasting, "Estimativa corrente da atividade econômica antes da divulgação oficial do IBC-Br.", r"\widehat{IBC}_t=f(FX_t,Selic_t,IPCA_{t-1},U_{t-1})", "Gradient Boosting combina indicadores que podem estar disponíveis antes do IBC-Br. O desempenho é avaliado fora da amostra e comparado ao benchmark ingênuo de persistência."),
    ("Câmbio", m6_clustering, "Agrupamento de países segundo padrões observados de volatilidade e retorno cambial.", r"\min_{C_1,\ldots,C_k}\sum_{j=1}^{k}\sum_{x_i\in C_j}\lVert x_i-\mu_j\rVert^2", "K-Means minimiza a distância quadrática dos pontos aos centroides. As features são volatilidade cambial rolling de 12 meses, retorno de 3 meses e retorno de 12 meses."),
    ("Tendência", m7_trend_cycle, "Decomposição não linear entre tendência e ciclo usando um autoencoder com gargalo latente.", r"x\xrightarrow{encoder}z\xrightarrow{decoder}\hat{x},\qquad \min\sum_t(x_t-\hat{x}_t)^2", "O autoencoder aprende uma representação de baixa dimensão em janelas móveis. O componente cíclico é calculado como observado menos a tendência latente reconstruída/calibrada."),
    ("Sentimento", m8_nlp_sentiment, "Classificação transparente de linguagem hawkish/dovish em comunicados reais fornecidos pelo pesquisador.", r"Score_t=1000\times\frac{N_{hawkish}-N_{dovish}}{N_{palavras}}", "O método usa um dicionário lexical explícito. Termos hawkish aumentam o score e termos dovish reduzem o score; a normalização por mil palavras permite comparar documentos de tamanhos diferentes."),
    ("Anomalias", m9_anomaly, "Detecção multivariada de meses macroeconomicamente incomuns.", r"s(x)=-Score_{IF}(x),\qquad \hat{y}_t\in\{-1,+1\}", "Isolation Forest isola observações por particionamentos aleatórios. Selic, variação cambial, IPCA e desemprego são padronizados antes da detecção; a contaminação controla a fração esperada de anomalias."),
    ("Yield Curve", m10_gp_yield_curve, "Modelagem suave da curva de juros do Treasury dos EUA com processo gaussiano e intervalo de incerteza.", r"f(x)\sim\mathcal{GP}(m(x),k(x,x')), \quad y=f(x)+\varepsilon", "O Gaussian Process usa os vértices observados de maturidade do Treasury e uma combinação RBF + ruído. A média posterior fornece a curva estimada e o desvio posterior sustenta a faixa de incerteza."),
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
.method-card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px 16px; margin:0 0 14px 0; }
.method-title { color:var(--accent); font:600 11px/1.2 "JetBrains Mono",monospace; letter-spacing:.1em; text-transform:uppercase; margin-bottom:6px; }
.method-text { color:var(--muted); font-size:13px; line-height:1.55; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="research-header">
  <div class="kicker">QUANTITATIVE MACRO RESEARCH · LIVE PUBLIC DATA</div>
  <div class="title">Macro-ML Panel</div>
  <div class="subtitle">Macroeconometria + Machine Learning · 10 modelos · BCB/SGS + FRED · sem dados sintéticos</div>
</div>
""", unsafe_allow_html=True)

# Tabs intentionally use names only: the module order is visual, not part of the label.
tabs = st.tabs([name for name, _, _, _, _ in MODULES])
for tab, (name, module, description, formula, methodology) in zip(tabs, MODULES):
    with tab:
        st.markdown(f"""
        <div class="method-card">
          <div class="method-title">O que é · {name}</div>
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
