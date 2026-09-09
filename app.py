"""
Macro-ML Panel — 10 exemplos compactos de Macroeconometria + Machine Learning.
Executar com: streamlit run app.py
"""
import streamlit as st

from modules import (
    m1_recession,
    m2_regime_hmm,
    m3_structural_break,
    m4_phillips_curve,
    m5_nowcasting,
    m6_clustering,
    m7_trend_cycle,
    m8_nlp_sentiment,
    m9_anomaly,
    m10_gp_yield_curve,
)

st.set_page_config(page_title="Macro-ML Panel", layout="wide")

MODULOS = {
    "1 · Recessão (Elastic Net Logit)": m1_recession,
    "2 · Regimes de juros (HMM)": m2_regime_hmm,
    "3 · Quebra estrutural (changepoint)": m3_structural_break,
    "4 · Curva de Phillips (Gradient Boosting)": m4_phillips_curve,
    "5 · Nowcasting de atividade (Gradient Boosting)": m5_nowcasting,
    "6 · Regimes cambiais (K-Means)": m6_clustering,
    "7 · Tendência-ciclo (Autoencoder)": m7_trend_cycle,
    "8 · Sentimento hawkish/dovish (NLP)": m8_nlp_sentiment,
    "9 · Anomalia macro (Isolation Forest)": m9_anomaly,
    "10 · Curva de juros (Gaussian Process)": m10_gp_yield_curve,
}

st.sidebar.title("📊 Macro-ML Panel")
st.sidebar.caption("Macroeconometria + Machine Learning em 10 exemplos compactos.")
escolha = st.sidebar.radio("Módulo", list(MODULOS.keys()))

st.sidebar.divider()
st.sidebar.caption(
    "Dados 100% reais: BCB/SGS e FRED (APIs públicas, sem chave). "
    "Sem fallback sintético — exige conexão com a internet."
)

try:
    MODULOS[escolha].render()
except RuntimeError as e:
    st.error(f"⚠️ Não foi possível obter dados reais para este módulo.\n\n**Detalhe:** {e}")
    st.caption("Verifique sua conexão com a internet ou tente novamente em alguns minutos "
               "(a API do BCB/SGS às vezes tem instabilidade pontual).")
