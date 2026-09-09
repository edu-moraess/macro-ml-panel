"""
Módulo 2 — Regimes de política monetária via Hidden Markov Model
Ideia: em vez de rotular "aperto"/"afrouxamento" na mão, um HMM Gaussiano
descobre os regimes de forma não-supervisionada, junto com a matriz de
transição entre eles.
"""
import pandas as pd
import streamlit as st

from data_utils import get_bcb, get_fred, SGS, FRED

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_OK = True
except ImportError:
    from sklearn.mixture import GaussianMixture
    HMM_OK = False


def render():
    st.header("2 · Regimes de política monetária (HMM)")
    st.caption(
        "HMM não-supervisionado sobre a variação mensal de juros — descobre os "
        "regimes e a probabilidade de transição entre eles, sem rótulo prévio."
    )

    fonte = st.radio("Série", ["Selic (BR)", "Fed Funds (US)"], horizontal=True)
    n_states = st.slider("Número de regimes", 2, 4, 2)

    s = get_bcb(SGS["selic_meta"]) if fonte.startswith("Selic") else get_fred(FRED["fedfunds"])
    s = s.resample("MS").mean().dropna()
    X = s.diff().dropna().to_numpy().reshape(-1, 1)

    if HMM_OK:
        model = GaussianHMM(
            n_components=n_states, covariance_type="diag", n_iter=200, random_state=0
        ).fit(X)
        states = model.predict(X)
    else:
        st.info(
            "`hmmlearn` não instalado — usando GaussianMixture como aproximação "
            "(sem modelar a persistência temporal entre regimes)."
        )
        model = GaussianMixture(n_components=n_states, random_state=0).fit(X)
        states = model.predict(X)

    out = pd.DataFrame(
        {"nivel": s.iloc[1:].values, "variação": X.ravel(), "regime": states},
        index=s.index[1:],
    )

    st.line_chart(out["nivel"])
    st.bar_chart(out["regime"])

    resumo = out.groupby("regime")["variação"].agg(["mean", "std", "count"])
    resumo.columns = ["Δ médio", "volatilidade", "nº meses"]
    st.write("**Caracterização de cada regime (variação mensal de juros):**")
    st.dataframe(resumo)

    if HMM_OK:
        st.write("**Matriz de transição (probabilidade de mudar de regime):**")
        st.dataframe(pd.DataFrame(model.transmat_).round(3))
