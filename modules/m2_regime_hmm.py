"""
Módulo 2 — Regimes de política monetária via Hidden Markov Model
HMM não-supervisionado com diagnóstico de seleção, persistência e estabilidade.
"""
import numpy as np
import pandas as pd
import streamlit as st

from data_utils import get_bcb, get_fred, SGS, FRED

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_OK = True
except ImportError:
    from sklearn.mixture import GaussianMixture
    HMM_OK = False


def _fit_hmm(X, n_states, seed):
    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=300, random_state=seed)
    model.fit(X)
    return model, model.predict(X)


def _duration_summary(states, index):
    rows = []
    start = 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            rows.append({"regime": int(states[start]), "início": index[start].date(), "fim": index[i - 1].date(), "duração_meses": i - start})
            start = i
    return pd.DataFrame(rows)


def render():
    st.header("2 · Regimes de política monetária (HMM)")
    st.caption("HMM não-supervisionado sobre a variação mensal de juros, com seleção de complexidade, persistência e estabilidade entre sementes.")

    fonte = st.radio("Série", ["Selic (BR)", "Fed Funds (US)"], horizontal=True)
    n_states = st.slider("Número de regimes", 2, 4, 2)

    s = get_bcb(SGS["selic_meta"]) if fonte.startswith("Selic") else get_fred(FRED["fedfunds"])
    s = s.resample("MS").mean().dropna()
    X = s.diff().dropna().to_numpy().reshape(-1, 1)
    idx = s.index[1:]

    if HMM_OK:
        model, states = _fit_hmm(X, n_states, 0)
        scores = []
        for k in range(2, 5):
            if len(X) <= k * 5:
                continue
            m, _ = _fit_hmm(X, k, 0)
            ll = float(m.score(X))
            p = k * (k - 1) + 2 * k + (k - 1)
            scores.append({"regimes": k, "AIC": 2 * p - 2 * ll, "BIC": np.log(len(X)) * p - 2 * ll})
        st.write("**Seleção de complexidade (AIC/BIC):**")
        st.dataframe(pd.DataFrame(scores).round(2), use_container_width=True)

        seed_rows = []
        for seed in [0, 7, 42]:
            _, stt = _fit_hmm(X, n_states, seed)
            seed_rows.append({"seed": seed, **{f"regime_{k}": float((stt == k).mean()) for k in range(n_states)}})
        st.write("**Estabilidade da ocupação por semente:**")
        st.dataframe(pd.DataFrame(seed_rows).round(3), use_container_width=True)
    else:
        st.info("`hmmlearn` não instalado — GaussianMixture é apenas aproximação e não modela persistência temporal.")
        model = GaussianMixture(n_components=n_states, random_state=0).fit(X)
        states = model.predict(X)

    out = pd.DataFrame({"nivel": s.iloc[1:].values, "variação": X.ravel(), "regime": states}, index=idx)
    st.line_chart(out["nivel"])
    st.bar_chart(out["regime"])
    resumo = out.groupby("regime")["variação"].agg(["mean", "std", "count"])
    resumo.columns = ["Δ médio", "volatilidade", "nº meses"]
    st.write("**Caracterização dos regimes:**")
    st.dataframe(resumo.round(4), use_container_width=True)
    st.write("**Duração dos episódios:**")
    st.dataframe(_duration_summary(states, idx), use_container_width=True)
    if HMM_OK:
        st.write("**Matriz de transição:**")
        st.dataframe(pd.DataFrame(model.transmat_).round(3), use_container_width=True)
        st.caption("Os rótulos 0…N não têm significado econômico fixo; compare média, volatilidade e duração antes de nomear os regimes.")
