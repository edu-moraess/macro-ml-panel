"""
Módulo 9 — Detecção de anomalia macro (Isolation Forest)
Ideia: em vez de esperar um evento conhecido (2008, 2020, 2015-16) para
rotular "crise", o Isolation Forest sinaliza meses cujo comportamento
conjunto de juros, câmbio, inflação e desemprego destoa do padrão
histórico — um sistema de alerta precoce não-supervisionado.
"""
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from data_utils import get_bcb, align, SGS


def render():
    st.header("9 · Detecção de anomalia macro (Isolation Forest)")
    st.caption(
        "Painel multivariado (Selic, câmbio, IPCA, desemprego) — o modelo aprende o "
        "'normal' e sinaliza meses fora do padrão, sem precisar de rótulo de crise."
    )

    contamination = st.slider("Fração esperada de meses anômalos", 0.02, 0.15, 0.05)

    selic = get_bcb(SGS["selic_meta"]).resample("MS").mean()
    cambio = get_bcb(SGS["cambio_ptax"]).resample("MS").mean()
    ipca = get_bcb(SGS["ipca_mensal"]).resample("MS").mean()
    desemprego = get_bcb(SGS["desemprego_pnad"]).resample("MS").mean()

    df = align(
        selic.rename("selic"), cambio.rename("cambio"),
        ipca.rename("ipca"), desemprego.rename("desemprego"),
    )
    df["cambio_var"] = df["cambio"].pct_change() * 100
    df = df.dropna()

    feats = ["selic", "cambio_var", "ipca", "desemprego"]
    X = StandardScaler().fit_transform(df[feats])

    model = IsolationForest(contamination=contamination, random_state=0).fit(X)
    df["anomalia"] = model.predict(X) == -1
    df["score"] = -model.score_samples(X)  # maior = mais anômalo

    st.line_chart(df["score"])

    st.write(f"**{df['anomalia'].sum()} meses sinalizados como anômalos:**")
    st.dataframe(df.loc[df["anomalia"], feats + ["score"]].sort_values("score", ascending=False))
