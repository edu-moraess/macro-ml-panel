"""Module 9 — macro anomaly detection via Isolation Forest."""
from __future__ import annotations

import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from data_utils import SGS, align, data_status, get_bcb


def render() -> None:
    """Estimate multivariate macro anomalies using only BCB/SGS observations."""
    st.header("9 · Detecção de anomalia macro")
    st.caption(
        "Isolation Forest sobre Selic, câmbio, IPCA e desemprego. O modelo aprende o "
        "padrão histórico observado e sinaliza meses estatisticamente incomuns."
    )

    contamination = st.slider("Fração esperada de meses anômalos", 0.02, 0.15, 0.05, step=0.01)

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
    if len(df) < 60:
        raise ValueError(f"Apenas {len(df)} observações mensais disponíveis; mínimo recomendado: 60.")

    feats = ["selic", "cambio_var", "ipca", "desemprego"]
    X = StandardScaler().fit_transform(df[feats])
    model = IsolationForest(contamination=contamination, random_state=42).fit(X)
    df["anomalia"] = model.predict(X) == -1
    df["score"] = -model.score_samples(X)

    c1, c2, c3 = st.columns(3)
    c1.metric("Observações", f"{len(df):,}")
    c2.metric("Meses anômalos", f"{int(df['anomalia'].sum())}")
    c3.metric("Última leitura", f"{df.index.max():%b/%Y}")

    st.line_chart(df["score"], height=320)
    st.caption(data_status("Banco Central do Brasil · SGS", "432 / 1 / 433 / 24369", df.index.max()))

    anomalies = df.loc[df["anomalia"], feats + ["score"]].sort_values("score", ascending=False)
    st.write("**Observações sinalizadas**")
    st.dataframe(anomalies, use_container_width=True)
    st.info(
        "Interpretação: anomalia estatística não equivale automaticamente a crise. "
        "O Isolation Forest é sensível à janela amostral, escala e contaminação escolhida."
    )
