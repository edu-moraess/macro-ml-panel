"""Module 6 — K-Means clustering of observed FX behavior."""
from __future__ import annotations

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from data_utils import data_status, get_fred

FX = {
    "Brasil": "EXBZUS", "México": "EXMXUS", "China": "EXCHUS",
    "Japão": "EXJPUS", "Índia": "EXINUS", "África do Sul": "EXSFUS",
}


def render() -> None:
    """Cluster countries using features derived only from real FRED FX observations."""
    st.header("6 · Clusterização de regimes cambiais (K-Means)")
    st.caption("K-Means agrupa países pelo comportamento observado da moeda frente ao dólar. Features são derivadas exclusivamente de séries reais FRED.")

    k = st.slider("Número de clusters", 2, 5, 3)
    rows: list[dict[str, float | str]] = []
    latest = None
    for country, code in FX.items():
        s = get_fred(code).resample("MS").mean().dropna()
        if len(s) < 24:
            continue
        latest = s.index.max() if latest is None else min(latest, s.index.max())
        ret = s.pct_change()
        rows.append({
            "pais": country,
            "vol_cambial": ret.rolling(12).std().iloc[-1] * 100,
            "retorno_3m": (s.iloc[-1] / s.iloc[-4] - 1) * 100,
            "retorno_12m": (s.iloc[-1] / s.iloc[-13] - 1) * 100,
        })

    panel = pd.DataFrame(rows).dropna()
    if len(panel) < k + 2:
        raise ValueError("Amostra real insuficiente para o número de clusters escolhido.")

    feats = ["vol_cambial", "retorno_3m", "retorno_12m"]
    X = StandardScaler().fit_transform(panel[feats])
    model = KMeans(n_clusters=k, n_init=20, random_state=42).fit(X)
    panel = panel.assign(cluster=model.labels_)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.scatter(panel["vol_cambial"], panel["retorno_12m"], c=panel["cluster"], cmap="tab10", s=90)
    for _, row in panel.iterrows():
        ax.annotate(row["pais"], (row["vol_cambial"], row["retorno_12m"]), fontsize=8)
    ax.set_xlabel("Volatilidade cambial — rolling 12m (%)")
    ax.set_ylabel("Retorno cambial — 12m (%)")
    ax.grid(alpha=0.15)
    st.pyplot(fig, clear_figure=True)

    st.dataframe(panel.sort_values("cluster"), use_container_width=True)
    if latest is not None:
        st.caption(data_status("Federal Reserve Bank of St. Louis · FRED", ", ".join(FX.values()), latest))
    st.info("Limitação: os clusters descrevem comportamento cambial observado; não constituem classificação oficial de regime cambial nem implicam causalidade.")
