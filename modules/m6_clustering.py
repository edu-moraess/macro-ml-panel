"""
Módulo 6 — Clusterização de regimes cambiais (K-Means)
Ideia: em vez de usar a classificação declarada de regime cambial (de facto
vs de jure, segundo o FMI), deixamos o K-Means agrupar países pelo
comportamento observado (volatilidade cambial, diferencial de juros,
colchão de reservas) — a classificação "revelada" pelos dados.
"""
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Painel ilustrativo (editável) — substitua pelos seus próprios dados
DEFAULT_PANEL = pd.DataFrame({
    "pais": ["Brasil", "México", "Argentina", "Chile", "EUA", "Suíça",
             "Turquia", "Japão", "Índia", "África do Sul"],
    "vol_cambial": [12, 10, 35, 11, 6, 5, 28, 7, 8, 14],
    "diferencial_juros": [9.0, 6.5, 30.0, 4.0, 0.0, -1.5, 20.0, -0.3, 3.0, 5.5],
    "reservas_pib": [18, 15, 8, 14, 3, 10, 9, 25, 17, 12],
})


def render():
    st.header("6 · Clusterização de regimes cambiais (K-Means)")
    st.caption(
        "Agrupa países pelo comportamento cambial observado, não pela classificação "
        "declarada. Edite a tabela abaixo com seus próprios dados se quiser."
    )

    panel = st.data_editor(DEFAULT_PANEL, num_rows="dynamic", use_container_width=True)
    k = st.slider("Número de clusters", 2, 5, 3)

    feats = ["vol_cambial", "diferencial_juros", "reservas_pib"]
    panel = panel.dropna(subset=feats)
    X = StandardScaler().fit_transform(panel[feats])

    model = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    panel = panel.assign(cluster=model.labels_)

    fig, ax = plt.subplots()
    scatter = ax.scatter(
        panel["vol_cambial"], panel["diferencial_juros"], c=panel["cluster"], cmap="tab10", s=90
    )
    for _, row in panel.iterrows():
        ax.annotate(row["pais"], (row["vol_cambial"], row["diferencial_juros"]), fontsize=8)
    ax.set_xlabel("Volatilidade cambial")
    ax.set_ylabel("Diferencial de juros")
    st.pyplot(fig)

    st.write("**Países por cluster:**")
    st.dataframe(panel.sort_values("cluster"))
