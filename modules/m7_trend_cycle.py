"""
Módulo 7 — Decomposição tendência-ciclo via Autoencoder
Ideia: alternativa não-linear ao filtro HP. Um autoencoder com gargalo de
1 neurônio, treinado em janelas móveis da série, aprende a reconstruir
cada janela a partir de uma única "tendência latente" — sem o viés de
fim de amostra do HP e sem impor linearidade.
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from data_utils import get_bcb, SGS


def _windows(x: np.ndarray, w: int) -> np.ndarray:
    n = len(x)
    idx = np.arange(w)[None, :] + np.arange(n - w + 1)[:, None]
    return x[idx]


def _bottleneck(model: MLPRegressor, X: np.ndarray) -> np.ndarray:
    """Forward pass manual até a camada de gargalo (2ª camada oculta)."""
    a = np.tanh(X @ model.coefs_[0] + model.intercepts_[0])
    a = np.tanh(a @ model.coefs_[1] + model.intercepts_[1])
    return a.ravel()


def render():
    st.header("7 · Tendência-ciclo via Autoencoder")
    st.caption(
        "Autoencoder com gargalo de 1 neurônio sobre janelas móveis do IPCA — "
        "a ativação do gargalo é a 'tendência latente' não-linear da série."
    )

    w = st.slider("Tamanho da janela (meses)", 7, 25, 13, step=2)

    s = get_bcb(SGS["ipca_mensal"]).resample("MS").mean().dropna()
    values = s.to_numpy().reshape(-1, 1)
    scaler = StandardScaler().fit(values)
    x = scaler.transform(values).ravel()

    X = _windows(x, w)
    model = MLPRegressor(
        hidden_layer_sizes=(8, 1, 8), activation="tanh", max_iter=3000, random_state=0
    ).fit(X, X)

    bottleneck = _bottleneck(model, X)
    # o gargalo é um fator latente não-linear (escala arbitrária); calibra de volta
    # para a escala da série via uma regressão linear simples (OLS de 1 variável)
    centro = x[w // 2 : w // 2 + len(bottleneck)]
    a, b = np.polyfit(bottleneck, centro, 1)
    trend_ae = scaler.inverse_transform((a * bottleneck + b).reshape(-1, 1)).ravel()

    idx_center = s.index[w // 2 : w // 2 + len(bottleneck)]
    trend_ma = s.rolling(w, center=True).mean().reindex(idx_center)

    df = pd.DataFrame(
        {
            "observado": s.reindex(idx_center).values,
            "tendência (autoencoder)": trend_ae,
            "tendência (média móvel)": trend_ma.values,
        },
        index=idx_center,
    )
    st.line_chart(df)

    ciclo = df["observado"] - df["tendência (autoencoder)"]
    st.write("**Componente cíclico (observado − tendência do autoencoder):**")
    st.line_chart(ciclo.rename("ciclo"))
