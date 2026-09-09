"""
Módulo 10 — Curva de juros via Gaussian Process
Ideia: alternativa não-paramétrica bayesiana ao Nelson-Siegel-Svensson.
Em vez de assumir a forma funcional do NSS, o Gaussian Process aprende a
curva diretamente dos vértices observados e entrega, de graça, a incerteza
(intervalo de confiança) em cada prazo — algo que o NSS não dá nativamente.
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

# Vértices ilustrativos (edite com os dados reais da sua curva)
DEFAULT_CURVE = pd.DataFrame({
    "prazo_anos": [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30],
    "taxa_pct": [10.75, 10.9, 11.1, 11.4, 11.5, 11.7, 11.9, 12.1, 12.3, 12.35],
})


def render():
    st.header("10 · Curva de juros via Gaussian Process")
    st.caption(
        "GP como alternativa ao NSS/BEIR: ajusta a curva e entrega a banda de "
        "incerteza em cada prazo. Edite os vértices abaixo com seus próprios dados."
    )

    curva = st.data_editor(DEFAULT_CURVE, num_rows="dynamic", use_container_width=True)
    curva = curva.dropna().sort_values("prazo_anos")

    X = curva["prazo_anos"].to_numpy().reshape(-1, 1)
    y = curva["taxa_pct"].to_numpy()

    kernel = C(1.0) * RBF(length_scale=5.0) + WhiteKernel(noise_level=0.01)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5).fit(X, y)

    grid = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    mean, std = gp.predict(grid, return_std=True)

    fig, ax = plt.subplots()
    ax.plot(grid.ravel(), mean, label="curva ajustada (GP)")
    ax.fill_between(grid.ravel(), mean - 1.96 * std, mean + 1.96 * std, alpha=0.25,
                     label="IC 95%")
    ax.scatter(X.ravel(), y, color="black", zorder=5, label="vértices observados")
    ax.set_xlabel("Prazo (anos)")
    ax.set_ylabel("Taxa (%)")
    ax.legend()
    st.pyplot(fig)

    st.caption(f"Kernel ajustado: `{gp.kernel_}`")
