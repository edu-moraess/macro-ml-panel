"""Module 10 — real Treasury yield curve fitted with a Gaussian Process."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

from data_utils import data_status, get_fred

FRED_TREASURY = {
    0.25: "DGS3MO", 0.5: "DGS6MO", 1.0: "DGS1", 2.0: "DGS2", 3.0: "DGS3",
    5.0: "DGS5", 7.0: "DGS7", 10.0: "DGS10", 20.0: "DGS20", 30.0: "DGS30",
}


def render() -> None:
    """Fit a GP to the latest real US Treasury curve observations."""
    st.header("10 · Curva de juros via Gaussian Process")
    st.caption(
        "Ajuste não-paramétrico da curva do Treasury com banda de incerteza. "
        "Os vértices são observações reais do FRED, não valores ilustrativos."
    )

    series = {maturity: get_fred(code).dropna().iloc[-1] for maturity, code in FRED_TREASURY.items()}
    curva = pd.Series(series, name="taxa_pct").sort_index()
    curva.index.name = "prazo_anos"
    curva = curva.dropna()
    if len(curva) < 6:
        raise ValueError("Menos de 6 vértices reais disponíveis para ajustar a curva.")

    X = curva.index.to_numpy(dtype=float).reshape(-1, 1)
    y = curva.to_numpy(dtype=float)
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=5.0) + WhiteKernel(noise_level=0.01)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=42).fit(X, y)

    grid = np.linspace(X.min(), X.max(), 240).reshape(-1, 1)
    mean, std = gp.predict(grid, return_std=True)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(grid.ravel(), mean, label="GP ajustado", linewidth=2)
    ax.fill_between(grid.ravel(), mean - 1.96 * std, mean + 1.96 * std, alpha=0.2, label="IC 95%")
    ax.scatter(X.ravel(), y, color="black", zorder=5, label="Treasury observado")
    ax.set_xlabel("Prazo (anos)")
    ax.set_ylabel("Yield (%)")
    ax.grid(alpha=0.15)
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    st.dataframe(curva.rename("yield_%").to_frame().T, use_container_width=True)
    st.caption(data_status("Federal Reserve Bank of St. Louis · FRED", "DGS3MO…DGS30", pd.Timestamp.now().normalize()))
    st.caption(f"Kernel ajustado: `{gp.kernel_}` · Interpretação: curva observada, não previsão causal.")
