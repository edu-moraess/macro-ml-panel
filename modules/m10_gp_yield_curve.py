"""Yield Curve Intelligence — real FRED Treasury + GP construction + curve regime."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from data_utils import data_status
from core.yield_curve import (
    load_latest_curve,
    load_history,
    curve_factors,
    classify_curve_regime,
    fit_gp,
    historical_spreads,
)


def render() -> None:
    st.subheader("YIELD CURVE INTELLIGENCE")
    st.caption(
        "Construção/suavização da estrutura a termo via Gaussian Process sobre vértices FRED reais. "
        "O GP não é um modelo causal de previsão macroeconômica."
    )

    try:
        curve = load_latest_curve()
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc

    factors = curve_factors(curve)
    regime = classify_curve_regime(factors)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CURVE REGIME", regime["regime"])
    c2.metric("CONFIDENCE", f"{regime['confidence']:.1f}%")
    c3.metric(
        "10Y−2Y",
        f"{factors['slope_10y2y_bps']:+.0f} bps"
        if factors["slope_10y2y_bps"] == factors["slope_10y2y_bps"]
        else "—",
    )
    c4.metric(
        "10Y−30Y",
        f"{factors['slope_30y10y_bps']:+.0f} bps"
        if factors.get("slope_30y10y_bps") == factors.get("slope_30y10y_bps")
        else "—",
    )

    c5, c6, c7 = st.columns(3)
    c5.metric("LEVEL", f"{factors['level']:.2f}%")
    c6.metric(
        "CURVATURE",
        f"{factors['curvature_bps']:+.0f} bps"
        if factors["curvature_bps"] == factors["curvature_bps"]
        else "—",
    )
    c7.metric("VERTICES", f"{factors['n_vertices']}")

    try:
        gp_out = fit_gp(curve)
    except Exception as exc:
        st.error(f"MODELO NÃO ESTIMÁVEL — {exc}")
        gp_out = None

    if gp_out is not None:
        st.subheader("Gaussian Process Curve Construction")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.fill_between(
            gp_out["grid_maturity"],
            gp_out["mean"] - 1.96 * gp_out["std"],
            gp_out["mean"] + 1.96 * gp_out["std"],
            alpha=0.2,
            color="#174EA6",
            label="95% band",
        )
        ax.plot(gp_out["grid_maturity"], gp_out["mean"], color="#174EA6", lw=1.5, label="GP mean")
        ax.scatter(
            gp_out["observed_X"], gp_out["observed_y"], color="#17202A", zorder=5, s=28, label="Observed"
        )
        ax.set_xlabel("Maturity (years)")
        ax.set_ylabel("Yield (%)")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(True, alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "Length Scale",
            f"{gp_out['length_scale']:.2f}" if gp_out["length_scale"] == gp_out["length_scale"] else "—",
        )
        k2.metric("Noise", f"{gp_out['noise']:.4f}" if gp_out["noise"] == gp_out["noise"] else "—")
        k3.metric("RMSE", f"{gp_out['rmse']:.4f}")
        k4.metric("MAE", f"{gp_out['mae']:.4f}")
        st.caption(
            f"Kernel: {gp_out['kernel_str']}\n\n"
            "length_scale = escala característica de correlação entre maturidades (anos), "
            "não memória temporal. GP = suavização/interpolação da curva observada."
        )
        with st.expander("Residuals"):
            resid_df = pd.DataFrame(
                {"maturity": gp_out["observed_X"], "residual": gp_out["residuals"]}
            ).set_index("maturity")
            st.dataframe(resid_df.style.format("{:+.4f}"), use_container_width=True)

    try:
        hist = load_history()
        spreads = historical_spreads(hist)
        st.subheader("Curve History")
        cols = [c for c in ["2s10s_bps", "3m10y_bps", "10s30s_bps"] if c in spreads.columns]
        if cols:
            st.line_chart(spreads[cols], height=280)
            st.caption("Spreads em bps · mensal · vértices FRED reais")
    except Exception as exc:
        st.warning(f"DATA UNAVAILABLE — histórico de curva: {exc}")

    st.caption(data_status("FRED", "DGS3MO…DGS30", pd.Timestamp.now().normalize()))
