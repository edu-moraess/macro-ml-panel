"""Yield Curve Intelligence — real FRED Treasury + GP + factors + history diagnostics."""
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from data_utils import data_status
from core.yield_curve import load_latest_curve,load_history,curve_factors,classify_curve_regime,fit_gp,historical_spreads,rolling_spread_stats,curve_regime_history,curve_regime_persistence

def _forward_rates(curve: pd.Series)->pd.DataFrame:
    pts=sorted((float(m),float(y)) for m,y in curve.items() if pd.notna(y)); rows=[]
    for (m1,y1),(m2,y2) in zip(pts[:-1],pts[1:]):
        if m2>m1: rows.append({"intervalo":f"{m1:g}y→{m2:g}y","forward_aprox_%":((1+y2/100)**m2/(1+y1/100)**m1)**(1/(m2-m1))*100-100})
    return pd.DataFrame(rows)

def render()->None:
    st.subheader("YIELD CURVE INTELLIGENCE")
    st.caption("Curva Treasury observada em vértices FRED reais; GP para interpolação, fatores Level/Slope/Curvature e diagnóstico histórico.")
    curve=load_latest_curve(); factors=curve_factors(curve); regime=classify_curve_regime(factors); persistence=None
    c1,c2,c3,c4=st.columns(4); c1.metric("CURVE REGIME",regime["regime"]); c2.metric("CONFIDENCE",f"{regime['confidence']:.1f}%"); c3.metric("10Y−2Y",f"{factors['slope_10y2y_bps']:+.0f} bps"); c4.metric("10Y−30Y",f"{factors['slope_30y10y_bps']:+.0f} bps")
    c5,c6,c7=st.columns(3); c5.metric("LEVEL",f"{factors['level']:.2f}%"); c6.metric("CURVATURE",f"{factors['curvature_bps']:+.0f} bps"); c7.metric("VERTICES",f"{factors['n_vertices']}")
    st.write("**Forward rates implícitas entre vértices:**"); st.dataframe(_forward_rates(curve).round(3),use_container_width=True)
    gp_out=fit_gp(curve)
    fig,ax=plt.subplots(figsize=(8,3.5)); ax.fill_between(gp_out["grid_maturity"],gp_out["mean"]-1.96*gp_out["std"],gp_out["mean"]+1.96*gp_out["std"],alpha=.2,label="95% band"); ax.plot(gp_out["grid_maturity"],gp_out["mean"],lw=1.5,label="GP mean"); ax.scatter(gp_out["observed_X"],gp_out["observed_y"],zorder=5,s=28,label="Observed"); ax.set_xlabel("Maturity (years)"); ax.set_ylabel("Yield (%)"); ax.legend(fontsize=8,frameon=False); ax.grid(True,alpha=.25); st.pyplot(fig); plt.close(fig)
    k1,k2,k3,k4=st.columns(4); k1.metric("Length Scale",f"{gp_out['length_scale']:.2f}"); k2.metric("Noise",f"{gp_out['noise']:.4f}"); k3.metric("RMSE",f"{gp_out['rmse']:.4f}"); k4.metric("MAE",f"{gp_out['mae']:.4f}")
    hist=load_history(); spreads=historical_spreads(hist); st.subheader("Curve History")
    cols=[c for c in ["2s10s_bps","3m10y_bps","10s30s_bps"] if c in spreads.columns]
    if cols:
        st.line_chart(spreads[cols],height=280); stats=rolling_spread_stats(spreads[cols],min(60,max(12,len(spreads)//3))); regimes=curve_regime_history(spreads); persistence=curve_regime_persistence(regimes)
        z=stats.get("2s10s_bps_z");
        if z is not None and not z.dropna().empty: st.metric("2s10s percentile (rolling)",f"{float((z.dropna()<=z.dropna().iloc[-1]).mean()*100):.0f}%")
        st.write("**Persistência do regime:**"); st.dataframe(pd.DataFrame([persistence]),use_container_width=True)
        st.write("**Z-score rolling dos spreads:**"); st.line_chart(stats[[c for c in stats.columns if c.endswith("_z")]],height=240)
    st.caption(data_status("FRED","DGS3MO…DGS30",pd.Timestamp.now().normalize()))
