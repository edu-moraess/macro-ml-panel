"""MACRO INTELLIGENCE dashboard — score, regime, contributions, historical validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from data_utils import FRED, SGS, align, data_status, get_bcb, get_fred
from core.macro_intelligence import MacroIntelligence
from core.yield_curve import load_history, historical_spreads


def render() -> None:
    st.subheader("MACRO INTELLIGENCE")
    st.caption("Quantitative macro research infrastructure — not investment advice. signal(t) → outcome(t+1…) anti-lookahead.")
    window = st.slider("Rolling window (months)", 36, 120, 60, 12, key="mi_window")
    try:
        selic = get_bcb(SGS["selic_meta"]).resample("MS").mean()
        ipca = get_bcb(SGS["ipca_mensal"]).resample("MS").mean()
        cambio = get_bcb(SGS["cambio_ptax"]).resample("MS").mean()
        unemp = get_bcb(SGS["desemprego_pnad"]).resample("MS").mean()
        ibc = get_bcb(SGS["ibc_br"]).resample("MS").mean()
        vix = get_fred(FRED["vix"]).resample("MS").mean()
        hy = get_fred(FRED["hy_spread"]).resample("MS").mean()
        spx = get_fred(FRED["sp500"]).resample("MS").last()
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc

    df = align(selic.rename("selic"), ipca.rename("ipca"), cambio.rename("fx"), unemp.rename("u"), ibc.rename("ibc"), vix.rename("vix"), hy.rename("hy"), spx.rename("spx"))
    if len(df) < max(48, window // 2):
        raise ValueError(f"Amostra alinhada insuficiente: {len(df)}")

    growth = df["ibc"].pct_change(3) * 100 - df["u"].diff(3)
    inflation = df["ipca"].diff(3)
    rates = df["selic"].diff(3)
    fx = -df["fx"].pct_change(3) * 100
    liquidity = -(df["hy"] + df["vix"])
    momentum = df["spx"].pct_change(12) * 100
    asset_ret = df["spx"].pct_change()

    curve_raw = None
    try:
        spreads = historical_spreads(load_history())
        if "2s10s_bps" in spreads.columns:
            curve_raw = spreads["2s10s_bps"]
    except Exception:
        pass

    out = MacroIntelligence(window=window).run(growth, inflation, rates, fx, liquidity, momentum, asset_ret, curve_raw=curve_raw)
    reg, sig, cp = out["regime"], out["signal"], out["curve_persistence"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MACRO SCORE", f"{sig['macro_score']:+.1f}" if sig["macro_score"] == sig["macro_score"] else "—")
    c2.metric("REGIME", reg.get("regime", "—"))
    c3.metric("REGIME PROB", f"{reg.get('probability', 0):.1f}%")
    c4.metric("SIGNAL CONF", f"{sig['confidence']:.1f}%")
    c5.metric("DURATION", f"{reg.get('duration_months', '—')}m")

    probs = reg.get("probabilities") or {}
    if probs:
        st.caption("  ·  ".join(f"{k} {v:.1f}%" for k, v in sorted(probs.items(), key=lambda x: -x[1])) + f"  ·  method={reg.get('method')}  ·  prev={reg.get('previous')}  ·  {reg.get('transition', '')}")

    st.subheader("FACTOR CONTRIBUTION")
    z, contrib = sig.get("z_latest") or {}, sig.get("contributions") or {}
    rows = []
    for name in ["Growth", "Inflation", "Rates", "FX", "Liquidity", "Momentum", "Curve"]:
        if name not in z and name not in contrib:
            continue
        rows.append({"Factor": name, "Z-Score": z.get(name, np.nan), "Contribution": contrib.get(name, np.nan), "Direction": "↑" if contrib.get(name, 0) > 0 else "↓" if contrib.get(name, 0) < 0 else "→"})
    if rows:
        st.dataframe(pd.DataFrame(rows).style.format({"Z-Score": "{:+.2f}", "Contribution": "{:+.3f}"}), use_container_width=True, hide_index=True)
        st.caption(sig.get("methodology", ""))

    st.subheader("CURVE STATE")
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("CURVE REGIME", cp.get("regime", "—"))
    cc2.metric("CURVE DURATION", f"{cp.get('duration', 0)}m")
    cc3.metric("PREVIOUS", str(cp.get("previous", "—")))

    score_s = sig.get("score_series")
    if score_s is not None and len(score_s):
        st.subheader("SIGNAL HISTORY")
        st.line_chart(score_s, height=240)

    hist_lab = out.get("history_labeled")
    if hist_lab is not None and len(hist_lab):
        st.subheader("REGIME HISTORY")
        st.line_chart(hist_lab.map({"RISK-OFF": -1, "NEUTRAL": 0, "RISK-ON": 1}).fillna(0).rename("regime_code"), height=180)

    st.subheader("HISTORICAL VALIDATION")
    st.caption("Forward outcomes from t+1 (anti-lookahead). Not investment advice.")
    val = out.get("validation") or {}
    if val.get("status") != "OK":
        st.warning(f"{val.get('status', 'SKIPPED')} — {val.get('detail', '')}")
    else:
        for key in ["event_risk_off", "event_risk_on"]:
            ev = val.get(key) or {}
            if ev.get("status") != "OK":
                st.caption(f"{ev.get('label', key)}: {ev.get('status')} n={ev.get('n', 0)}")
                continue
            st.markdown(f"**{ev.get('label')}** · N={ev.get('n')}")
            rows = [{"Horizon": h, "Avg": s.get("avg"), "Median": s.get("median"), "Hit Ratio": s.get("hit_ratio"), "Vol": s.get("vol"), "Avg MDD": s.get("mdd_avg")} for h, s in (ev.get("horizons") or {}).items()]
            if rows:
                st.dataframe(pd.DataFrame(rows).style.format({"Avg": "{:+.2%}", "Median": "{:+.2%}", "Hit Ratio": "{:.1%}", "Vol": "{:.2%}", "Avg MDD": "{:.2%}"}), use_container_width=True, hide_index=True)

    lineage = out.get("lineage") or {}
    with st.expander("DATA LINEAGE / REPRODUCIBILITY", expanded=False):
        st.json(lineage)

    st.caption(data_status("BCB/SGS + FRED", "macro intelligence panel", out["factors"].index.max()))
