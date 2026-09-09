"""Quant Lab V2 — Factors · Signals · Risk · TS · Backtest · Portfolio using core engines."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from data_utils import FRED, SGS, data_status, get_bcb, get_fred
from core.factors import FactorEngine
from core.risk import RiskEngine
from core.backtest import BacktestEngine
from core.portfolio import PortfolioEngine


def render() -> None:
    st.subheader("QUANT LAB")
    st.caption("Engines: Factor · Signal · Risk · Time-Series · Backtest · Portfolio")

    n = st.slider("Rolling window", 36, 120, 60, 12, key="v2_lab_n")

    try:
        x = pd.concat(
            {
                "IBC": get_bcb(SGS["ibc_br"]).resample("MS").mean(),
                "U": get_bcb(SGS["desemprego_pnad"]).resample("MS").mean(),
                "IPCA": get_bcb(SGS["ipca_mensal"]).resample("MS").mean(),
                "Selic": get_bcb(SGS["selic_meta"]).resample("MS").mean(),
                "FX": get_bcb(SGS["cambio_ptax"]).resample("MS").mean(),
                "VIX": get_fred(FRED["vix"]).resample("MS").mean(),
                "HY": get_fred(FRED["hy_spread"]).resample("MS").mean(),
                "SP500": get_fred(FRED["sp500"]).resample("MS").last(),
            },
            axis=1,
        ).dropna()
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc

    if len(x) < n:
        raise ValueError(f"Amostra insuficiente: {len(x)} < {n}")

    growth = x["IBC"].pct_change(3) * 100 - x["U"].diff(3)
    inf = x["IPCA"].diff(3)
    rates = x["Selic"].diff(3)
    fx = -x["FX"].pct_change(3) * 100
    liq = -(x["HY"] + x["VIX"])
    mom = x["SP500"].pct_change(12) * 100

    fe = FactorEngine(window=n)
    factors = fe.build(growth, inf, rates, fx, liq, mom)
    state = fe.latest_state(factors)
    score = fe.composite(factors)

    tabs = st.tabs(["FACTORS", "SIGNALS", "RISK", "TIME SERIES", "BACKTEST", "PORTFOLIO"])

    with tabs[0]:
        st.dataframe(
            pd.Series(state["z_latest"]).sort_values(ascending=False).to_frame("z-score").style.format("{:+.2f}"),
            use_container_width=True,
        )
        st.line_chart(factors, height=340)

    with tabs[1]:
        c = st.columns(4)
        c[0].metric("MACRO SCORE", f"{state['score']:+.2f}")
        c[1].metric("SIGNAL", state["regime"])
        c[2].metric("CONFIDENCE", f"{state['confidence']:.1f}%")
        c[3].metric("PERCENTILE", f"{state.get('percentile', float('nan')):.0f}%")
        st.line_chart(score.rename("composite"), height=280)

    with tabs[2]:
        r = x["SP500"].pct_change().dropna()
        summary = RiskEngine.summary(r)
        if summary.get("status") != "OK":
            st.error(summary.get("status", "FALHA"))
        else:
            c = st.columns(6)
            c[0].metric("VOL 20M", f"{summary['vol_20']:.1%}")
            c[1].metric("EWMA", f"{summary['ewma_vol']:.1%}")
            c[2].metric("VaR 95%", f"{summary['var_95']:.2%}")
            c[3].metric("ES 95%", f"{summary['es_95']:.2%}")
            c[4].metric("MAX DD", f"{summary['max_dd']:.2%}")
            c[5].metric("SHARPE", f"{summary['sharpe']:.2f}")
            st.line_chart(
                pd.DataFrame({"Vol": summary["vol_series"], "EWMA": summary["ewma_series"], "DD": summary["dd_series"]}),
                height=300,
            )

    with tabs[3]:
        s = x["SP500"].pct_change().dropna()
        if len(s) > 36:
            lag = s.shift(1)
            valid = pd.concat([s.rename("y"), lag.rename("x")], axis=1).dropna()
            beta = np.cov(valid.y, valid.x)[0, 1] / np.var(valid.x) if np.var(valid.x) > 0 else np.nan
            resid = valid.y - beta * valid.x
            st.metric("AR(1) φ̂", f"{beta:.3f}")
            st.line_chart(pd.DataFrame({"ret": s, "resid": resid}), height=260)
        else:
            st.warning("AMOSTRA INSUFICIENTE")

    with tabs[4]:
        bt = BacktestEngine(lag=1, threshold=0.15)
        result = bt.run(score, x["SP500"].pct_change())
        c = st.columns(5)
        c[0].metric("CAGR", f"{result['cagr']:.1%}" if result["cagr"] == result["cagr"] else "—")
        c[1].metric("SHARPE", f"{result['sharpe']:.2f}")
        c[2].metric("MAX DD", f"{result['max_dd']:.1%}")
        c[3].metric("HIT", f"{result['hit_ratio']:.1%}")
        c[4].metric("TRADES", f"{result['n_trades']}")
        st.line_chart(pd.DataFrame({"Strategy": result["equity"], "B&H": result["benchmark"]}), height=300)
        st.caption(f"Lag={result['lag']} · threshold={result['threshold']}")

    with tabs[5]:
        try:
            assets = pd.concat(
                {
                    "SP500": get_fred(FRED["sp500"]),
                    "WTI": get_fred(FRED["wti"]),
                    "Gold": get_fred(FRED["gold"]),
                },
                axis=1,
            ).resample("MS").last().pct_change().dropna()
            method = st.selectbox("Method", ["equal", "inverse_vol", "min_var", "risk_parity"], key="port_m")
            pe = PortfolioEngine(method=method, max_weight=0.6, min_weight=0.05)
            ev = pe.evaluate(assets)
            st.dataframe(ev["weights"].to_frame().style.format("{:.1%}"), use_container_width=True)
            c = st.columns(3)
            c[0].metric("VOL", f"{ev['vol']:.1%}")
            c[1].metric("SHARPE", f"{ev['sharpe']:.2f}")
            c[2].metric("MAX DD", f"{ev['max_dd']:.1%}")
            st.line_chart(ev["equity"], height=260)
        except Exception as exc:
            st.error(f"DADOS INDISPONÍVEIS / AMOSTRA INSUFICIENTE — {exc}")

    st.caption(data_status("BCB/SGS + FRED", "factors + risk assets", factors.index.max()))
