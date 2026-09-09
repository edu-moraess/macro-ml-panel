"""Quant Lab — research tools consuming the canonical macro state."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.macro_state import build_macro_state, load_macro_panel
from core.risk import RiskEngine
from core.backtest import BacktestEngine
from core.portfolio import PortfolioEngine
from data_utils import FRED, data_status, get_fred


def render() -> None:
    st.subheader("QUANT LAB")
    st.caption("Factor · Signal · Risk · Time-Series · Backtest · Portfolio")

    n = st.slider("Rolling window", 36, 120, 60, 12, key="quant_lab_window")
    state = build_macro_state(load_macro_panel(), window=n)
    factors = state["factors"]
    signal = state["signal"]
    x = state["panel"]
    score = signal["score_series"]

    tabs = st.tabs(["FACTORS", "SIGNALS", "RISK", "TIME SERIES", "BACKTEST", "PORTFOLIO"])

    with tabs[0]:
        st.dataframe(
            pd.Series(signal["z_latest"]).sort_values(ascending=False).to_frame("z-score").style.format("{:+.2f}"),
            use_container_width=True,
        )
        st.line_chart(factors, height=340)

    with tabs[1]:
        c = st.columns(4)
        c[0].metric("MACRO SCORE", f"{signal['macro_score']:+.1f}")
        c[1].metric("REGIME HINT", signal.get("regime_hint", "N/A"))
        c[2].metric("SIGNAL CONFIDENCE", f"{signal['confidence']:.1f}%")
        percentile = float((score <= score.iloc[-1]).mean() * 100) if len(score) else float("nan")
        c[3].metric("PERCENTILE", f"{percentile:.0f}%" if percentile == percentile else "—")
        st.line_chart(score.rename("MACRO SCORE"), height=280)
        st.caption("A confiança é um indicador composto de cobertura, concordância entre fatores, regime e qualidade dos dados; não é probabilidade de retorno.")

    with tabs[2]:
        r = x["SP500"].pct_change().dropna()
        summary = RiskEngine.summary(r)
        if summary.get("status") != "OK":
            st.error(summary.get("status", "FALHA CONTROLADA"))
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
            beta = float(valid["y"].cov(valid["x"]) / valid["x"].var()) if valid["x"].var() > 0 else float("nan")
            resid = valid["y"] - beta * valid["x"]
            st.metric("AR(1) φ̂", f"{beta:.3f}" if beta == beta else "—")
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
        st.caption(f"Lag={result['lag']} · threshold={result['threshold']} · custos={result['cost_bps']:.1f} bps")

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
            method = st.selectbox("Method", ["equal", "inverse_vol", "min_var", "risk_parity"], key="port_method")
            pe = PortfolioEngine(method=method, max_weight=0.6, min_weight=0.05, ann=12)
            ev = pe.evaluate(assets)
            st.dataframe(ev["weights"].to_frame().style.format("{:.1%}"), use_container_width=True)
            c = st.columns(3)
            c[0].metric("VOL", f"{ev['vol']:.1%}")
            c[1].metric("SHARPE", f"{ev['sharpe']:.2f}")
            c[2].metric("MAX DD", f"{ev['max_dd']:.1%}")
            st.line_chart(ev["equity"], height=260)
        except Exception as exc:
            st.error(f"DADOS INDISPONÍVEIS / AMOSTRA INSUFICIENTE — {exc}")

    st.caption(data_status("BCB/SGS + FRED", "canonical factors + risk assets", factors.index.max()))
