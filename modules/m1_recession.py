"""Recession probability model — expanding-window Elastic Net Logit."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from data_utils import get_fred, align, data_status


FEATURES = [
    "spread_10y3m",
    "spread_10y2y",
    "unemployment",
    "fed_funds",
    "vix",
    "hy_spread",
    "sp500_return3m",
]
FEATURE_LABELS = {
    "spread_10y3m": "10Y–3M spread",
    "spread_10y2y": "10Y–2Y spread",
    "unemployment": "Unemployment",
    "fed_funds": "Fed Funds",
    "vix": "VIX",
    "hy_spread": "HY OAS",
    "sp500_return3m": "S&P 500 return 3M",
}


def _future_recession(usrec: pd.Series, horizon: int) -> pd.Series:
    """1 when any recession month occurs in t+1...t+horizon."""
    out = pd.Series(np.nan, index=usrec.index, dtype=float)
    values = pd.to_numeric(usrec, errors="coerce").to_numpy()
    for i in range(len(values) - horizon):
        window = values[i + 1 : i + 1 + horizon]
        if len(window) == horizon and np.isfinite(window).all():
            out.iloc[i] = float(np.any(window >= 1.0))
    return out


def _fit_elastic_net(train: pd.DataFrame, features: list[str], l1_ratio: float):
    scaler = StandardScaler().fit(train[features])
    model = LogisticRegression(
        penalty="elasticnet",
        l1_ratio=l1_ratio,
        solver="saga",
        C=1.0,
        max_iter=5000,
        random_state=42,
    ).fit(scaler.transform(train[features]), train["target"].astype(int))
    return model, scaler


def _walk_forward(labeled: pd.DataFrame, features: list[str], l1_ratio: float, min_train: int):
    oos = pd.DataFrame(index=labeled.index)
    oos["target"] = labeled["target"]
    oos["elastic_net"] = np.nan
    oos["baseline_prevalence"] = np.nan
    oos["logit_spread"] = np.nan

    for i in range(min_train, len(labeled)):
        train = labeled.iloc[:i]
        test = labeled.iloc[[i]]
        if train["target"].nunique() < 2:
            continue

        model, scaler = _fit_elastic_net(train, features, l1_ratio)
        oos.iloc[i, oos.columns.get_loc("elastic_net")] = model.predict_proba(
            scaler.transform(test[features])
        )[0, 1]
        oos.iloc[i, oos.columns.get_loc("baseline_prevalence")] = float(train["target"].mean())

        spread_model, spread_scaler = _fit_elastic_net(train, ["spread_10y3m"], l1_ratio)
        oos.iloc[i, oos.columns.get_loc("logit_spread")] = spread_model.predict_proba(
            spread_scaler.transform(test[["spread_10y3m"]])
        )[0, 1]

    return oos.dropna(subset=["elastic_net"])


def _metrics(y: pd.Series, p: pd.Series) -> dict[str, float]:
    result = {"auc": np.nan, "brier": np.nan, "log_loss": np.nan}
    if len(y) >= 10 and y.nunique() == 2:
        result["auc"] = float(roc_auc_score(y, p))
        result["brier"] = float(brier_score_loss(y, p))
        result["log_loss"] = float(log_loss(y, p, labels=[0, 1]))
    return result


def _calibration_frame(oos: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    y = oos["target"].astype(int)
    p = oos["elastic_net"]
    if len(oos) < 20 or y.nunique() != 2:
        return pd.DataFrame()
    frac, mean_pred = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    return pd.DataFrame({"Predito": mean_pred, "Observado": frac})


def _alert_diagnostics(oos: pd.DataFrame, threshold: float) -> dict[str, float]:
    p = oos["elastic_net"]
    signal = (p >= threshold) & (p.shift(1).fillna(0) < threshold)
    hits = int(((signal) & (oos["target"] >= 1)).sum())
    alerts = int(signal.sum())
    false_positives = alerts - hits
    return {
        "alerts": alerts,
        "hits": hits,
        "false_positives": false_positives,
        "precision": hits / alerts if alerts else np.nan,
    }


def _lead_time_table(oos: pd.DataFrame, usrec: pd.Series, threshold: float) -> pd.DataFrame:
    """Find first OOS threshold crossing before each NBER recession start."""
    starts = usrec[(usrec >= 1) & (usrec.shift(1).fillna(0) < 1)].index
    rows: list[dict[str, object]] = []
    for start in starts:
        history = oos.loc[oos.index < start].tail(36).copy()
        crossing = (history["elastic_net"] >= threshold) & (
            history["elastic_net"].shift(1).fillna(0) < threshold
        )
        hit = history.loc[crossing]
        if hit.empty:
            rows.append({"Recessão iniciada": start, "Sinal >= limiar": pd.NaT, "Lead time (meses)": np.nan})
            continue
        signal_date = hit.index[0]
        lead = (start.year - signal_date.year) * 12 + (start.month - signal_date.month)
        rows.append({"Recessão iniciada": start, "Sinal >= limiar": signal_date, "Lead time (meses)": lead})
    return pd.DataFrame(rows)


def render() -> None:
    st.header("1 · Probabilidade condicional de recessão — Elastic Net Logit")
    st.caption(
        "Probabilidade de qualquer mês de recessão nos próximos h meses, estimada por janela expansiva. "
        "Cada origem temporal ajusta scaler e modelo somente com observações disponíveis naquela origem."
    )

    horizon = st.slider("Horizonte de previsão (meses)", 6, 24, 12)
    l1_ratio = st.slider("L1 ratio", 0.0, 1.0, 0.5)
    threshold = st.slider("Limiar para alerta histórico", 0.1, 0.9, 0.5, 0.05)

    series = {
        "spread_10y3m": get_fred("T10Y3M").resample("MS").mean(),
        "spread_10y2y": (get_fred("DGS10") - get_fred("DGS2")).resample("MS").mean(),
        "unemployment": get_fred("UNRATE").resample("MS").mean(),
        "fed_funds": get_fred("FEDFUNDS").resample("MS").mean(),
        "vix": get_fred("VIXCLS").resample("MS").mean(),
        "hy_spread": get_fred("BAMLH0A0HYM2").resample("MS").mean(),
        "sp500_return3m": get_fred("SP500").resample("MS").last().pct_change(3),
        "usrec": get_fred("USREC").resample("MS").mean().round(),
    }
    df = align(*[series[k].rename(k) for k in series])
    df["target"] = _future_recession(df["usrec"], horizon)

    feature_df = df.dropna(subset=FEATURES).copy()
    labeled = feature_df.dropna(subset=["target"]).copy()
    if len(labeled) < 72 or labeled["target"].nunique() < 2:
        raise ValueError("AMOSTRA INSUFICIENTE para estimação da probabilidade de recessão")

    min_train = max(60, int(len(labeled) * 0.4))
    oos = _walk_forward(labeled, FEATURES, l1_ratio, min_train)
    metrics = _metrics(oos["target"], oos["elastic_net"])
    benchmark_metrics = _metrics(oos["target"], oos["baseline_prevalence"])
    spread_metrics = _metrics(oos["target"], oos["logit_spread"])

    current = feature_df.iloc[[-1]]
    model, scaler = _fit_elastic_net(labeled, FEATURES, l1_ratio)
    current_prob = float(model.predict_proba(scaler.transform(current[FEATURES]))[0, 1])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"P(recessão em {horizon}M)", f"{current_prob:.1%}")
    c2.metric("AUC OOS", f"{metrics['auc']:.3f}" if np.isfinite(metrics["auc"]) else "—")
    c3.metric("Brier OOS", f"{metrics['brier']:.3f}" if np.isfinite(metrics["brier"]) else "—")
    c4.metric("Log Loss OOS", f"{metrics['log_loss']:.3f}" if np.isfinite(metrics["log_loss"]) else "—")

    st.subheader("Validação OOS e benchmarks")
    comparison = pd.DataFrame(
        {
            "AUC OOS": [metrics["auc"], spread_metrics["auc"], benchmark_metrics["auc"]],
            "Brier OOS": [metrics["brier"], spread_metrics["brier"], benchmark_metrics["brier"]],
            "Log Loss OOS": [metrics["log_loss"], spread_metrics["log_loss"], benchmark_metrics["log_loss"]],
        },
        index=["Elastic Net multivariado", "Logit somente 10Y–3M", "Prevalência histórica expansiva"],
    )
    st.dataframe(comparison.style.format("{:.3f}"), use_container_width=True)
    st.caption(f"Walk-forward expanding window · treino mínimo: {min_train} observações · OOS: {len(oos):,}")

    st.subheader(f"Probabilidade histórica OOS — horizonte {horizon}M")
    if len(oos):
        chart = oos[["elastic_net", "target"]].rename(
            columns={"elastic_net": f"P(recessão {horizon}M)", "target": "Evento futuro"}
        )
        st.line_chart(chart)

    st.subheader("Calibração")
    calibration = _calibration_frame(oos)
    if not calibration.empty:
        st.line_chart(calibration.set_index("Predito")["Observado"])
        st.dataframe(calibration, use_container_width=True)
        st.caption("Quanto mais próxima a curva estiver da diagonal, melhor a calibração probabilística.")
    else:
        st.info("Dados OOS insuficientes para uma curva de calibração estável.")

    st.subheader(f"Alertas históricos — limiar {threshold:.0%}")
    alert = _alert_diagnostics(oos, threshold)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alertas", f"{alert['alerts']}")
    c2.metric("Alertas seguidos por evento", f"{alert['hits']}")
    c3.metric("Falsos positivos", f"{alert['false_positives']}")
    c4.metric("Precisão do alerta", f"{alert['precision']:.1%}" if np.isfinite(alert["precision"]) else "—")

    st.subheader(f"Lead time histórico — limiar {threshold:.0%}")
    lead = _lead_time_table(oos, df["usrec"], threshold)
    if not lead.empty:
        valid_lead = lead["Lead time (meses)"].dropna()
        c1, c2, c3 = st.columns(3)
        c1.metric("Recessões avaliadas", f"{len(lead)}")
        c2.metric("Detectadas", f"{len(valid_lead)}")
        c3.metric("Lead time médio", f"{valid_lead.mean():.1f}M" if len(valid_lead) else "—")
        st.dataframe(lead, use_container_width=True)
    else:
        st.info("Não há recessões históricas disponíveis na janela OOS para calcular lead time.")

    st.subheader("Contribuição dos fatores")
    coef = pd.Series(model.coef_[0], index=FEATURES, name="coeficiente padronizado")
    coef.index = [FEATURE_LABELS[x] for x in coef.index]
    coef = coef.sort_values(ascending=False)
    st.dataframe(coef.to_frame(), use_container_width=True)
    st.caption("Coeficientes são calculados sobre features padronizadas; sinal positivo aumenta a contribuição para a classe recessão.")

    st.subheader("Cobertura das séries")
    coverage = pd.DataFrame(
        {
            "Última observação": [series[k].dropna().index.max() for k in FEATURES],
            "N observações": [int(series[k].notna().sum()) for k in FEATURES],
        },
        index=[FEATURE_LABELS[k] for k in FEATURES],
    )
    st.dataframe(coverage, use_container_width=True)
    st.info(
        "O modelo combina curva de juros, mercado, desemprego, condições financeiras e volatilidade. "
        "A probabilidade é condicional às séries selecionadas, não uma previsão macroeconômica completa. "
        "As séries FRED podem sofrer revisões históricas; esta validação não usa vintages em tempo real."
    )
    st.caption(data_status("Federal Reserve Bank of St. Louis · FRED", ", ".join(FEATURES) + " / USREC", feature_df.index.max()))
