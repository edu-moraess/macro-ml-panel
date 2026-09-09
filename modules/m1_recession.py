"""Recession probability model — expanding-window Elastic Net Logit."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

from data_utils import get_fred, align, data_status


def _future_recession(usrec: pd.Series, horizon: int) -> pd.Series:
    """1 when any recession month occurs in t+1...t+horizon."""
    out = pd.Series(np.nan, index=usrec.index, dtype=float)
    values = pd.to_numeric(usrec, errors="coerce").to_numpy()
    for i in range(len(values) - horizon):
        window = values[i + 1 : i + 1 + horizon]
        if len(window) == horizon and np.isfinite(window).all():
            out.iloc[i] = float(np.any(window >= 1.0))
    return out


def _fit_predict(train: pd.DataFrame, current: pd.DataFrame, l1_ratio: float) -> tuple[LogisticRegression, np.ndarray, np.ndarray]:
    feats = ["spread", "spread_chg3m", "spread_min6m"]
    scaler = StandardScaler().fit(train[feats])
    model = LogisticRegression(
        penalty="elasticnet", l1_ratio=l1_ratio, solver="saga", C=1.0, max_iter=5000, random_state=42
    ).fit(scaler.transform(train[feats]), train["target"].astype(int))
    return model, scaler.transform(current[feats]), scaler.transform(train[feats])


def render() -> None:
    st.header("1 · Probabilidade de recessão — Elastic Net Logit")
    st.caption(
        "P(recessão em qualquer mês do horizonte futuro), estimada por janela expansiva. "
        "O scaler e o modelo são ajustados somente com informação disponível em cada origem temporal."
    )

    horizon = st.slider("Horizonte de previsão (meses)", 6, 24, 12)
    l1_ratio = st.slider("L1 ratio", 0.0, 1.0, 0.5)

    spread = get_fred("T10Y3M").resample("MS").mean()
    usrec = get_fred("USREC").resample("MS").mean().round()
    df = align(spread.rename("spread"), usrec.rename("usrec"))
    df["spread_chg3m"] = df["spread"].diff(3)
    df["spread_min6m"] = df["spread"].rolling(6).min()
    df["target"] = _future_recession(df["usrec"], horizon)

    feats = ["spread", "spread_chg3m", "spread_min6m"]
    feature_df = df.dropna(subset=feats).copy()
    labeled = feature_df.dropna(subset=["target"]).copy()
    if len(labeled) < 72 or labeled["target"].nunique() < 2:
        raise ValueError("AMOSTRA INSUFICIENTE para estimação da probabilidade de recessão")

    min_train = max(60, int(len(labeled) * 0.4))
    oos_prob = pd.Series(np.nan, index=labeled.index, dtype=float)
    for i in range(min_train, len(labeled)):
        train = labeled.iloc[:i]
        test = labeled.iloc[[i]]
        if train["target"].nunique() < 2:
            continue
        scaler = StandardScaler().fit(train[feats])
        model = LogisticRegression(
            penalty="elasticnet", l1_ratio=l1_ratio, solver="saga", C=1.0, max_iter=5000, random_state=42
        ).fit(scaler.transform(train[feats]), train["target"].astype(int))
        oos_prob.iloc[i] = model.predict_proba(scaler.transform(test[feats]))[0, 1]

    oos = labeled.assign(prob_oos=oos_prob).dropna(subset=["prob_oos"])
    if len(oos) >= 10 and oos["target"].nunique() == 2:
        auc = roc_auc_score(oos["target"], oos["prob_oos"])
        brier = brier_score_loss(oos["target"], oos["prob_oos"])
    else:
        auc, brier = np.nan, np.nan

    current = feature_df.iloc[[-1]]
    train_final = labeled
    if train_final["target"].nunique() < 2:
        raise ValueError("AMOSTRA INSUFICIENTE para modelo final")
    scaler = StandardScaler().fit(train_final[feats])
    model = LogisticRegression(
        penalty="elasticnet", l1_ratio=l1_ratio, solver="saga", C=1.0, max_iter=5000, random_state=42
    ).fit(scaler.transform(train_final[feats]), train_final["target"].astype(int))
    current_prob = float(model.predict_proba(scaler.transform(current[feats]))[0, 1])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"P(recessão em {horizon}M)", f"{current_prob:.1%}")
    c2.metric("AUC OOS", f"{auc:.3f}" if np.isfinite(auc) else "—")
    c3.metric("Brier OOS", f"{brier:.3f}" if np.isfinite(brier) else "—")
    c4.metric("Amostra OOS", f"{len(oos):,}")

    if len(oos):
        chart = oos[["prob_oos", "target"]].rename(columns={"prob_oos": "P(recessão)", "target": "Evento futuro"})
        st.line_chart(chart)

    coef = pd.Series(model.coef_[0], index=feats, name="coeficiente")
    st.write("**Coeficientes do modelo final:**")
    st.dataframe(coef.to_frame(), use_container_width=True)
    st.caption(data_status("Federal Reserve Bank of St. Louis · FRED", "T10Y3M / USREC", feature_df.index.max()))
    st.info("Limitação: o modelo usa apenas a curva 10Y–3M e derivados. A probabilidade é condicional a esse conjunto de variáveis; não é uma previsão macroeconômica completa.")
