"""
Módulo 1 — Recessão via Elastic Net Logit
Ideia: o spread de juros (10y-3m) é o preditor clássico de recessão
(Estrella & Mishkin). Em vez de um logit simples com o spread cru,
deixamos o Elastic Net escolher, entre spread + variações do spread,
quais features realmente carregam sinal — regularização fazendo
seleção de variável no lugar do pesquisador.
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data_utils import get_fred, align


def render():
    st.header("1 · Probabilidade de recessão (Elastic Net Logit)")
    st.caption(
        "Spread 10y–3m (T10Y3M) prevendo recessão do NBER (USREC) 12 meses à frente. "
        "Elastic Net faz a seleção de variável entre spread e suas variações."
    )

    horizon = st.slider("Horizonte de previsão (meses)", 6, 24, 12)
    l1_ratio = st.slider("L1 ratio (0 = Ridge, 1 = Lasso puro)", 0.0, 1.0, 0.5)

    spread = get_fred("T10Y3M").resample("MS").mean()
    usrec = get_fred("USREC").resample("MS").mean().round()
    df = align(spread.rename("spread"), usrec.rename("usrec"))

    df["spread_chg3m"] = df["spread"].diff(3)
    df["spread_min6m"] = df["spread"].rolling(6).min()
    df["target"] = df["usrec"].shift(-horizon)
    df = df.dropna()

    feats = ["spread", "spread_chg3m", "spread_min6m"]
    X = StandardScaler().fit_transform(df[feats])
    y = df["target"].astype(int)

    model = LogisticRegression(
        penalty="elasticnet", l1_ratio=l1_ratio, solver="saga", C=1.0, max_iter=5000
    ).fit(X, y)
    df["prob"] = model.predict_proba(X)[:, 1]

    st.line_chart(df[["prob", "usrec"]].rename(
        columns={"prob": "P(recessão)", "usrec": "Recessão (NBER)"}
    ))

    coef = pd.Series(model.coef_[0], index=feats, name="coeficiente")
    st.write("**Coeficientes (0 = variável descartada pelo Elastic Net):**")
    st.dataframe(coef.to_frame())

    st.metric(
        f"P(recessão em {horizon} meses) — leitura mais recente",
        f"{df['prob'].iloc[-1]:.1%}",
    )
