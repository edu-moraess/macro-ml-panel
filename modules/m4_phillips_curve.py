"""
Módulo 4 — Curva de Phillips não-linear (Gradient Boosting)
Ideia: a curva de Phillips "aumentada por expectativas" raramente é linear
na prática. Um Gradient Boosting captura a não-linearidade sem impor forma
funcional, e a importância por permutação substitui o coeficiente da
regressão como medida de relevância de cada variável.
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from data_utils import get_fred, align, FRED


def render():
    st.header("4 · Curva de Phillips não-linear (Gradient Boosting)")
    st.caption(
        "Desemprego (nível e defasagem) e inflação passada prevendo inflação atual (EUA), "
        "deixando o modelo achar a não-linearidade e ranqueando as variáveis por "
        "importância de permutação (em vez de um coeficiente linear fixo)."
    )

    unrate = get_fred(FRED["unrate"]).resample("MS").mean()
    cpi = get_fred(FRED["cpi_yoy"]).resample("MS").mean()
    inflacao = cpi.pct_change(12) * 100  # YoY %

    df = align(unrate.rename("desemprego"), inflacao.rename("inflacao"))
    df["desemprego_lag12"] = df["desemprego"].shift(12)
    df["inflacao_lag12"] = df["inflacao"].shift(12)
    df = df.dropna()

    feats = ["desemprego", "desemprego_lag12", "inflacao_lag12"]
    X, y = df[feats], df["inflacao"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = GradientBoostingRegressor(random_state=0).fit(X_train, y_train)
    r2 = model.score(X_test, y_test)

    imp = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=0)
    imp_df = pd.Series(imp.importances_mean, index=feats, name="importância").sort_values(
        ascending=False
    )

    st.metric("R² fora da amostra", f"{r2:.2f}")

    pred = pd.Series(model.predict(X), index=df.index, name="previsto")
    st.line_chart(pd.concat([df["inflacao"].rename("observado"), pred], axis=1))

    st.write("**Importância por permutação (quanto o R² cai ao embaralhar a variável):**")
    st.bar_chart(imp_df)
