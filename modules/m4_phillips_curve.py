"""Módulo 4 — Curva de Phillips não-linear com validação walk-forward."""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_utils import get_fred, align, FRED


def render():
    st.header("4 · Curva de Phillips não-linear (Gradient Boosting)")
    st.caption("Desemprego e inflação passada prevendo inflação atual, com validação temporal expanding-window e benchmark ingênuo.")
    unrate = get_fred(FRED["unrate"]).resample("MS").mean()
    cpi = get_fred(FRED["cpi_yoy"]).resample("MS").mean()
    inflacao = cpi.pct_change(12) * 100
    df = align(unrate.rename("desemprego"), inflacao.rename("inflacao"))
    df["desemprego_lag12"] = df["desemprego"].shift(12); df["inflacao_lag12"] = df["inflacao"].shift(12); df = df.dropna()
    feats = ["desemprego", "desemprego_lag12", "inflacao_lag12"]
    min_train = max(60, int(len(df)*0.5)); preds=[]; actual=[]
    for i in range(min_train, len(df)):
        tr=df.iloc[:i]; te=df.iloc[[i]]
        model=GradientBoostingRegressor(random_state=0).fit(tr[feats], tr["inflacao"])
        preds.append(float(model.predict(te[feats])[0])); actual.append(float(te["inflacao"].iloc[0]))
    y=pd.Series(actual); p=pd.Series(preds)
    naive=df["inflacao"].shift(1).iloc[min_train:].reset_index(drop=True)
    mae=mean_absolute_error(y,p); rmse=mean_squared_error(y,p)**0.5; r2=r2_score(y,p)
    mae_naive=mean_absolute_error(y.iloc[1:], naive.iloc[1:]) if len(y)>1 else np.nan
    c=st.columns(4); c[0].metric("OOS MAE",f"{mae:.3f}"); c[1].metric("OOS RMSE",f"{rmse:.3f}"); c[2].metric("OOS R²",f"{r2:.2f}"); c[3].metric("MAE benchmark",f"{mae_naive:.3f}")
    pred_series=pd.Series(preds,index=df.index[min_train:],name="previsto OOS")
    st.line_chart(pd.concat([df["inflacao"].rename("observado"),pred_series],axis=1))
    final=GradientBoostingRegressor(random_state=0).fit(df.iloc[:min_train][feats],df.iloc[:min_train]["inflacao"])
    imp=permutation_importance(final,df.iloc[min_train:][feats],df.iloc[min_train:]["inflacao"],n_repeats=20,random_state=0) if len(df)>min_train+5 else None
    if imp is not None:
        st.write("**Importância por permutação no bloco OOS:**")
        st.bar_chart(pd.Series(imp.importances_mean,index=feats).sort_values(ascending=False))
    st.caption(f"Walk-forward expanding-window · treino inicial={min_train} observações · OOS={len(y)}. O benchmark é persistência t−1.")
