"""Módulo 5 — Nowcasting de atividade econômica com validação temporal."""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_utils import get_bcb, align, SGS


def render():
    st.header("5 · Nowcasting de atividade (Gradient Boosting)")
    st.caption("Estimativa do IBC-Br usando apenas informação defasada no painel. A versão abaixo é um backtest pseudo-real-time; não usa vintages históricas de publicação.")
    ibc=get_bcb(SGS["ibc_br"]).resample("MS").mean(); cambio=get_bcb(SGS["cambio_ptax"]).resample("MS").mean(); selic=get_bcb(SGS["selic_meta"]).resample("MS").mean(); ipca=get_bcb(SGS["ipca_mensal"]).resample("MS").mean(); desemp=get_bcb(SGS["desemprego_pnad"]).resample("MS").mean()
    df=align(ibc.rename("ibc"),cambio.rename("cambio"),selic.rename("selic"),ipca.rename("ipca"),desemp.rename("desemprego")).dropna()
    for c in ["cambio","selic","ipca","desemprego"]: df[f"{c}_lag1"]=df[c].shift(1)
    df=df.dropna(); feats=["cambio_lag1","selic_lag1","ipca_lag1","desemprego_lag1"]
    min_train=max(48,int(len(df)*0.5)); pred=[]; actual=[]; naive=[]
    for i in range(min_train,len(df)):
        tr=df.iloc[:i]; te=df.iloc[[i]]
        model=GradientBoostingRegressor(random_state=0).fit(tr[feats],tr["ibc"])
        pred.append(float(model.predict(te[feats])[0])); actual.append(float(te["ibc"].iloc[0])); naive.append(float(df["ibc"].iloc[i-1]))
    y=pd.Series(actual); p=pd.Series(pred); n=pd.Series(naive)
    mae=mean_absolute_error(y,p); rmse=mean_squared_error(y,p)**0.5; mae_n=mean_absolute_error(y,n)
    c=st.columns(3); c[0].metric("OOS MAE",f"{mae:.3f}"); c[1].metric("OOS RMSE",f"{rmse:.3f}"); c[2].metric("MAE benchmark t−1",f"{mae_n:.3f}")
    pred_s=pd.Series(pred,index=df.index[min_train:],name="nowcast OOS")
    st.line_chart(pd.concat([df["ibc"].rename("observado"),pred_s],axis=1))
    final=GradientBoostingRegressor(random_state=0).fit(df.iloc[:min_train][feats],df.iloc[:min_train]["ibc"])
    st.write("**Importância das variáveis (treino inicial):**")
    st.bar_chart(pd.Series(final.feature_importances_,index=feats).sort_values(ascending=False))
    st.caption(f"Walk-forward expanding-window · treino inicial={min_train} · OOS={len(y)}. Limitação importante: sem dados vintage/release-date, a validação não reconstrói perfeitamente o conjunto de informação disponível em cada data histórica.")
