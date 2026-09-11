"""Módulo 7 — Tendência-ciclo via Autoencoder com comparação ao filtro HP."""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.filters.hp_filter import hpfilter
from data_utils import get_bcb, SGS

def _windows(x,w):
    return x[np.arange(w)[None,:]+np.arange(len(x)-w+1)[:,None]]
def _bottleneck(model,X):
    a=np.tanh(X@model.coefs_[0]+model.intercepts_[0]); a=np.tanh(a@model.coefs_[1]+model.intercepts_[1]); return a.ravel()

def render():
    st.header("7 · Tendência-ciclo via Autoencoder")
    st.caption("Decomposição não-linear exploratória do IPCA, comparada ao HP e à média móvel; inclui erro de reconstrução e estabilidade por janela.")
    w=st.slider("Tamanho da janela (meses)",7,25,13,step=2)
    s=get_bcb(SGS["ipca_mensal"]).resample("MS").mean().dropna(); values=s.to_numpy().reshape(-1,1); scaler=StandardScaler().fit(values); x=scaler.transform(values).ravel(); X=_windows(x,w)
    model=MLPRegressor(hidden_layer_sizes=(8,1,8),activation="tanh",max_iter=3000,random_state=0).fit(X,X); recon=model.predict(X); mse=float(np.mean((X-recon)**2)); bottleneck=_bottleneck(model,X)
    centro=x[w//2:w//2+len(bottleneck)]; a,b=np.polyfit(bottleneck,centro,1); trend_ae=scaler.inverse_transform((a*bottleneck*0+a*bottleneck+b).reshape(-1,1)).ravel()
    idx=s.index[w//2:w//2+len(bottleneck)]; trend_ma=s.rolling(w,center=True).mean().reindex(idx)
    _,cycle_hp=hpfilter(s,lamb=129600); hp_trend=cycle_hp.reindex(idx) if hasattr(cycle_hp,"reindex") else pd.Series(cycle_hp,index=s.index).reindex(idx)
    df=pd.DataFrame({"observado":s.reindex(idx).values,"tendência (AE)":trend_ae,"tendência (média móvel)":trend_ma.values,"tendência (HP)":hp_trend.values},index=idx)
    c=st.columns(2); c[0].metric("Erro MSE de reconstrução",f"{mse:.4f}"); c[1].metric("Observações",f"{len(s)}")
    st.line_chart(df); ciclo=df["observado"]-df["tendência (AE)"]; st.write("**Componente cíclico do autoencoder:**"); st.line_chart(ciclo.rename("ciclo"))
    st.write("**Diferença AE − HP:**"); st.line_chart((df["tendência (AE)"]-df["tendência (HP)"]).rename("diferença"))
    st.caption("O AE é estimado sobre a amostra disponível e, portanto, esta tela é análise descritiva; não trate a tendência como estimativa OOS nem como ausência garantida de viés de fim de amostra.")
