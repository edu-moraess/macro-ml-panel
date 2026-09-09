"""Institutional Quant Lab."""
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
from data_utils import FRED, SGS, data_status, get_bcb, get_fred

def z(s, n):
    r=s.rolling(n,min_periods=max(24,n//2)); return (s-r.mean())/r.std().replace(0,np.nan)

def render():
    st.header('Quant Lab')
    st.caption('Factor Engine · Signal Engine · Risk Engine · Time-Series Lab · Backtest Lab · Portfolio Engine')
    n=st.slider('Rolling window',36,120,60,12)
    x=pd.concat({'IBC':get_bcb(SGS['ibc_br']).resample('MS').mean(),'U':get_bcb(SGS['desemprego_pnad']).resample('MS').mean(),'IPCA':get_bcb(SGS['ipca_mensal']).resample('MS').mean(),'Selic':get_bcb(SGS['selic_meta']).resample('MS').mean(),'FX':get_bcb(SGS['cambio_ptax']).resample('MS').mean(),'Spread':get_fred(FRED['t10y3m']).resample('MS').mean(),'VIX':get_fred(FRED['vix']).resample('MS').mean(),'HY':get_fred(FRED['hy_spread']).resample('MS').mean(),'SP500':get_fred(FRED['sp500']).resample('MS').last()},axis=1).dropna()
    f=pd.DataFrame(index=x.index)
    f['Growth']=z(x.IBC.pct_change(3)*100,n)-z(x.U.diff(3),n)
    f['Inflation']=z(x.IPCA.diff(3),n); f['Rates']=z(x.Selic.diff(3),n); f['FX']=-z(x.FX.pct_change(3)*100,n); f['Liquidity']=-z(x.HY,n)-z(x.VIX,n); f['Momentum']=z(x.SP500.pct_change(12)*100,n)
    f=f.replace([np.inf,-np.inf],np.nan).dropna()
    if len(f)<n: raise ValueError('Amostra insuficiente para o rolling window escolhido.')
    tabs=st.tabs(['FACTORS','SIGNALS','RISK','TIME SERIES','BACKTEST','PORTFOLIO']); latest=f.iloc[-1]
    with tabs[0]:
        st.subheader('Factor State'); st.dataframe(latest.sort_values(ascending=False).to_frame('z-score').style.format('{:+.2f}'),use_container_width=True); st.line_chart(f,height=360)
    with tabs[1]:
        score=float((latest.Growth-latest.Inflation-latest.Rates+latest.FX+latest.Liquidity+latest.Momentum)/6); signal='LONG / RISK-ON' if score>.25 else 'SHORT / RISK-OFF' if score<-.25 else 'NEUTRAL'; c=st.columns(3); c[0].metric('COMPOSITE',f'{score:+.2f}σ'); c[1].metric('SIGNAL',signal); c[2].metric('CONFIDENCE',f'{min(99,50+abs(score)*25):.1f}%'); st.line_chart(f.mean(axis=1).rename('Composite factor score'),height=300)
    with tabs[2]:
        r=x.SP500.pct_change().dropna(); vol=r.rolling(20).std()*np.sqrt(12); wealth=(1+r).cumprod(); dd=wealth/wealth.cummax()-1; q=r.quantile(.05); es=r[r<=q].mean(); c=st.columns(5); c[0].metric('VOL 20M',f'{vol.iloc[-1]:.1%}'); c[1].metric('VaR 95%',f'{-q:.2%}'); c[2].metric('ES 95%',f'{-es:.2%}'); c[3].metric('MAX DD',f'{dd.min():.2%}'); c[4].metric('OBS',f'{len(r):,}'); st.line_chart(pd.DataFrame({'Volatility':vol,'Drawdown':dd}),height=320)
    with tabs[3]:
        lam=st.slider('EWMA λ',.80,.99,.94,.01); r=x.SP500.pct_change().dropna(); ewma=r.pow(2).ewm(alpha=1-lam,adjust=False).mean().pow(.5)*np.sqrt(12); st.line_chart(pd.DataFrame({'EWMA vol':ewma}),height=300); st.caption('EWMA atua como filtro de estado sobre observações reais.')
    with tabs[4]:
        sig=f.mean(axis=1).shift(1); asset=x.SP500.pct_change().reindex(sig.index); strat=asset.where(sig>.15,-asset.where(sig<-.15,0.0)).fillna(0); eq=(1+strat).cumprod(); bh=(1+asset.fillna(0)).cumprod(); sharpe=strat.mean()/strat.std()*np.sqrt(12) if strat.std() else np.nan; c=st.columns(3); c[0].metric('SHARPE',f'{sharpe:.2f}'); c[1].metric('MAX DD',f'{(eq/eq.cummax()-1).min():.1%}'); c[2].metric('TRADES',f'{int((sig.abs()>.15).sum()):,}'); st.line_chart(pd.DataFrame({'Strategy':eq,'Buy & Hold':bh}),height=320); st.caption('Sinal defasado em um período para evitar look-ahead.')
    with tabs[5]:
        a=pd.concat({'S&P500':get_fred(FRED['sp500']),'WTI':get_fred(FRED['wti']),'Gold':get_fred(FRED['gold'])},axis=1).resample('MS').last().dropna().pct_change().dropna(); inv=1/np.sqrt(np.diag(a.cov())); w=inv/inv.sum(); p=a@w; c=st.columns(4); [c[i].metric(k,f'{w[i]:.1%}') for i,k in enumerate(a.columns)]; st.metric('PORTFOLIO VOL',f'{p.std()*np.sqrt(12):.1%}'); st.dataframe(pd.DataFrame({'weight':w},index=a.columns).style.format('{:.1%}'),use_container_width=True); st.line_chart((1+p).cumprod().rename('Portfolio wealth'),height=280)
    st.caption(data_status('BCB/SGS + FRED','macro factors + SP500/VIX/HY/WTI/GOLD',f.index.max()))
