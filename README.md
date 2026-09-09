# Macro Quant Research Terminal (V2)

Terminal quantitativo institucional de pesquisa macroeconômica em Streamlit.

**Arquitetura**

```
PUBLIC DATA (BCB/SGS + FRED)
        ↓
   DATA ENGINE          (cache, quality, alignment)
        ↓
 FEATURE ENGINE         (YoY, MoM, 3m, z-rolling, accel)
        ↓
  FACTOR ENGINE         (Growth, Inflation, Rates, FX, Liquidity, Momentum)
        ↓
  REGIME ENGINE         (HMM / GMM · probability · duration)
        ↓
  SIGNAL ENGINE         (MACRO SCORE · confidence · contributions)
        ↓
   RISK ENGINE          (VaR/ES, EWMA, DD, Sharpe/Sortino/Calmar)
        ↓
BACKTEST / PORTFOLIO    (lag≥1 · constraints · real returns)
        ↓
   QUANT TERMINAL
```

## Princípios

- Dados públicos reais apenas. Sem sintético, mock ou inventado.
- Causalidade temporal: `shift(1)` obrigatório no backtest.
- Rolling z-score preferido a global.
- Mensagens controladas: `DADOS INDISPONÍVEIS` · `AMOSTRA INSUFICIENTE` · `MODELO NÃO ESTIMÁVEL` · `FALHA CONTROLADA`.
- Cache 15 min (`st.cache_data`).

## Navegação

1. QUANT STATE — macro dashboard (regime, score, factors)
2. QUANT LAB — factors / signals / risk / TS / backtest / portfolio
3. DATA QUALITY — provenance e freshness de todas as séries
4. Legacy research modules (educacionais)

## Instalação

```bash
pip install -r requirements.txt
# .streamlit/secrets.toml
# FRED_API_KEY = "sua_chave"
streamlit run app.py
```

## Testes

```bash
PYTHONPATH=. pytest tests/ -q
python -m compileall -q .
```

## Limitações

- FRED requer chave; sem ela falha de forma controlada.
- HMM requer `hmmlearn` (fallback GMM).
- Time-series lab: AR(1) diagnóstico; ARIMA/GARCH/VAR completos exigem amostra e não estão forçados.
- Black-Litterman e stress histórico completo não implementados (dados/expectativas).
