# Macro Quant Research Terminal

Terminal quantitativo de pesquisa macro em Streamlit, construído exclusivamente sobre **dados públicos reais** (BCB/SGS + FRED).

Arquitetura orientada a:

**dados reais → engenharia quantitativa → fatores → regime → sinais → risco → visualização institucional**

Não utiliza dados sintéticos, mock ou fallback artificial. Falhas de API são expostas de forma controlada.

## Execução

```bash
pip install -r requirements.txt
# Configure FRED_API_KEY em .streamlit/secrets.toml
streamlit run app.py
```

## Fontes de dados

| Fonte | Séries | Autenticação |
|-------|--------|--------------|
| BCB/SGS | Selic, IPCA, IBC-Br, PTAX, Desemprego PNAD | Nenhuma |
| FRED | Term spread, USREC, VIX, HY OAS, S&P 500, WTI, Gold, etc. | `FRED_API_KEY` via Streamlit Secrets |

## Módulos principais

| Tab | Função |
|-----|--------|
| **Quant State** | Estado macro agregado (fatores, regime RISK-ON/OFF/NEUTRAL, score, data quality) |
| **Quant Lab** | Factor / Signal / Risk / Time-Series / Backtest / Portfolio engines |
| Recessão … Yield Curve | Módulos educacionais de macroeconometria + ML (preservados) |

## Princípios

- **Dados reais apenas** — nenhuma série inventada.
- **Look-ahead bias controlado** — sinais defasados no backtest.
- **Rolling z-score** preferido a z-score global para séries não-estacionárias.
- **Cache 15 min** (`st.cache_data`) — evita hammering de APIs.
- **Mensagens controladas**: `DADOS INDISPONÍVEIS`, `AMOSTRA INSUFICIENTE`, `FALHA CONTROLADA`.
- **UI institucional**: Light — Research / Dark — Terminal.

## Testes

```bash
PYTHONPATH=. pytest tests/ -q
python -m compileall -q .
```

## Limitações conhecidas

- FRED exige chave configurada; sem ela os módulos dependentes de FRED falham de forma controlada.
- Modelos avançados (ARIMA completo, GARCH-X, VECM, Black-Litterman) não estão plenamente estimados no Lab por restrições de amostra e dependências; diagnósticos básicos estão presentes.
- HMM e changepoint dependem de pacotes opcionais (`hmmlearn`, `ruptures`); há fallbacks controlados.
