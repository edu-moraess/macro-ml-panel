# Macro-ML Panel

Painel Streamlit com 10 exemplos compactos de **Macroeconometria + Machine
Learning**. Cada módulo é deliberadamente enxuto (uma técnica, poucas linhas,
sem enfeite) — a ideia é mostrar que o cruzamento entre as duas áreas não
exige um codebase gigante para gerar um insight relevante.

## Rodando

```bash
pip install -r requirements.txt
streamlit run app.py
```

`hmmlearn` e `ruptures` são opcionais: se não estiverem instalados, os
módulos 2 e 3 caem automaticamente para um fallback equivalente feito só com
scikit-learn/numpy.

## Dados

- **BCB/SGS** (Selic, IPCA, IBC-Br, câmbio PTAX, desemprego PNAD) — API
  pública, sem chave.
- **FRED** (spread 10y-3m, USREC, desemprego EUA, CPI, Fed Funds) — CSV
  público, sem chave.
- **Sem fallback sintético.** Se a API falhar (sem internet, série fora do
  ar, resposta vazia), o módulo mostra um erro explícito na tela em vez de
  disfarçar com dado fake. Exige conexão com a internet para funcionar.

## Os 10 módulos

| # | Módulo | Técnica | O que resolve |
|---|--------|---------|----------------|
| 1 | Recessão | Elastic Net Logit | Seleção de variável na previsão de recessão via spread de juros |
| 2 | Regimes de juros | Hidden Markov Model | Descobre regimes de política monetária sem rótulo |
| 3 | Quebra estrutural | Changepoint detection (Binseg) | Encontra datas de mudança de regime numa série |
| 4 | Curva de Phillips | Gradient Boosting + importância por permutação | Captura não-linearidade desemprego-inflação |
| 5 | Nowcasting | Gradient Boosting | Estima atividade corrente antes da divulgação oficial |
| 6 | Regimes cambiais | K-Means | Agrupa países pelo comportamento cambial revelado |
| 7 | Tendência-ciclo | Autoencoder (gargalo) | Alternativa não-linear ao filtro HP |
| 8 | Sentimento de comunicados | Léxico hawkish/dovish | Texto-como-dado em atas de política monetária |
| 9 | Anomalia macro | Isolation Forest | Alerta precoce de comportamento fora do padrão |
| 10 | Curva de juros | Gaussian Process | Alternativa bayesiana ao NSS/BEIR, com banda de incerteza |

## Estrutura

```
macro_ml_panel/
├── app.py            # navegação entre módulos
├── data_utils.py      # BCB/SGS + FRED + fallback sintético
├── modules/
│   ├── m1_recession.py
│   ├── m2_regime_hmm.py
│   ├── m3_structural_break.py
│   ├── m4_phillips_curve.py
│   ├── m5_nowcasting.py
│   ├── m6_clustering.py
│   ├── m7_trend_cycle.py
│   ├── m8_nlp_sentiment.py
│   ├── m9_anomaly.py
│   └── m10_gp_yield_curve.py
└── requirements.txt
```
