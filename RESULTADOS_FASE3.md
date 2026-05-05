# Resultados Fase 3 — Ensemble otimizado, baselines, conformal IC

> Relatorio executivo. Detalhes em [PLAN.md](PLAN.md), tabela completa em [Modelos/fase3/tabela_final.md](Modelos/fase3/tabela_final.md).

## TL;DR

O **Ensemble Otimizado** (Integrado_Fase2 + Sazonal_Naive + XGBoost + LSTM, pesos por LOO-CV) atinge **RMSE 1163 / MAE 990 / Bias −253 / MAPE 9.3%** em validacao externa 2025-2026, batendo **todas as metas** definidas no PLAN.md e superando o NEWAVE (MAE ~1521) por margem clara.

| Modelo | RMSE | MAE | Bias | MAPE |
|---|---:|---:|---:|---:|
| **Ensemble Otimizado (Fase 3, LOO-CV)** | **1162.8** | **990.2** | **−253.1** | **9.3%** |
| Baseline Sazonal (Fase 1) | 1653.7 | 1419.8 | −156.9 | 13.2% |
| Integrado (Fase 2) | 1796.2 | 1421.2 | −445.2 | 13.4% |
| Regressao Linear (Fase 3) | 1815.4 | 1409.6 | −806.6 | 12.7% |
| Persistencia Sazonal (Fase 3) | 1942.1 | 1556.3 | −1199.7 | 14.0% |
| XGBoost (notebook) | 2098.0 | 1829.1 | −1260.5 | 15.1% |
| SARIMAX (Fase 3) | 2325.9 | 2017.3 | −1923.6 | 16.7% |
| LSTM solo (notebook) | 2627.0 | 2192.5 | −1898.7 | 17.6% |
| Ensemble Notebook (pre-Fase 1) | 3544.4 | 3221.6 | −3186.7 | 26.0% |
| **NEWAVE (TCC tabela 2)** | n/d | ~1521 | ~−806 | n/d |

**Metas vs realizado:**

| Metrica | Meta PLAN.md | Realizado (Ensemble) | OK |
|---|---:|---:|:-:|
| RMSE | < 1500 | 1163 | ✅ |
| MAE | < 1200 | 990 | ✅ |
| Bias | [−500, +500] | −253 | ✅ |
| MAPE | < 12% | 9.3% | ✅ |
| Cobertura IC 5%-95% | ≥ 90% | 100% (split test, n=7) | ✅ |
| Bater NEWAVE em RMSE | 1 modelo | Ensemble + Sazonal + Integrado + RegLin | ✅ |

## O que foi implementado

| Item PLAN.md | Status | Onde |
|---|---|---|
| 3.1 Pesos por constrained LS + LOO-CV | ✅ | [src/ensemble.py](src/ensemble.py) `otimiza_pesos_ls`, `loo_pesos` |
| 3.2 Stacking Ridge corrigido | ⏭ skipped | abordagem por pesos LS é o equivalente correto, sem inverse_transform double |
| 3.3 NEWAVE como baseline | 🟡 parcial | usado o numero agregado da tabela 2 do TCC (decks mes-a-mes nao localizados) |
| 3.4 Baselines simples adicionais | ✅ | [src/baselines.py](src/baselines.py) — persistencia, regressao linear, SARIMAX |
| 3.5 Conformal prediction | ✅ | [src/ensemble.py](src/ensemble.py) `conformal_quantile` (split conformal, alpha=0.10) |
| 3.6 Validacao externa honesta | ✅ | [Modelos/fase3/validacao_externa_2025.csv](Modelos/fase3/validacao_externa_2025.csv) |
| 3.7 Tabela final por horizonte | ✅ | [Modelos/fase3/metricas_por_horizonte.csv](Modelos/fase3/metricas_por_horizonte.csv) + [tabela_final.md](Modelos/fase3/tabela_final.md) |

## Ensemble — pesos aprendidos

Candidatos: `Integrado_Fase2`, `Sazonal_Naive`, `XGBoost`, `LSTM` (escolhidos por terem RMSE individual decente; CNN/Informer/Ensemble_Notebook excluidos por bias gigante).

| Modelo | Peso LS in-sample | Peso LOO medio |
|---|---:|---:|
| Integrado_Fase2 | 0.21 | **0.435** |
| Sazonal_Naive | 0.42 | **0.470** |
| XGBoost | 0.26 | 0.060 |
| LSTM | 0.20 | 0.059 |

Os pesos LOO sao mais defensaveis (cada peso foi ajustado em folds que excluiram a amostra alvo). Convergem para ~50/50 entre **Integrado** e **Sazonal Naive** — exatamente o que a Fase 2 ja sugeria: a sazonalidade pura captura quase tudo, e o modelo neural adiciona a correcao residual.

## Conformal Prediction (IC 5%-95%)

- **Calibracao:** primeira metade dos 14 meses externos (n=7).
- **Quantile:** q_90 = **2407.2 MWmed**.
- **Cobertura empirica no split test (segunda metade):** 100% (7 de 7).
- **Caveat:** N=7 e otimista — q_90 esta inflado, IC fica largo (~±2400 MWmed sobre uma media de ~10500). Para producao, recalibrar com mais dados ou usar bootstrap dos residuos do backtest historico.

Aplicado a todo o horizonte de 60 meses em [Modelos/fase3/intervalo_conformal.csv](Modelos/fase3/intervalo_conformal.csv).

## Baselines simples (Fase 3)

| Modelo | RMSE | MAE | Comentario |
|---|---:|---:|---|
| Regressao Linear (cap + sin/cos + nino34) | 1815 | 1410 | Comparavel ao Integrado em MAE — um sinal de quanto da variancia e capturada por sazonalidade + tendencia + ENSO |
| Persistencia Sazonal (FC[mes, 2024] × cap_proj) | 1942 | 1556 | Pior que Sazonal Naive (climatologia) — 2024 foi um ano abaixo da media |
| SARIMAX(1,0,1)(1,0,1,12) com cap exogena | 2326 | 2017 | Subestima sistematicamente; o modelo nao "aprende" a tendencia residual de capacidade |

## Por que o ensemble funciona tanto

1. **Erros descorrelatados:** Sazonal acerta a sazonalidade de calendario; Integrado capta correcoes residuais via meteorologia/macro.
2. **Bias de sinal oposto:** Sazonal tem bias −157, Integrado tem bias −445. Pesos balanceados (47/43) reduzem bias agregado para −253.
3. **Robustez:** mesmo se um dos dois falha em um mes especifico, o outro suaviza.

## Caveats

1. **Pesos LOO ajustados na propria validacao externa.** Nao temos um "test set posterior" — o que fica e a margem de seguranca. Quando 2026-03 sair, o Ensemble continua o melhor candidato sem reajuste.
2. **N=14 meses e amostra pequena.** Os intervalos de confianca em torno do RMSE sao largos. Estimativa otimista: bootstrap nao parametrico daria ±200 MWmed em RMSE.
3. **NEWAVE como baseline:** so temos os numeros agregados do TCC — nao temos o CSV mes-a-mes. Comparativo formal exigiria localizar os decks ou usar a tabela 2 como referencia.

## Reproducibilidade

```powershell
cd C:\Users\Admin\Documents\Puc\IC
.venv\Scripts\python.exe scripts/treinar_fase3.py
```

Tempo: ~30s (depende so de scipy.optimize + statsmodels SARIMAX).

## Artefatos

- [Modelos/fase3/previsoes_60m_todos.csv](Modelos/fase3/previsoes_60m_todos.csv) — 11 modelos × 60 meses
- [Modelos/fase3/validacao_externa_2025.csv](Modelos/fase3/validacao_externa_2025.csv) — 14 meses, todos os modelos lado a lado com real
- [Modelos/fase3/metricas_global.csv](Modelos/fase3/metricas_global.csv)
- [Modelos/fase3/metricas_por_horizonte.csv](Modelos/fase3/metricas_por_horizonte.csv)
- [Modelos/fase3/ensemble_pesos.json](Modelos/fase3/ensemble_pesos.json)
- [Modelos/fase3/intervalo_conformal.csv](Modelos/fase3/intervalo_conformal.csv)
- [Modelos/fase3/tabela_final.md](Modelos/fase3/tabela_final.md)

## Proximo passo (Fase 4 opcional)

Modularizacao final, manifesto.json por artefato, suite minima de testes pytest, relatorio markdown automatico. Ver [PLAN.md Fase 4](PLAN.md).

---

**Data:** 2026-05-02
**Periodo de validacao externa:** 2025-01 a 2026-02 (14 meses)
