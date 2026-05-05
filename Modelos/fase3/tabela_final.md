# Tabela final — Fase 3

Validacao externa: 16 meses (2025-01-01 a 2026-04-01)

## Metricas globais (todos os horizontes agregados)

| Modelo | N | RMSE | MAE | Bias | MAPE |
|---|---:|---:|---:|---:|---:|
| Ensemble_Otimizado_LOO | 16 | 1180.0 | 1019.8 | -252.8 | 9.8% |
| Sazonal_Naive | 16 | 1620.5 | 1381.6 | -66.0 | 13.5% |
| Integrado_Fase2 | 16 | 1714.7 | 1324.1 | -470.1 | 12.6% |
| Regressao_Linear | 16 | 1720.3 | 1342.5 | -668.2 | 12.5% |
| Persistencia_Sazonal | 16 | 1896.7 | 1530.5 | -1109.9 | 14.3% |
| SARIMAX | 16 | 2250.0 | 1936.0 | -1741.5 | 16.7% |
| **NEWAVE (TCC tabela 2)** | n/d | n/d | ~1521 | ~-806 | n/d |

## Metricas por bucket de horizonte

| Modelo | Bucket | N | RMSE | MAE | Bias |
|---|---|---:|---:|---:|---:|
| Sazonal_Naive | 1-3m | 3 | 2443.2 | 2377.6 | -993.8 |
| Sazonal_Naive | 4-6m | 3 | 1962.2 | 1796.9 | -1796.9 |
| Sazonal_Naive | 7-12m | 6 | 1090.5 | 934.2 | +934.2 |
| Sazonal_Naive | 13m+ | 4 | 1164.4 | 994.3 | +428.0 |
| Integrado_Fase2 | 1-3m | 3 | 1672.1 | 1209.1 | +1209.1 |
| Integrado_Fase2 | 4-6m | 3 | 2065.3 | 1715.7 | +420.8 |
| Integrado_Fase2 | 7-12m | 6 | 1601.2 | 1205.3 | -1205.3 |
| Integrado_Fase2 | 13m+ | 4 | 1618.5 | 1294.9 | -1294.9 |
| Persistencia_Sazonal | 1-3m | 3 | 2736.3 | 2257.9 | -2257.9 |
| Persistencia_Sazonal | 4-6m | 3 | 2132.0 | 2092.5 | -2092.5 |
| Persistencia_Sazonal | 7-12m | 6 | 844.0 | 726.9 | -225.8 |
| Persistencia_Sazonal | 13m+ | 4 | 2073.0 | 1768.8 | -837.9 |
| Regressao_Linear | 1-3m | 3 | 2727.3 | 2579.8 | -1676.7 |
| Regressao_Linear | 4-6m | 3 | 2439.7 | 2261.5 | -2261.5 |
| Regressao_Linear | 7-12m | 6 | 789.4 | 598.7 | +356.8 |
| Regressao_Linear | 13m+ | 4 | 927.3 | 840.8 | -254.6 |
| SARIMAX | 1-3m | 3 | 2545.0 | 2005.4 | -2005.4 |
| SARIMAX | 4-6m | 3 | 2987.4 | 2938.6 | -2938.6 |
| SARIMAX | 7-12m | 6 | 1839.3 | 1631.1 | -1631.1 |
| SARIMAX | 13m+ | 4 | 1903.7 | 1589.5 | -811.5 |

## Ensemble otimizado

- Candidatos: Integrado_Fase2, Sazonal_Naive
- Pesos LOO-CV (medios): {'Integrado_Fase2': 0.494, 'Sazonal_Naive': 0.506}
- Pesos LS in-sample: {'Integrado_Fase2': 0.5, 'Sazonal_Naive': 0.5}

## Conformal prediction (alpha=0.10)

- q90 (residuos calibracao) = 2414.6 MWmed
- Cobertura empirica no split test (2a metade): 100%
- Intervalo aplicado a todo horizonte 60m em `intervalo_conformal.csv`