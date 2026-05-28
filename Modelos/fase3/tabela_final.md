# Tabela final — Fase 3

Validacao externa: 16 meses (2025-01-01 a 2026-04-01)

## Metricas globais (todos os horizontes agregados)

| Modelo | N | RMSE | MAE | Bias | MAPE |
|---|---:|---:|---:|---:|---:|
| Ensemble_Otimizado_LOO | 16 | 1221.1 | 1028.4 | -285.7 | 9.9% |
| Sazonal_Naive | 16 | 1620.8 | 1382.5 | -66.8 | 13.5% |
| Regressao_Linear | 16 | 1720.6 | 1343.3 | -669.1 | 12.5% |
| Integrado_Fase2 | 16 | 1741.6 | 1397.1 | -499.5 | 13.4% |
| Persistencia_Sazonal | 16 | 1897.4 | 1531.3 | -1110.7 | 14.3% |
| SARIMAX | 16 | 2250.7 | 1936.9 | -1742.4 | 16.7% |
| **NEWAVE (TCC tabela 2)** | n/d | n/d | ~1521 | ~-806 | n/d |

## Metricas por bucket de horizonte

| Modelo | Bucket | N | RMSE | MAE | Bias |
|---|---|---:|---:|---:|---:|
| Sazonal_Naive | 1-3m | 3 | 2443.2 | 2377.6 | -993.8 |
| Sazonal_Naive | 4-6m | 3 | 1962.2 | 1796.9 | -1796.9 |
| Sazonal_Naive | 7-12m | 6 | 1090.5 | 934.2 | +934.2 |
| Sazonal_Naive | 13m+ | 4 | 1166.0 | 997.7 | +424.6 |
| Integrado_Fase2 | 1-3m | 3 | 1865.0 | 1396.9 | +1288.6 |
| Integrado_Fase2 | 4-6m | 3 | 1957.5 | 1733.9 | +202.2 |
| Integrado_Fase2 | 7-12m | 6 | 1568.7 | 1207.6 | -1124.9 |
| Integrado_Fase2 | 13m+ | 4 | 1720.4 | 1428.8 | -1428.8 |
| Persistencia_Sazonal | 1-3m | 3 | 2736.3 | 2257.9 | -2257.9 |
| Persistencia_Sazonal | 4-6m | 3 | 2132.0 | 2092.5 | -2092.5 |
| Persistencia_Sazonal | 7-12m | 6 | 844.0 | 726.9 | -225.8 |
| Persistencia_Sazonal | 13m+ | 4 | 2075.4 | 1772.3 | -841.4 |
| Regressao_Linear | 1-3m | 3 | 2727.3 | 2579.8 | -1676.7 |
| Regressao_Linear | 4-6m | 3 | 2439.7 | 2261.5 | -2261.5 |
| Regressao_Linear | 7-12m | 6 | 789.4 | 598.7 | +356.8 |
| Regressao_Linear | 13m+ | 4 | 929.9 | 844.3 | -258.0 |
| SARIMAX | 1-3m | 3 | 2545.0 | 2005.4 | -2005.4 |
| SARIMAX | 4-6m | 3 | 2987.4 | 2938.6 | -2938.6 |
| SARIMAX | 7-12m | 6 | 1839.3 | 1631.1 | -1631.1 |
| SARIMAX | 13m+ | 4 | 1907.0 | 1592.9 | -814.9 |

## Ensemble otimizado

- Candidatos: Integrado_Fase2, Sazonal_Naive
- Pesos LOO-CV (medios): {'Integrado_Fase2': 0.491, 'Sazonal_Naive': 0.509}
- Pesos LS in-sample: {'Integrado_Fase2': 0.464, 'Sazonal_Naive': 0.536}

## Conformal prediction (alpha=0.10)

- q90 (residuos calibracao) = 2573.1 MWmed
- Cobertura empirica no split test (2a metade): 100%
- Intervalo aplicado a todo horizonte 60m em `intervalo_conformal.csv`