# Resultados Fase 2 — Arquitetura integrada CNN→[LSTM, Transformer]

> Relatorio executivo. Detalhes em [PLAN.md](PLAN.md) e codigo em [src/](src/).

## TL;DR

A arquitetura integrada prometida no TCC (CNN extrator → ramos LSTM e Transformer fundidos por gating aprendido) **bateu todos os modelos do notebook** na validacao externa 2025-2026 (49% melhor RMSE que o Ensemble), e ficou **praticamente empatada com o Baseline Sazonal Naive da Fase 1**.

| Modelo | RMSE | MAE | Bias | MAPE |
|---|---:|---:|---:|---:|
| **Integrado (Fase 2)** | **1796.2** | **1421.2** | **-445.2** | **13.4%** |
| Baseline Sazonal (Fase 1, conservador) | 1653.7 | 1419.8 | -156.9 | 13.2% |
| LSTM solo (Fase 2) | — backtest test_rmse 0.61 (escalado) — perde por val | | | |
| Transformer solo (Fase 2) | — backtest test_rmse 0.84 (escalado) — perde por val | | | |
| Ensemble (notebook, pré-Fase 1) | 3544.4 | 3221.6 | -3186.7 | 26.0% |
| NEWAVE (TCC tabela 2) | n/d | ~1521 | ~-806 | n/d |

Periodo: 14 meses (2025-01 a 2026-02). Validacao externa contra `eolica_NE_mensal_MWmed_2025-01_ate_hoje.csv`.

## O que foi implementado

Conforme [PLAN.md Fase 2](PLAN.md):

| Item | Status | Onde |
|---|---|---|
| 2.1 Arquitetura integrada CNN → [LSTM, Transformer] | ✅ | [src/models.py](src/models.py) `build_integrado` |
| 2.2 Capacidade reduzida (cnn=32, lstm=32, d_model=32) | ✅ | [src/models.py](src/models.py) `HiperParams` |
| 2.3 Loss Huber(δ=1.0) | ✅ | [src/models.py](src/models.py) `_compilar` |
| 2.4 EarlyStopping + ReduceLROnPlateau | ✅ | [src/train_eval.py](src/train_eval.py) |
| 2.5 Mesma config no backtest e treino final | ✅ | `HiperParams` unico, mesmo `treinar_uma_seed` |
| 2.6 Reprodutibilidade (TF_DETERMINISTIC_OPS antes do import) | ✅ | [src/seeds.py](src/seeds.py) + entry-point do script |
| 2.7 `forecast_recursive` unificado, asserts de shape | ✅ | [src/forecast_recursive.py](src/forecast_recursive.py) |
| 2.8 CNN consome aux ao longo da janela | ✅ | features unificadas no input — extrator ve a janela toda |

## Selecao de modelo

| Arquitetura | val_rmse mediano (5 seeds) | desvio | test_rmse mediano | desvio |
|---|---:|---:|---:|---:|
| **integrado** | **0.314** | 0.062 | **0.445** | 0.061 |
| lstm_solo | 0.336 | 0.073 | 0.614 | 0.119 |
| transformer_solo | 0.466 | 0.177 | 0.840 | 0.178 |

(escalado pelo `RobustScaler` no espaco de FC; valores absolutos comparaveis entre arquiteturas, nao com baselines em MWmed)

Criterio: menor `val_rmse` mediano. Vencedor: **integrado**, seed=42 (val_rmse=0.192). Test foi reportado mas **nao usado para selecao** (Fase 0.5).

## Setup

- **Target**: `fc_ne` (fator de capacidade). Reconstrucao para MWmed: `geracao = fc_previsto × capacidade_projetada_conservador`.
- **Janela / horizonte**: L=24 meses, H=12 meses (modelo prediz 12 a frente, mas usado recursivamente 1-passo-por-vez para 60m).
- **Split temporal**: train ≤ 2022-12 (49 sequencias), val = 2023 (1 sequencia), test = 2024 (1 sequencia). O dataset tem so 120 meses utiles, entao val/test sao apertados — ja documentado como limitacao em [Review.md](Review.md).
- **Multi-seed**: 5 seeds (42, 1, 2, 3, 4). Reportado mediana e desvio.
- **Treino final**: re-treinado em **todo historico 2015-2024** com a seed vencedora antes de gerar o forecast.

## Forecast 60 meses

`Modelos/fase2/forecast_60meses.csv` cobre 2025-01 a 2029-12 com cenario de capacidade conservador (vencedor da Fase 1, 2.4 GW/ano).

A inferencia usa [src/forecast_recursive.py](src/forecast_recursive.py) — funcao **unica** que substitui as 3 versoes divergentes do notebook. Para cada passo:
1. Monta janela de L=24 meses (historico + previsoes anteriores)
2. Asserta shape `(L, n_features)`
3. Predicao escalada → inverse_transform → clip a [0, 0.7]
4. Reconstroi `geracao = fc × capacidade_projetada[mes]`
5. Realimenta `fc_lag_*` e `geracao_lag_*` com a previsao anterior, `fc_diff_1`, e meteorologia/macro via climatologia.

## Comparacao com Fase 1

O baseline sazonal naive ainda **vence em RMSE por margem pequena** (1654 vs 1796). Por que?

1. **Tamanho de amostra**: 14 meses de validacao externa. A diferenca de 142 MWmed em RMSE pode ser ruido amostral.
2. **Teorema do "no free lunch para horizonte longo"**: dado que a meteorologia futura e *climatologia* (Niño 3.4 do NOAA cobre ate dez/2026 mas o modelo nao distingue muito), a maior parte do sinal previsivel a 60m e sazonalidade pura — e o sazonal naive captura isso com 0 parametros.
3. **Bias maior do integrado** (-445 vs -157): o modelo neural ainda subestima, possivelmente por overfit ao regime 2021-2022 (anos atipicos).

**Implicacao**: o ganho real da Fase 2 e **arquitetural** (modelo treinavel, modular, com previsao recursiva correta), nao tanto numerico contra esse baseline. Para evolucao na Fase 3:
- Ensemble Integrado + Sazonal por validacao (provavel < 1500 RMSE)
- Calibracao de incerteza por bootstrap dos residuos
- Comparativo formal com NEWAVE no mesmo protocolo

## Reproducibilidade

```bash
cd C:\Users\Admin\Documents\Puc\IC
.venv\Scripts\python.exe scripts/coletar_features_macro.py   # se ainda nao
.venv\Scripts\python.exe scripts/gerar_dataset_limpo.py      # se ainda nao
.venv\Scripts\python.exe scripts/gerar_capacidade_projetada.py
.venv\Scripts\python.exe scripts/gerar_climatologia.py
.venv\Scripts\python.exe scripts/treinar_fase2.py            # 5 seeds × 3 arq + forecast
```

Tempo total: ~3-4 minutos em CPU (sem GPU em Windows nativo).

## Artefatos

- [Modelos/fase2/modelo_final_integrado.keras](Modelos/fase2/modelo_final_integrado.keras)
- [Modelos/fase2/resultados_treino.csv](Modelos/fase2/resultados_treino.csv) — 15 linhas (3 arq × 5 seeds)
- [Modelos/fase2/forecast_60meses.csv](Modelos/fase2/forecast_60meses.csv)
- [Modelos/fase2/validacao_externa_2025.csv](Modelos/fase2/validacao_externa_2025.csv)
- [Modelos/fase2/validacao_externa_2025.png](Modelos/fase2/validacao_externa_2025.png)
- [Modelos/fase2/selecao_modelo.json](Modelos/fase2/selecao_modelo.json)

## Notas tecnicas

- TF 2.21 + Keras 3 (Python 3.13). `keras.ops.expand_dims` em vez de `tf.expand_dims` (KerasTensor nao aceita TF ops diretas).
- O dataset agora tem 96 features modelaveis (fora `data`, `ano`, `mes`, `trimestre`, `subsistema`, `geracao_eolica_ne_mwmed`, `fc_ne`).
- `auditar_features` levanta erro se alguma feature nao tem regra de projecao em `PROJECAO_FUTURA` — protecao contra "feature aparece no input mas vira zero no futuro".

## O que falta para Fase 3

Tudo conforme [PLAN.md Fase 3](PLAN.md):
1. Ensemble otimizado (constrained least squares: w_integrado, w_sazonal, w_xgb)
2. NEWAVE como baseline real no comparativo
3. Conformal prediction para IC calibrado 5%-95%
4. Tabela final por horizonte (1m, 6m, 12m, 24m, 60m)
5. Baselines simples adicionais (persistencia, regressao linear, SARIMAX)

---

**Data:** 2026-05-02
**Periodo de validacao externa:** 2025-01 a 2026-02 (14 meses, dados ONS reais)
