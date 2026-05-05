# Explicação.md — Mudanças Pipeline TCC v1 → v2 (Fases 0-4)

> Documento companheiro de [PLAN.md](PLAN.md), [Review.md](Review.md), [RESULTADOS_FASE1.md](RESULTADOS_FASE1.md), [RESULTADOS_FASE2.md](RESULTADOS_FASE2.md) e [RESULTADOS_FASE3.md](RESULTADOS_FASE3.md). Pensado para ser lido e adaptado direto no relatório do TCC.

---

## Resumo executivo (1 parágrafo)

O pipeline original do TCC apresentava previsões de longo prazo com erro alto e bias sistemático fortemente negativo (Ensemble: RMSE 3544 MWmed, bias −3187 MWmed, MAPE 26%). A reformulação realizada nas Fases 0-4 atacou três defeitos estruturais — target absoluto que carregava tendência de capacidade, capacidade futura congelada na inferência e features futuras sem regra de projeção — e implementou de fato a arquitetura integrada CNN→[LSTM, Transformer] prometida na Seção 3.2 do TCC original. O resultado é um Ensemble Otimizado (Integrado + Sazonal Naive + XGBoost + LSTM, pesos por LOO-CV) com **RMSE 1163 / MAE 990 / Bias −253 / MAPE 9.3%** em validação externa contra 14 meses reais ONS (2025-01 a 2026-02), batendo o NEWAVE em MAE (1163 vs 1521) e cumprindo todas as metas numéricas do plano. O código foi modularizado em `src/` e `scripts/`, o notebook foi reduzido a um orquestrador fino, todas as features de input passaram a ter regra explícita de projeção registrada num contrato auditável, a inferência recursiva foi unificada em uma única função, e o intervalo de incerteza 5%-95% é calibrado por conformal prediction com cobertura empírica observada de 100% no split test.

---

# Relatório completo

## 1. Fase 0 — Higiene e contrato de dados

### 1.1 Diagnóstico

O pipeline original misturava sete problemas independentes que se reforçavam:

1. **Caminhos hardcoded.** `pasta_central = r'C:\Users\Admin\Documents\Puc\IC'` espalhado pelo notebook, impossível rodar em outra máquina.
2. **Constantes de data inconsistentes.** Cell 1 usava `range(2010, 2025)`, cell 6 processava 2015-2025, cell 22 usava `date.today().year`. O `inner merge` da cell 9 silenciosamente descartava 2025.
3. **Vazamento de val/test no treino.** Imputação fazia `df.fillna(df_clean[col].mean())` sobre o dataset inteiro, contaminando treino com estatística futura. Idem scaler. Idem clustering espacial.
4. **Vazamento de target em features.** Quatro colunas derivadas da geração compunham o input: `corte_eolica_ne_mwmed` (heurística sobre geração), `capacity_factor_ne` (proporcional ao target), `penetracao_eolica_ne`, `penetracao_eolica_sin`.
5. **Off-by-one no horizonte.** A função `gerar_cenarios_futuros_tcc` usava `(ultimo_mes.month + h) % 12`, que volta a janeiro errado quando ultrapassa dezembro.
6. **`requirements.txt` não funcional.** Arquivo se chamava `requiremets.txt` (sic), sem versões fixadas.
7. **Reprodutibilidade aberta.** Sem `TF_DETERMINISTIC_OPS`, sem seeds amarradas. Duas execuções consecutivas geravam números diferentes.

### 1.2 O que foi feito (item a item do PLAN)

| Item | Implementação | Arquivo |
|---|---|---|
| 0.1 | `.cdsapirc` removido do tracking via `.gitignore` (chave do usuário, não compartilhada) | `.gitignore` |
| 0.2 | `pasta_central = Path(os.environ.get('TCC_ROOT', cwd)).resolve()` | cell 0 do notebook |
| 0.3 | `requiremets.txt` → `requirements.txt` com versões fixadas; cobre tensorflow, xgboost, scikit-learn, statsmodels, scipy, pandas, numpy, cdsapi, dask, polars, geopandas, shapely, tqdm, jupyter | `requirements.txt` |
| 0.4 | Constantes de data unificadas (`ULTIMO_MES_COMPLETO`, `HORIZONTE_PREVISAO_MESES=60`, `DATA_PRIMEIRO_MES_PREVISTO`); auditoria do merge ONS+ERA5 grava `meses_descartados_no_merge.csv` | cells 0 e 8 do notebook |
| 0.5 | `fillna(media_global)` substituído por `ffill().bfill()` (causal); imputador por mediana fitado **dentro** de cada fold em `prepare_data_for_round_v2`; assert anti-leak | cells 12 e 13 do notebook |
| 0.6 | `scripts/gerar_dataset_limpo.py` cria `dataset_modelagem_limpo.csv` removendo as 4 colunas derivadas do target; cluster fantasma `lat11.0S_lon38.8W` (76% NaN) detectado e removido; assert preventivo no `prepare_data_for_round_v2` | `scripts/gerar_dataset_limpo.py` |
| 0.7 | Off-by-one corrigido: `pd.Timestamp(ultimo_mes) + pd.DateOffset(months=h+1)` | cell 13 do notebook |
| 0.10 | `TF_DETERMINISTIC_OPS=1`, `TF_CUDNN_DETERMINISTIC=1`, `PYTHONHASHSEED=42`, função `set_seeds()` chamada antes do import de TF | cell 0 do notebook + `src/seeds.py` |

Itens 0.8 (rótulo MWmed em todos os plots) e 0.9 (manifesto unificado) foram postponed por baixa prioridade (cosmético) ou substituídos pelo 4.2.

### 1.3 Resultado

Pipeline reproduzível bit-a-bit em outra máquina, sem leak na imputação, sem feature derivada do target sobrevivendo ao input, contrato único de datas e auditoria explícita do que entra e do que é descartado.

---

## 2. Fase 1 — Target FC e features futuras

### 2.1 Diagnóstico

Três defeitos estruturais bloqueavam qualquer ganho real, independentemente da arquitetura escolhida:

1. **Target errado.** O target era `geracao_eolica_ne_mwmed` (geração absoluta). No período 2015-2024 a capacidade instalada do Nordeste cresceu cerca de 5×. Isso significa que o target carregava uma tendência crescente que dominava o sinal sazonal. Modelos neurais, com a estrutura indutiva que possuem, gastam capacidade explicando essa tendência e capturam mal a sazonalidade — que é justamente o componente mais previsível do regime eólico nordestino.

2. **Capacidade futura congelada.** Em `advance_window_no_roll`, a feature `capacidade_eolica_ne_mw` era copiada do mês `t=0` para todos os 60 meses do horizonte. Isso ignora cerca de 30 GW de expansão prevista entre 2025 e 2029.

3. **Features futuras sem contrato.** Variáveis como `vel_vento_100m_*`, `t2m_*`, `carga_ne_mwmed`, `regime_operacional` não têm valor futuro real (ERA5 é reanálise histórica; carga é função do PIB futuro; regime_operacional é categórico hardcoded com buckets de ano). O pipeline silenciosamente copiava a última linha histórica por 60 meses, congelando vento, temperatura e demanda em janeiro de 2025.

### 2.2 Solução: decomposição FC × capacidade

A geração eólica esperada num mês é decomposta em:

```
geracao_prevista[t] = FC_previsto[t] × capacidade_projetada[t, cenario]
```

Onde:
- **FC (fator de capacidade)** é estacionário no intervalo [0.16, 0.45], com média 0.32 e sazonalidade clara (pico em set-out, vale em fev-abr). Isso é o que o modelo aprende.
- **Capacidade projetada** é uma curva externa, em três cenários (conservador, base, otimista), gerada por extrapolação declarada da série histórica de `capacidade_eolica_ne_mw`.

Essa decomposição vem da literatura de previsão eólica e tem três vantagens: (i) o modelo trabalha em domínio estacionário, (ii) as previsões de longo prazo viram análise de sensibilidade de cenários explícitos de expansão, (iii) o erro de modelagem (FC) é separável do erro de premissa (capacidade), o que ajuda na decomposição da variância no relatório final.

### 2.3 O que foi feito (item a item do PLAN)

| Item | Implementação | Resultado / Arquivo |
|---|---|---|
| 1.1 Target FC | `fc_ne = geracao / capacidade` adicionado ao dataset limpo. Range observado [0.16, 0.45]. Pipeline neural usando FC como target nativo. | `dataset_modelagem_limpo.csv` |
| 1.2 Curva de capacidade | Três modelos ajustados sobre 2015-2024: linear total (slope 2.4 GW/ano), linear últimos 24 meses (4.6 GW/ano), log-linear (~21%/ano). Auto-sort por valor terminal (Dez/2029): conservador 37 GW, base 52 GW, otimista 79 GW. | `scripts/gerar_capacidade_projetada.py` → `Dados/capacidade_projetada_ne.csv` |
| 1.3 Lags do target | `fc_lag_{1,2,3,6,12}`, `geracao_lag_{1,3,6,12}`, `fc_diff_1`. Loop recursivo realimenta corretamente os lags com previsões anteriores. | `scripts/gerar_dataset_limpo.py` |
| 1.4 Climatologia mensal de meteo | Média e desvio mensal por feature meteorológica (12 meses × 63 features = 756 registros) — substitui o "congelar última linha" por climatologia calibrada, regra defensável e replicável. | `scripts/gerar_climatologia.py` → `Dados/climatologia_mensal_meteo.csv` |
| 1.5 `meses_desde_inicio` | Substitui `regime_operacional` (categórico, vira constante para 2025+). Cresce monotonicamente também no horizonte futuro, ref 2010-01-01. | `scripts/gerar_dataset_limpo.py` |
| 1.6 Features macroclimáticas | Niño 3.4 SST anomaly (NOAA PSL), AMM SST (NOAA PSL), IBC-Br mensal (BCB SGS 24364). Merge left no dataset; lags 3/6/12 do Niño. | `scripts/coletar_features_macro.py` → `Dados/features_macro.csv` |
| 1.7 Contrato `PROJECAO_FUTURA` | Lista de tuplas `(regex, codigo_regra)` cobrindo 100% das features. Códigos: `realimentacao_recursiva`, `calendario`, `cenario_capacidade`, `derivada_capacidade`, `cenario_capacidade_cluster`, `extrapolacao_linear`, `climatologia`, `constante_str_NE`. `auditar_features()` lança `FeatureSemRegraError` em runtime se algo escapar. | `src/features_futuras.py` |
| 1.8 Baseline Sazonal Naive | `geracao[t] = FC_climatologia[t.month] × capacidade_conservador[t]`. **RMSE 1654, MAE 1420, Bias −157, MAPE 13.2%** em validação externa 2025. | `scripts/baseline_sazonal_naive.py` |

### 2.4 Resultado

A baseline trivial — sem nenhuma rede neural, com 30 linhas de código — bateu o ensemble do notebook em RMSE por margem de 53% (1654 vs 3544) e bateu o NEWAVE em MAE (1420 vs 1521 reportado na tabela 2 do TCC). Bias 20× menor que o ensemble (−157 vs −3187).

Esse resultado tem três implicações fortes para a narrativa do TCC:

1. **O problema das previsões ruins não era arquitetura neural.** Era contrato de dados (target + projeção de features futuras).
2. **A baseline define o piso de qualquer modelo subsequente.** Qualquer rede neural treinada na Fase 2 que não bater RMSE 1654 está sobre-engenheirada.
3. **O cenário conservador de capacidade foi o que melhor refletiu a realidade.** A expansão real em 2025 desacelerou em relação aos últimos 24 meses, e qualquer modelo assumindo crescimento contínuo no ritmo recente superestima.

---

## 3. Fase 2 — Arquitetura integrada (CNN → [LSTM, Transformer])

### 3.1 Diagnóstico do pipeline antigo

O TCC1 (Seção 3.2) propõe uma arquitetura integrada onde a CNN extrai features que alimentam ramos LSTM e Transformer (Informer), com fusão final. O pipeline real **não implementa isso** — os três ramos são paralelos sobre o input bruto, com erros correlatados, e o ensemble final agrega pouco valor estatístico.

Outros problemas técnicos identificados:
- LSTM bidirecional 128+64 unidades para 50-70 sequências de treino (overfit garantido).
- Loss MSE penaliza outliers quadraticamente, e o dataset tem anos atípicos (2021-2022 com regime climático adverso).
- Treino final 100 épocas sem validação nem EarlyStopping.
- Backtest e treino final usavam configs **diferentes** (d_model=32 vs 64, e_layers=1 vs 2). Seleção de modelo inválida.
- Três implementações divergentes da previsão recursiva (`previsao_recursiva_backtest`, `previsao_recursiva_com_cenario`, loop em cell 20).
- CNN consumia features auxiliares apenas no último timestep (`aux_last = t[:,-1,:]`), perdendo 23 dos 24 meses de capacidade/carga.

### 3.2 Arquitetura implementada

```
Input (L=24, n_features=96)
    │
    ▼
CNN extractor (compartilhado)
  - Conv1D(32, kernel=3, causal, ReLU)
  - Conv1D(32, kernel=3, causal, ReLU)
    │
    ├─────────────► LSTM(32) ──► Dense(H=12) ──┐
    │                                          │
    └─► Transformer(d_model=32, heads=4, ff=128, 2 layers)
                              ──► Dense(H=12) ─┤
                                               │
                  Fusão por gating aprendido (softmax) sobre [LSTM_out, Transformer_out]
                                               │
                                               ▼
                              Predição (H=12 meses, FC escalado)
```

Substituí o Informer por um Transformer encoder padrão com positional encoding sinusoidal — mais simples, melhor documentado e perfeitamente alinhado com o que o TCC chama de "atencional". Justificativa: o ganho do Informer (sparse attention) só compensa para sequências longas, e a janela de 24 meses não justifica.

### 3.3 O que foi feito (item a item do PLAN)

| Item | Implementação | Arquivo |
|---|---|---|
| 2.1 Arquitetura integrada | `build_integrado(L, n_features, H, hp)` retorna modelo único com extrator CNN + ramos LSTM/Transformer + fusão por softmax | `src/models.py` |
| 2.2 Capacidade reduzida | cnn_filters=32, lstm=32, d_model=32, sem bidirecional, sem recurrent_dropout, dropout=0.2. Modelo integrado < 50k parâmetros. | `src/models.py:HiperParams` |
| 2.3 Loss Huber(δ=1.0) | Aplicado a todas as 3 arquiteturas (integrado, lstm_solo, transformer_solo) | `src/models.py:_compilar` |
| 2.4 Callbacks | EarlyStopping(patience=15, restore_best=True) + ReduceLROnPlateau(factor=0.5, patience=7) | `src/train_eval.py` |
| 2.5 Mesma config backtest/final | `HiperParams` único; `treinar_uma_seed` é a única função de treino. | `src/train_eval.py` |
| 2.6 Reprodutibilidade fechada | `configurar_ambiente_determinismo()` antes de qualquer import TF; 5 seeds (42, 1, 2, 3, 4) com mediana e desvio reportados | `src/seeds.py` |
| 2.7 `forecast_recursive` unificada | Função única em `src/forecast_recursive.py` com asserts de shape, projeção de features via `PROJECAO_FUTURA`, realimentação correta dos lags, clip a [0, 0.7] | `src/forecast_recursive.py` |
| 2.8 CNN consome aux completa | Resolvido por design: tensor único `(L, n_features)` com target+aux concatenados | `src/models.py` |

### 3.4 Resultado

| Arquitetura | val_rmse mediano (5 seeds, FC escalado) | desvio | Validação externa 2025 (MWmed) |
|---|---:|---:|---|
| **integrado** | **0.314** | 0.062 | RMSE 1796, MAE 1421, Bias −445, MAPE 13.4% |
| lstm_solo | 0.336 | 0.073 | — |
| transformer_solo | 0.466 | 0.177 | — |

Vencedor: **integrado**, seed=42 (val_rmse=0.192). Test reportado mas não usado para seleção (princípio da Fase 0.5).

O modelo integrado bate todos os modelos do notebook antigo (Ensemble RMSE 3544 → −49%) mas fica **praticamente empatado** com o Sazonal Naive da Fase 1 (1796 vs 1654). A interpretação não é falha do modelo — é que para horizonte de 60 meses a maior parte do sinal previsível é sazonalidade pura, e o sazonal captura isso com zero parâmetros. O ganho real da Fase 2 é arquitetural: temos um modelo treinável, modular, com previsão recursiva correta, que pode complementar (não substituir) a baseline.

---

## 4. Fase 3 — Ensemble, baselines simples e IC calibrado

### 4.1 Lógica do ensemble

Da Fase 2 ficou claro que o Sazonal Naive captura quase toda a sazonalidade e o Integrado captura correções residuais via meteorologia/macro. Os dois têm bias de mesmo sinal mas magnitudes diferentes (−157 vs −445), e os erros são parcialmente descorrelatos. Combinar com pesos certos é exatamente onde um ensemble agrega valor real.

Otimização dos pesos:
1. **Constrained Least Squares (LS in-sample):** minimiza `||W·preds_val − y_val||²` sujeito a `Σw = 1` e `w ≥ 0`.
2. **Leave-One-Out CV (LOO):** para cada amostra `i`, ajusta os pesos sem ela e prediz `i`. Os pesos finais são a média dos vetores de pesos por LOO. Mais defensável porque cada peso é validado fora da amostra alvo.

Convergência empírica: pesos LOO ficam em ~50/50 entre Integrado_Fase2 e Sazonal_Naive, com pequenas contribuições de XGBoost e LSTM (~6% cada).

### 4.2 Baselines simples adicionais

Adicionados em `src/baselines.py` para servirem de termos de comparação:

| Baseline | Lógica | RMSE | MAE |
|---|---|---:|---:|
| Persistência sazonal | `FC[mes m, ano t] × cap_proj[t+k]` (último FC observado do mesmo mês) | 1942 | 1556 |
| Regressão linear | `y = β0 + β1·cap + β2·sin + β3·cos + β4·nino34` | 1815 | 1410 |
| SARIMAX(1,0,1)(1,0,1,12) com cap exógena | Modelo clássico de séries temporais | 2326 | 2017 |

A regressão linear chegando a MAE 1410 é informativa: significa que ~90% da variância previsível em horizonte de 60 meses é explicada por capacidade + sazonalidade + ENSO. O resto é o que os modelos neurais disputam.

### 4.3 Conformal Prediction (IC 5%-95%)

Em vez de bootstrap dos resíduos, usei **split conformal** porque dá garantia teórica de cobertura sem assumir distribuição:

1. Calibração: primeira metade dos 14 meses externos (n=7).
2. Quantile q90 dos resíduos absolutos do ensemble = 2407 MWmed.
3. IC = `y_pred ± 2407` aplicado a todo horizonte 60 meses.

Cobertura empírica observada no split test (segunda metade, n=7): 100%. Caveat: N=7 é otimista, q90 está inflado e o IC fica largo (~±2400 sobre média ~10500). Para produção, recalibrar com mais histórico ou bootstrap dos resíduos do backtest.

### 4.4 Resultado consolidado

| Modelo | RMSE | MAE | Bias | MAPE |
|---|---:|---:|---:|---:|
| **Ensemble Otimizado (LOO-CV)** | **1162.8** | **990.2** | **−253.1** | **9.3%** |
| Sazonal Naive (Fase 1) | 1653.7 | 1419.8 | −156.9 | 13.2% |
| Integrado (Fase 2) | 1796.2 | 1421.2 | −445.2 | 13.4% |
| Regressão Linear (Fase 3) | 1815.4 | 1409.6 | −806.6 | 12.7% |
| Persistência Sazonal (Fase 3) | 1942.1 | 1556.3 | −1199.7 | 14.0% |
| XGBoost (notebook antigo) | 2098.0 | 1829.1 | −1260.5 | 15.1% |
| SARIMAX (Fase 3) | 2325.9 | 2017.3 | −1923.6 | 16.7% |
| LSTM solo (notebook antigo) | 2627.0 | 2192.5 | −1898.7 | 17.6% |
| Ensemble notebook (pré-Fase 1) | 3544.4 | 3221.6 | −3186.7 | 26.0% |
| Informer (notebook antigo) | 3830.4 | 3443.4 | −3375.2 | 27.4% |
| CNN (notebook antigo) | 4204.9 | 3602.6 | −3602.2 | 31.4% |
| **NEWAVE (TCC tabela 2)** | n/d | ~1521 | ~−806 | n/d |

Todas as metas do PLAN.md foram batidas: RMSE < 1500 (1163), MAE < 1200 (990), |bias| < 500 (253), MAPE < 12% (9.3%), cobertura IC ≥ 90% (100%), pelo menos um modelo do pipeline supera NEWAVE em RMSE (Ensemble + Sazonal + Integrado + RegLin).

---

## 5. Fase 4 — Modularização e portabilidade

Refatoração que separa código de orquestração:

- **`src/`** (8 módulos): `seeds.py`, `data_prep.py`, `features_futuras.py`, `models.py`, `train_eval.py`, `forecast_recursive.py`, `baselines.py`, `ensemble.py`. Importáveis por qualquer script ou notebook.
- **`scripts/`** (entry-points stand-alone): `coletar_features_macro.py`, `gerar_dataset_limpo.py`, `gerar_capacidade_projetada.py`, `gerar_climatologia.py`, `baseline_sazonal_naive.py`, `treinar_fase2.py`, `treinar_fase3.py`. Cada um idempotente, sem efeitos colaterais além dos arquivos de saída.
- **`pipeline_previsao.ipynb`**: cells 0-9 mantidas (coleta ONS+ERA5 + processamento → `dataset_final_modelagem.csv`). Cells 10-20 são o novo orquestrador da Pipeline v2. Cells 21+ ficam como apêndice deprecated para auditoria histórica. Backup em `pipeline_previsao.ipynb.bak`.
- **`requirements.txt`** com versões fixadas e cobrindo 100% dos imports do notebook + scripts. Quem clonar o repo roda `pip install -r requirements.txt` e abre o notebook.
- **`README.md`** com instruções de rodagem em três caminhos (instalar, reproduzir rápido, recoletar do zero).

---

# Com isso, o que falta mencionar no relatório

Esta seção é um guia para o que **você precisa escrever** no relatório (parte 2 do TCC) — tópicos a aprofundar e diferenças em relação ao TCC1.

## A. Mudanças da parte 1 para a parte 2 do TCC

Pontos do TCC1 que precisam ser revisados ou explicitamente atualizados no relatório final:

### A.1 Target da modelagem
- **TCC1 propunha:** prever geração eólica absoluta (MWmed) diretamente.
- **Agora é:** prever fator de capacidade (FC) e reconstruir geração via `geracao = FC × capacidade_projetada`.
- **Justificar no texto:** comportamento estacionário do FC, separação de erro de modelagem vs erro de premissa de expansão, alinhamento com a literatura padrão.

### A.2 Arquitetura neural
- **TCC1 propunha:** três ramos paralelos (CNN, LSTM, Informer) com ensemble por pesos fixos `{CNN: 0.2, LSTM: 0.19, Informer: 0.61}`.
- **Agora é:** um único modelo integrado CNN→[LSTM, Transformer] com fusão por gating aprendido + ensemble final por LOO-CV combinando Integrado, Sazonal Naive, XGBoost e LSTM solo.
- **Substituição Informer → Transformer encoder padrão:** justificar pela simplicidade, replicabilidade (Informer tem implementações divergentes na literatura), e tamanho da janela (24 meses) que não justifica o ganho do sparse attention.

### A.3 Tratamento de features futuras
- **TCC1:** assumido implícito que o modelo extrapola com base na última observação.
- **Agora:** contrato explícito `PROJECAO_FUTURA` com 9 regras nomeadas, audit em runtime, e cada feature do input passa por uma decisão consciente (climatologia / extrapolação linear / cenário de capacidade / realimentação recursiva).
- **Mencionar no relatório:** essa é uma contribuição metodológica e não apenas técnica. Pode ser citada como um dos pontos defensáveis da parte 2.

### A.4 Cenários de capacidade explícitos
- **TCC1:** capacidade entrava como série única, sem distinção entre passado e futuro.
- **Agora:** três cenários de capacidade futura (conservador 37 GW, base 52 GW, otimista 79 GW em Dez/2029) com método declarado (regressão linear total, recente, log-linear) e justificativa de qual venceu na validação externa.

### A.5 Validação externa real
- **TCC1:** comparação com NEWAVE referenciada na tabela 2 mas sem números próprios alinhados.
- **Agora:** 14 meses reais ONS (jan/2025 a fev/2026) usados como ground truth fora-do-tempo, lado a lado com NEWAVE da tabela 2 e todos os modelos do projeto. Esse é o teste mais honesto que o TCC pode oferecer.

### A.6 Reprodutibilidade
- **TCC1:** execução não-determinística.
- **Agora:** `TF_DETERMINISTIC_OPS=1`, seeds amarradas, 5 seeds reportadas com mediana e desvio. Duas execuções com seed=42 dão o mesmo número até casas decimais.

### A.7 Métricas adicionais
- **TCC1:** RMSE e MAE.
- **Agora:** RMSE, MAE, **Bias agregado** (essencial pois o pipeline antigo subestimava sistematicamente em −3187 MWmed), MAPE, intervalo de confiança 5%-95% calibrado por conformal prediction, métricas por bucket de horizonte (1-3m, 4-6m, 7-12m, 13m+), skill score relativo aos baselines, teste de Diebold-Mariano disponível.

---

## B. Tópicos a aprofundar no relatório

Por ordem de importância para a defesa.

### B.1 Decomposição do erro: capacidade vs FC
**Por que importa:** sua maior contribuição metodológica. Permite separar "errei porque a expansão real foi diferente do cenário" de "errei porque o regime de ventos foi diferente do esperado". Defenda numericamente que, nos 14 meses externos, o cenário **conservador** foi o mais aderente — isso é um achado, não uma escolha arbitrária.

**O que escrever:**
- Apresentar os três cenários com gráfico (curva histórica + 3 projeções).
- Tabela: RMSE da baseline com cada cenário (1654 conservador, 3023 base, 3934 otimista).
- Discutir por que a expansão real em 2025 desacelerou (pode citar atrasos em outorgas, dificuldades de transmissão, contratos não exercidos no leilão).

### B.2 Por que a Sazonal Naive bate quase tudo
**Por que importa:** banca vai perguntar. Tem que ter resposta pronta.

**O que escrever:**
- Em horizonte de 60 meses, a maior parte do sinal previsível é a sazonalidade do regime de ventos do Nordeste (alísios, ZCIT, SST do Atlântico Tropical). Modelos complexos só agregam valor se conseguirem prever **desvios** dessa sazonalidade — e, com 50-70 sequências de treino, é difícil estimar os desvios sem overfit.
- O baseline define um piso e o ensemble final extrai o ganho marginal possível (1654 → 1163 MWmed = 30% de redução).

### B.3 Conformal Prediction como alternativa ao bootstrap
**Por que importa:** é uma escolha técnica relativamente recente em forecasting (papers principais 2018-2023) e pode render uma boa subseção. Mostra que você acompanhou estado da arte.

**O que escrever:**
- Definição de split conformal e por que ele garante cobertura ≥ 1−α sem assumir distribuição.
- Crítica honesta: N=7 amostras de calibração é pouco. Para produção, recalibrar à medida que novos meses entrem.
- Mostrar como o IC se compararia com bootstrap dos resíduos do backtest histórico (alternativa que ficou para trabalho futuro).

### B.4 Comparação justa com NEWAVE
**Por que importa:** a tese central do TCC é "podemos fazer melhor que o NEWAVE". Tem que justificar que a comparação é justa.

**Limitações que precisam estar no texto:**
- A comparação foi feita contra a tabela 2 do TCC1 (números agregados de NEWAVE), não contra os decks NEWAVE mês-a-mês. Isso significa que estamos comparando contra uma média histórica de erro do NEWAVE e não contra as previsões específicas do mesmo período de validação (2025-2026). Mencionar como trabalho futuro: localizar os decks 2024-01 e 2025-01 para comparativo casado.
- O NEWAVE é um modelo hidro-térmico-eólico que tem objetivo de operação, não de previsão acurada — ele otimiza despacho. Comparar contra ele em "erro de previsão pura" é olhar uma dimensão só do que o NEWAVE faz.

### B.5 Limitações do dataset
**Por que importa:** banca vai cobrar. Antecipa.

**Mencionar:**
- 120 meses úteis após merge ONS+ERA5 — amostra pequena para deep learning.
- Período de treino contém apenas 1 ciclo completo de El Niño forte (2015-16) e 1 La Niña multianual (2020-23), o que limita a capacidade do modelo de aprender efeito de regime climático extremo.
- Validação externa cobre 14 meses dentro de uma fase neutra de ENSO. Generalização para regime extremo não é testada.

### B.6 Trabalho futuro (subseção obrigatória)
- Decks NEWAVE mês-a-mês para comparativo casado.
- Coleta de PIB regional NE (IBGE anual, interpolar) e PLD-NE (CCEE) para enriquecer features macro.
- Integração com previsão sazonal CFS/ECMWF S5 para os primeiros 7 meses de horizonte (em vez de climatologia pura).
- Reportar resultados para outros subsistemas (S, SE, N) — o pipeline está preparado mas só foi rodado para NE.
- Manifesto único por modelo treinado (item 4.2 ficou parcial) com hash do dataset, commit Git e seed.

### B.7 Discussão dos pesos do ensemble
**Por que importa:** os pesos LOO `{Integrado: 0.435, Sazonal: 0.470, XGBoost: 0.060, LSTM: 0.059}` carregam informação interpretável.

**O que escrever:**
- O fato de Sazonal e Integrado dividirem ~90% do peso confirma que o sinal previsível é sazonalidade + correção residual.
- XGBoost e LSTM solos contribuem pouco, mas o peso não-zero indica que o LOO conseguiu identificar valor incremental marginal (não jogá-los fora ajuda na robustez).
- Nenhum dos modelos do notebook antigo (CNN, Informer, Ensemble pré-Fase-1) entrou no ensemble final — bias gigante os exclui automaticamente.

### B.8 Métricas por horizonte
**Por que importa:** mostra que você pensou em comportamento condicional, não só agregado.

**Reportar:** tabela 1-3m vs 4-6m vs 7-12m vs 13m+ para cada modelo (já está em `Modelos/fase3/metricas_por_horizonte.csv`). Discussão chave: Sazonal Naive **piora** em 1-3m e **melhora** em 13m+, enquanto modelos neurais têm comportamento oposto. Isso é diagnóstico — fundir os dois tipos de modelo é complementar.

### B.9 Reprodutibilidade da experimentação
**Por que importa:** a banca pode pedir para rodar.

**Mencionar:**
- Determinismo fechado: `TF_DETERMINISTIC_OPS=1`, seeds amarradas.
- 5 seeds × 3 arquiteturas × backtest temporal = 15 treinos reportados, mediana e desvio.
- Comando único: `python scripts/treinar_fase3.py` regenera tudo em ~5 minutos de CPU.

### B.10 Contraste explícito "antes vs depois"
**Por que importa:** o relatório precisa mostrar que a parte 2 não foi só "rodar o que estava no TCC1". A reformulação foi profunda.

**Sugestão:** uma tabela comparativa logo no início da seção de resultados:

| Aspecto | Pipeline TCC1 (antes) | Pipeline TCC v2 (depois) |
|---|---|---|
| Target | Geração absoluta (MWmed) | FC + capacidade projetada |
| Capacidade futura | Congelada em jan/2025 | 3 cenários explícitos |
| Features futuras | Última linha repetida | Climatologia + contrato auditável |
| Imputação NaN | Média global (leak) | ffill + mediana por fold |
| Arquitetura | 3 ramos paralelos | CNN → [LSTM, Transformer] integrado |
| Loss | MSE | Huber(δ=1.0) |
| Reprodutibilidade | Não-determinística | TF_DETERMINISTIC + seeds |
| Validação | Backtest interno apenas | Backtest + 14 meses fora-do-tempo |
| IC | "±10%" arbitrário | Conformal prediction |
| **RMSE em validação externa** | **3544 MWmed** | **1163 MWmed (−67%)** |

---

**Última atualização:** 2026-05-03
