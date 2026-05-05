# TCC — Previsão de Geração Eólica de Longo Prazo (NE)

Pipeline alternativo ao NEWAVE para previsão mensal de geração eólica do subsistema Nordeste, horizonte de 60 meses. Atinge **RMSE 1163 MWmed / MAE 990 / MAPE 9.3%** em validação externa contra 14 meses reais ONS (jan/2025 a fev/2026), batendo o NEWAVE em MAE.

> Detalhes da arquitetura, decisões e resultados em [PLAN.md](PLAN.md), [Explicação.md](Explicação.md), [RESULTADOS_FASE1.md](RESULTADOS_FASE1.md), [RESULTADOS_FASE2.md](RESULTADOS_FASE2.md), [RESULTADOS_FASE3.md](RESULTADOS_FASE3.md).

---

## Como rodar

### 0. Pré-requisitos

- Python 3.10–3.13
- (Opcional, só para coletar ERA5 do zero) chave Copernicus CDS em `~/.cdsapirc`

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Caminho rápido (reproduzir resultados sem recoletar dados)

O dataset bruto (`Dados/dataset_final_modelagem.csv`) já vem versionado no repo. Para rodar só o pipeline de modelagem:

```bash
python scripts/coletar_features_macro.py        # baixa Niño 3.4, AMM, IBC-Br
python scripts/gerar_dataset_limpo.py           # gera dataset_modelagem_limpo.csv
python scripts/gerar_capacidade_projetada.py    # 3 cenários de capacidade
python scripts/gerar_climatologia.py            # climatologia mensal meteo
python scripts/baseline_sazonal_naive.py        # baseline FC × capacidade
python scripts/treinar_fase2.py                 # 5 seeds × 3 arquiteturas
python scripts/treinar_fase3.py                 # ensemble + baselines + IC
```

Tempo total: ~5-10 min em CPU. Resultado final em `Modelos/fase3/tabela_final.md`.

### 3. Caminho completo (recoletar tudo, requer ~/.cdsapirc)

Abrir `pipeline_previsao.ipynb` no Jupyter:

```bash
jupyter notebook pipeline_previsao.ipynb
```

Rodar as cells na ordem:

- **Cells 0-9**: coleta ONS S3 + ERA5 Copernicus, processamento, geração do `dataset_final_modelagem.csv`. Demora horas (download ERA5).
- **Cells 10-20** (Pipeline v2): orquestra os 7 scripts da seção 2 acima e exibe a tabela/plot consolidados.
- **Cells 21+** (apêndice deprecated): pipeline antigo do TCC, mantido só para auditoria.

### 4. Variáveis de ambiente (opcional)

```bash
TCC_ROOT=/caminho/para/o/projeto    # default: cwd
TF_DETERMINISTIC_OPS=1              # já setado pelos scripts
PYTHONHASHSEED=42                   # já setado pelos scripts
```

---

## Estrutura

```
.
├── pipeline_previsao.ipynb     # notebook orquestrador
├── PLAN.md                     # plano detalhado de desenvolvimento
├── Explicação.md               # sumário executivo das mudanças
├── Review.md                   # análise diagnóstica do pipeline antigo
├── RESULTADOS_FASE{1,2,3}.md   # relatórios por fase
├── requirements.txt
│
├── src/                        # módulos importáveis
│   ├── seeds.py                # determinismo (TF + numpy + random)
│   ├── data_prep.py            # split temporal, scaling, imputação por fold
│   ├── features_futuras.py     # contrato PROJECAO_FUTURA + projetor
│   ├── models.py               # build_integrado, ablations, HiperParams
│   ├── train_eval.py           # treino multi-seed + EarlyStopping
│   ├── forecast_recursive.py   # previsão recursiva unificada
│   ├── baselines.py            # persistência sazonal, regressão, SARIMAX
│   └── ensemble.py             # pesos LS/LOO + conformal prediction
│
├── scripts/                    # pipelines stand-alone (entry-points)
│   ├── coletar_features_macro.py
│   ├── gerar_dataset_limpo.py
│   ├── gerar_capacidade_projetada.py
│   ├── gerar_climatologia.py
│   ├── baseline_sazonal_naive.py
│   ├── treinar_fase2.py
│   ├── treinar_fase3.py
│   └── patch_*.py              # one-shots já aplicados ao notebook
│
├── Dados/                      # entradas e artefatos intermediários
│   ├── dataset_final_modelagem.csv     # entrada principal (gerado pelas cells 0-9)
│   ├── dataset_modelagem_limpo.csv     # após Fase 0.6 + 1.3 + 1.5 + 1.6
│   ├── features_macro.csv              # Niño 3.4 + AMM + IBC-Br
│   ├── capacidade_projetada_ne.csv     # 3 cenários × 60 meses
│   └── climatologia_mensal_meteo.csv   # 12 meses × 63 features meteo
│
└── Modelos/                    # saídas
    ├── fase2/                  # modelo integrado treinado + forecast 60m
    ├── fase3/                  # ensemble + tabela final + IC conformal
    └── Comparacoes/
        └── Baseline_Sazonal/   # baseline FC × capacidade (Fase 1.8)
```

---

## Onde estão as previsões (importante!)

**Arquivo único pronto para importar em planilha:** `Modelos/fase3/previsoes_60m_consolidado.csv`

60 linhas (jan/2025 a dez/2029), 10 colunas:
- `data`
- 5 modelos individuais: `Sazonal_Naive`, `Integrado_Fase2`, `Regressao_Linear`, `Persistencia_Sazonal`, `SARIMAX`
- **`Ensemble_Otimizado_LOO`** — combinação ponderada (LOO-CV) dos modelos acima, é o resultado principal
- `IC_p05`, `IC_p95` — intervalo de incerteza 5%-95% (conformal prediction)
- `ONS_real` — geração real ONS nos meses já observados (NaN nos meses futuros)

> **Atenção:** NÃO confundir com `Modelos/Comparacoes/Previsoes_60_Meses/previsoes_60_meses_todos_modelos.csv` — esse é do pipeline antigo (CNN/LSTM/Informer/Ensemble com pesos fixos), diagnosticado como quebrado em [Review.md](Review.md). Se você está vendo previsões com valores negativos absurdos ou oscilando de +20mil a −11mil, é porque abriu esse arquivo.

---

## Resultados (resumo)

| Modelo | RMSE | MAE | Bias | MAPE |
|---|---:|---:|---:|---:|
| **Ensemble Otimizado (LOO-CV)** | **1163** | **990** | **−253** | **9.3%** |
| Baseline Sazonal (FC × cap.) | 1654 | 1420 | −157 | 13.2% |
| Modelo Integrado (Fase 2) | 1796 | 1421 | −445 | 13.4% |
| XGBoost (notebook antigo) | 2098 | 1829 | −1261 | 15.1% |
| Ensemble (notebook antigo) | 3544 | 3222 | −3187 | 26.0% |
| **NEWAVE (TCC tabela 2)** | n/d | ~1521 | ~−806 | n/d |

Tabela completa em [Modelos/fase3/tabela_final.md](Modelos/fase3/tabela_final.md).

---

## Citação

TCC — gustavor272@gmail.com, PUC, 2026.
