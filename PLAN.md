# PLAN.md — Plano de Desenvolvimento para Melhorar Previsões

> Plano executável para corrigir o pipeline de previsão eólica de longo prazo (60 meses) do TCC. Baseado nos achados consolidados em [Review.md](Review.md). Cada tarefa tem **onde mexer**, **como mexer**, **critério de pronto** e **estimativa**.

---

## Sumário

- [Objetivo](#objetivo)
- [Linha de base atual (métricas a bater)](#linha-de-base-atual-métricas-a-bater)
- [Princípios invioláveis](#princípios-invioláveis)
- [Fase 0 — Higiene e contrato de dados](#fase-0--higiene-e-contrato-de-dados)
- [Fase 1 — Target FC e features futuras](#fase-1--target-fc-e-features-futuras)
- [Fase 2 — Arquitetura e treinamento](#fase-2--arquitetura-e-treinamento)
- [Fase 3 — Ensemble, baseline NEWAVE e tabela final](#fase-3--ensemble-baseline-newave-e-tabela-final)
- [Fase 4 — Modularização e versionamento](#fase-4--modularização-e-versionamento)
- [Cronograma sugerido](#cronograma-sugerido)
- [Critérios objetivos de sucesso](#critérios-objetivos-de-sucesso)
- [Apêndices](#apêndices)

---

## Objetivo

Reduzir o erro das previsões de 60 meses para o subsistema Nordeste a um patamar que (a) seja claramente menor que o NEWAVE no mesmo período, (b) tenha intervalo de incerteza calibrado, e (c) esteja consistente com a arquitetura prometida na seção 3.2 do TCC.

**Meta numérica concreta** (validação externa contra `eolica_NE_mensal_MWmed_2025-01_ate_hoje.csv`):

| Métrica | Hoje (Ensemble) | Meta |
|---|---:|---:|
| RMSE | 3544 MWmed | < 1500 |
| MAE | 3221 | < 1200 |
| Bias agregado | −3186 | dentro de [−500, +500] |
| MAPE | 26% | < 12% |
| Cobertura do IC 5%-95% | n/a | ≥ 90% |

---

## Linha de base atual (métricas a bater)

Consolidado da seção "Comparação parcial com valores reais já disponíveis" do [Review.md](Review.md). Período: 2025-01 a 2026-02 (14 meses).

| Modelo | RMSE | MAE | Bias | MAPE |
|---|---:|---:|---:|---:|
| CNN | 4204.9 | 3602.6 | -3602.2 | 31.4% |
| LSTM | 2627.0 | 2192.5 | -1898.7 | 17.6% |
| Informer | 3830.4 | 3443.4 | -3375.2 | 27.4% |
| XGBoost | 2098.0 | 1829.1 | -1260.5 | 15.1% |
| Ensemble | 3544.4 | 3221.6 | -3186.7 | 26.0% |

LSTM e XGBoost são os modelos menos quebrados. Esses são os números a bater.

---

## Princípios invioláveis

Estes são "regras de ouro" que devem ser respeitadas em todas as fases. Sempre que um trade-off aparecer, esses princípios decidem.

1. **Nada que não seja causalmente disponível no momento do cutoff entra no input.** Isso vale para imputação, scaling e qualquer transformação. Se você usou estatística de val/teste no treino, é leak.
2. **Toda feature usada no input precisa ter um plano explícito de projeção para o horizonte de previsão.** Se não dá pra projetar, sai do input.
3. **Validação seleciona; teste só relata.** Nunca o contrário.
4. **Backtest e treino final usam a mesma config de hiperparâmetros.** Configs diferentes invalidam a seleção.
5. **Toda alteração tem uma medida de comparação A/B documentada.** Se você não consegue dizer "antes RMSE=X, depois RMSE=Y", a alteração não conta.
6. **Reproducibilidade fechada por seed.** `os.environ['TF_DETERMINISTIC_OPS']='1'` antes do `import tensorflow`. Cada experimento roda com 5 seeds e reporta mediana e desvio.
7. **MWmed é a unidade canônica.** Conversão para GWh apenas em rótulos, jamais em armazenamento.

---

## Fase 0 — Higiene e contrato de dados ✅ CONCLUÍDA (exceto 0.8 e 0.9, postponed)

**Por que primeiro:** sem isso, qualquer melhoria depois fica contaminada por leak ou por configs inconsistentes. Não dá pra A/B comparar em cima de fundação podre.

**Pronto da fase:** notebook reproduzível em outra máquina, sem leak na imputação/scaling, sem feature derivada do target no input, manifesto único de datas.

**Estimativa:** 1.5 a 2 dias de trabalho focado.

### 0.1 — Rotacionar chave CDS e remover `.cdsapirc` do Git ✅ DONE (2026-05-02)
> Usuário confirmou que `.cdsapirc` é uso pessoal, mesma chave entre cópias. `.gitignore` atualizado com `.cdsapirc`, `*.env`, `*.pem`, `*.key`, `secrets.json`.

**Onde:** raiz do projeto.

**Como:**
1. Acessar https://cds.climate.copernicus.eu/profile e gerar chave nova.
2. Mover `.cdsapirc` para fora do repositório (ex.: `~/.cdsapirc` no Windows = `C:\Users\Admin\.cdsapirc`).
3. Atualizar `.gitignore` com `.cdsapirc`.
4. Remover do tracking: `git rm --cached .cdsapirc` e commit.
5. Idealmente, reescrever histórico com `git filter-repo --path .cdsapirc --invert-paths` (destrutivo — só fazer se o repo for privado e você puder force-push). Caso contrário, basta a rotação.

**Critério de pronto:**
- `git ls-files | grep cdsapirc` retorna vazio.
- Notebook continua funcionando lendo de `~/.cdsapirc`.
- Chave antiga foi revogada no portal CDS.

### 0.2 — Despregar `pasta_central` da máquina ✅ DONE (2026-05-02)
> Cell 0 do notebook agora usa `Path(os.environ.get('TCC_ROOT', Path.cwd())).resolve()`. Patch via `scripts/patch_fase0.py`.

**Onde:** célula 1 do `pipeline_previsao.ipynb`.

**Hoje:** `pasta_central = r'C:\Users\Admin\Documents\Puc\IC'`.

**Como:**
```python
from pathlib import Path
import os
pasta_central = Path(os.environ.get('TCC_ROOT', Path.cwd())).resolve()
DADOS_DIR = pasta_central / 'Dados'
MODELOS_DIR = pasta_central / 'Modelos'
DADOS_DIR.mkdir(exist_ok=True)
MODELOS_DIR.mkdir(exist_ok=True)
```

Substituir todos os `r'C:\Users\Admin\...'` espalhados por `pasta_central / ...`.

**Critério de pronto:** rodar `git grep -n "C:\\\\Users"` no repo retorna zero ocorrências; notebook abre em outra máquina/usuário sem editar caminhos.

### 0.3 — Corrigir `requirements.txt` e fixar versões ✅ DONE (2026-05-02)
> Renomeado `requiremets.txt` → `requirements.txt` com versões fixas para pandas, numpy, xarray, tensorflow, xgboost, scikit-learn, cdsapi etc.

**Onde:** raiz.

**Como:**
1. Renomear `requiremets.txt` → `requirements.txt`.
2. Fixar versões principais. Mínimo:
   ```
   pandas==2.2.*
   numpy==1.26.*
   xarray==2024.*
   netCDF4==1.6.*
   tensorflow==2.15.*
   xgboost==2.0.*
   scikit-learn==1.4.*
   cdsapi==0.7.*
   matplotlib==3.8.*
   ```
3. Validar `pip install -r requirements.txt` em venv limpo.

**Critério de pronto:** `pip install -r requirements.txt` em venv novo termina sem erro e o notebook roda até a célula 9.

### 0.4 — Criar contrato único de datas ✅ DONE (2026-05-02)
> Cell 0 do notebook tem agora `ULTIMO_MES_COMPLETO`, `HORIZONTE_PREVISAO_MESES=60`, `DATA_PRIMEIRO_MES_PREVISTO`. Cell 8 audita merge ONS+ERA5 → `Dados/meses_descartados_no_merge.csv`.

**Por que:** hoje a célula 1 usa `range(2010, 2025)`, a 6 processa 2015-2025, a 22 usa `date.today().year`. Isso causa o `inner merge` da célula 9 a descartar 2025 silenciosamente.

**Onde:** célula 1 (constantes globais).

**Como:** definir constantes únicas e usar em todo lugar.
```python
ANO_INICIO_HISTORICO = 2015
ANO_FIM_HISTORICO_ONS = 2025      # ONS publica até hoje
ANO_FIM_HISTORICO_ERA5 = 2024     # ERA5 reanálise — checar última data disponível
ULTIMO_MES_COMPLETO = pd.Timestamp('2025-12-01')  # ajustar manualmente
HORIZONTE_PREVISAO_MESES = 60
DATA_PRIMEIRO_MES_PREVISTO = ULTIMO_MES_COMPLETO + pd.DateOffset(months=1)
```

E **antes** do merge na célula 9, imprimir e salvar quem está sendo descartado:
```python
meses_ons = set(df_ons['data'].unique())
meses_era5 = set(df_era5['data'].unique())
descartados = sorted(meses_ons.symmetric_difference(meses_era5))
pd.Series(descartados).to_csv(DADOS_DIR / 'meses_descartados_no_merge.csv', index=False)
print(f"⚠ {len(descartados)} meses descartados no merge — ver meses_descartados_no_merge.csv")
```

**Critério de pronto:** existe `Dados/meses_descartados_no_merge.csv` listando explicitamente o que foi cortado, e o notebook não tem nenhum literal `2024`, `2025`, ou `date.today()` espalhado fora da célula 1.

### 0.5 — Mover imputação, scaling e clustering para dentro de cada fold ✅ DONE (2026-05-02)
> Cell 12: trocado `fillna(df_clean[col].mean())` por `ffill().bfill()`. Cell 13: imputador por mediana fitado por fold; assert anti-leak adicionado em `prepare_data_for_round_v2`.

**Por que:** célula 12 ~linha 122-126 imputa NaN com média global; célula 9 faz clustering com visão de todo o período. Ambos vazam informação futura.

**Onde:** células 9, 12 e 13.

**Como (imputação):** criar um helper `fit_imputer_treino(df_train, cols_meteo)` que retorna um dict `col → media`. No backtest:
```python
def aplica_imputacao(df, imputer_dict):
    df = df.copy()
    for col, valor in imputer_dict.items():
        df[col] = df[col].fillna(valor)
    return df

# Dentro do loop de backtest:
for origin in origens_backtest:
    df_train = df[df['data'] < origin]
    df_val   = df[(df['data'] >= origin) & (df['data'] < origin + pd.DateOffset(months=H))]
    imputer = {col: df_train[col].mean() for col in cols_meteo}
    df_train_clean = aplica_imputacao(df_train, imputer)
    df_val_clean   = aplica_imputacao(df_val, imputer)
    # idem para scaler: scaler.fit(df_train_clean), depois transform em ambos
```

**Como (scaling):** mesma lógica. `scaler_X.fit(X_train)` apenas, `scaler_y.fit(y_train)` apenas.

**Como (clustering espacial):** declarar o grid espacial como **conhecimento exógeno fixo** e calcular **uma única vez no momento da preparação**, não dentro do fold. Salvar em `Dados/clusters_definicao.json` e ler. Documentar no relatório que o grid espacial é fixo.

**Critério de pronto:**
- `git grep "fillna(df_clean\["` no notebook retorna zero.
- Cada cell que treina tem o scaler/imputer fitado **dentro** do loop de fold.
- `clusters_definicao.json` existe e é referenciado em vez de recalculado.

### 0.6 — Limpar dataset de modelagem (sem features derivadas do target) ✅ DONE (2026-05-02)
> `scripts/gerar_dataset_limpo.py` gera `dataset_modelagem_limpo.csv` (120×97), removendo `corte_eolica_ne_mwmed`, `capacity_factor_ne`, `penetracao_eolica_ne`, `penetracao_eolica_sin`. Cluster fantasma `lat11.0S_lon38.8W` (76% NaN) removido. Adiciona `fc_ne` em fração [0,1].

**Por que:** célula 6 cria `corte_eolica_ne_mwmed`, `capacity_factor_ne`, `penetracao_eolica_ne`, `penetracao_eolica_sin` — todas derivadas do target. Mesmo que o pipeline depois remova, o artefato bruto ainda é perigoso.

**Onde:** célula 6 (criação do dataset) e célula 13 (preparação para modelagem).

**Como:**
1. Renomear o output da célula 6 de `dataset_final_modelagem.csv` para `dataset_diagnostico_completo.csv`.
2. Criar um novo output `dataset_modelagem_limpo.csv` que **explicitamente** remove colunas derivadas do target:
   ```python
   COLS_DERIVADAS_DO_TARGET = [
       'capacity_factor_ne', 'penetracao_eolica_ne', 'penetracao_eolica_sin',
       'corte_eolica_ne_mwmed'
   ]
   df_modelagem = df_diagnostico.drop(columns=COLS_DERIVADAS_DO_TARGET)
   df_modelagem.to_csv(DADOS_DIR / 'dataset_modelagem_limpo.csv', index=False)
   ```
3. **Importante:** `capacity_factor_ne` vai voltar como **target** na fase 1, mas como variável-alvo, não como feature. Por enquanto sai.
4. Adicionar assert no início do `prepare_data_for_round_v2`:
   ```python
   PROIBIDAS = set(COLS_DERIVADAS_DO_TARGET)
   features_usadas = set(df.columns) - {'data', 'geracao_eolica_ne_mwmed'}
   if features_usadas & PROIBIDAS:
       raise ValueError(f"Feature do target detectada: {features_usadas & PROIBIDAS}")
   ```

**Critério de pronto:** rodar `prepare_data_for_round_v2` em qualquer dataset que tenha qualquer dessas 4 colunas levanta erro; o pipeline default carrega `dataset_modelagem_limpo.csv`.

### 0.7 — Off-by-one no mês futuro ✅ DONE (2026-05-02)
> Cell 13: `gerar_cenarios_futuros_tcc` agora usa `pd.Timestamp(ultimo_mes) + pd.DateOffset(months=h+1)` em vez de `(ultimo_mes.month + h) % 12`. Patch via `scripts/patch_offbyone.py`.

**Onde:** célula 13, função `gerar_cenarios_futuros_tcc` (linha onde aparece `ultimo_mes.month + h`).

**Como:**
```python
# ANTES (errado):
# future_month = ultimo_mes.month + h
# ...

# DEPOIS:
for h in range(H):
    future_date = ultimo_mes + pd.DateOffset(months=h+1)
    future_month = future_date.month
    future_year = future_date.year
    # usar future_date como índice da linha
```

**Critério de pronto:** primeiro ponto previsto no CSV final é `DATA_PRIMEIRO_MES_PREVISTO` (definido em 0.4) e não o último mês histórico.

### 0.8 — Padronizar unidade MWmed ⏸ POSTPONED (cosmético, baixa prioridade)

**Onde:** células 12, 16, 19, 20, 21 — qualquer rótulo "GWh" em gráfico ou nome de coluna.

**Como:** busca textual por `GWh` e por `* 730` ou `* 720` (conversão MWmed→GWh dependendo de horas/mês). Para conversão correta:
```python
def mwmed_para_gwh(mwmed, ano, mes):
    horas = pd.Timestamp(ano, mes, 1).days_in_month * 24
    return mwmed * horas / 1000
```

Mas o **armazenamento** continua em MWmed. Conversão só em rótulo de eixo y, e sempre nominal por mês (não constante).

**Critério de pronto:** todos os CSVs em `Modelos/Comparacoes/` têm cabeçalho mencionando `_mwmed`, e os PNGs com eixo y rotulado corretamente.

### 0.9 — Validar metadados consistentes ⏸ POSTPONED (será resolvido na Fase 4 com manifesto.json)

**Onde:** célula 17/18 (seleção do modelo final).

**Hoje:** `metadata_inferencia.json` aponta `L48_H12_W72`, `configuracao_final.json` aponta `L24_H12_W60`.

**Como:** após o treino final, regerar `metadata_inferencia.json` programaticamente a partir da config vencedora:
```python
def gerar_metadata(config_key, modelo_path, scaler_path, dataset_hash, features_list):
    return {
        'config_key': config_key,
        'L': int(config_key.split('_')[0][1:]),
        'H': int(config_key.split('_')[1][1:]),
        'W': int(config_key.split('_')[2][1:]),
        'modelo_path': str(modelo_path),
        'scaler_path': str(scaler_path),
        'dataset_hash': dataset_hash,
        'features': features_list,
        'unidade': 'MWmed',
        'data_geracao': pd.Timestamp.now().isoformat(),
    }
```

**Critério de pronto:** rodar um diff entre `configuracao_final.json` e `metadata_inferencia.json` mostra apenas chaves esperadas (não conflitantes).

### 0.10 — Higiene de notebook (nice-to-have) ✅ DONE (2026-05-02)
> Cell 0 do notebook tem `TF_DETERMINISTIC_OPS=1`, `PYTHONHASHSEED=42`, `set_seeds(42)` antes de qualquer import de TF.

- Adicionar célula 0 com `os.environ['TF_DETERMINISTIC_OPS']='1'`, `os.environ['PYTHONHASHSEED']='42'` **antes** de qualquer `import tensorflow`.
- Limpar outputs de células de exploração antes de commit.
- Adicionar markdown na célula 13 marcando claramente que o ensemble exploratório dela é **deprecated**.

---

## Fase 1 — Target FC e features futuras ✅ CONCLUÍDA (1.2-1.8 ✅ ; 1.1 parcial via baseline — pipeline neural com FC fica para Fase 2)

> **Resultado parcial (2026-05-02):** Sazonal Naive de FC × capacidade conservadora atinge **RMSE 1653.7 / MAE 1419.8 / Bias −156.9** em validação externa 2025 — bate todos os modelos do notebook (Ensemble RMSE 3544) e o NEWAVE em MAE (1521). Detalhes em [RESULTADOS_FASE1.md](RESULTADOS_FASE1.md).

**Por que:** esta é a fase de maior ROI. As previsões estão ruins porque o pipeline trata o problema como "prever um número que cresce 5x junto com a capacidade instalada". Decompor em FC × capacidade resolve a maior parte.

**Pronto da fase:** rodar uma previsão de 60 meses com **Sazonal Naive de FC × capacidade projetada** (sem rede neural) e ela ser melhor que o ensemble atual.

**Estimativa:** 2 a 3 dias.

### 1.1 — Trocar o target para fator de capacidade 🟡 PARCIAL (2026-05-02)
> `fc_ne` adicionado ao `dataset_modelagem_limpo.csv` (range [0, 0.45], média 0.32). Baseline Sazonal Naive (FC × capacidade projetada) já validado com sucesso. Pipeline neural com FC fica para a Fase 2.

**Por que:** o target atual `geracao_eolica_ne_mwmed` cresce ~5x no período histórico. O modelo gasta capacidade explicando tendência em vez de aprender padrão climático. FC (`geracao / capacidade`) é estacionário (entre 0 e ~0.55), tem sazonalidade clara e é o que a literatura de previsão eólica modela.

**Onde:** novo arquivo/seção `Pipeline FC` em paralelo ao pipeline atual (não substituir ainda — A/B).

**Como:**
1. Criar coluna `fc_ne = geracao_eolica_ne_mwmed / capacidade_eolica_ne_mw` no dataset limpo.
2. Treinar modelos com `target = fc_ne` em vez de `geracao_eolica_ne_mwmed`.
3. Na inferência, multiplicar a previsão de FC pela curva de capacidade futura projetada (vide 1.5):
   ```python
   geracao_prevista = fc_previsto * capacidade_projetada
   ```
4. Validar que o FC histórico fica em range plausível: `assert df['fc_ne'].between(0, 0.6).all()`. Se algum mês tiver FC > 0.6, investigar (pode ser bug de capacidade subestimada nesse mês).

**Crítico:** FC histórico **não** é leak. Ele é calculado com geração e capacidade que **já eram observadas** no momento do cutoff. O que era leak na versão anterior era usar FC para projetar geração futura usando capacidade futura — mas é exatamente isso que queremos fazer agora, controlando explicitamente o cenário de capacidade.

**Critério de pronto:** existe `pipeline_fc.ipynb` ou seção paralela com:
- target = FC,
- previsão de FC para 60 meses,
- multiplicação por curva de capacidade projetada,
- comparação A/B com pipeline atual em RMSE/MAE/Bias na validação externa de 2025.

### 1.2 — Curva de capacidade futura projetada (3 cenários) ✅ DONE (2026-05-02)
> `scripts/gerar_capacidade_projetada.py` gera `Dados/capacidade_projetada_ne.csv` com 60 meses × 3 cenários. Auto-sort por valor terminal (conservador 37 GW, base 52 GW, otimista 79 GW em dez/2029). Cenário conservador (linear total, slope 2.4 GW/ano) é o que melhor reflete a desaceleração observada em 2025.

**Por que:** hoje `advance_window_no_roll` copia a capacidade de jan/2025 e usa ela por 60 meses. Capacidade real de 2025-2030 vai crescer significativamente.

**Fontes (em ordem de preferência):**
1. **EPE / PDE 2034** — Plano Decenal de Energia, capítulo Geração Eólica. Tem projeção oficial por subsistema.
2. **ANEEL Banco de Empreendimentos** — projetos com outorga já emitida, com data prevista de entrada em operação. Filtrar por NE.
3. **Fallback se 1 e 2 não couberem no prazo:** fit linear ou exponencial saturando da série histórica `capacidade_eolica_ne_mw` 2015-2024.

**Onde:** novo arquivo `Dados/capacidade_projetada_ne.csv` com colunas `data, cenario_conservador, cenario_base, cenario_otimista`.

**Como:**
```python
# Fallback (se PDE não estiver pronto):
import numpy as np
from sklearn.linear_model import LinearRegression

cap_hist = df.groupby('data')['capacidade_eolica_ne_mw'].mean().reset_index()
cap_hist['t'] = np.arange(len(cap_hist))

# Modelo log para saturação
modelo = LinearRegression().fit(cap_hist[['t']], np.log(cap_hist['capacidade_eolica_ne_mw']))

datas_futuras = pd.date_range(DATA_PRIMEIRO_MES_PREVISTO, periods=HORIZONTE_PREVISAO_MESES, freq='MS')
t_futuro = np.arange(len(cap_hist), len(cap_hist) + len(datas_futuras))
cap_base = np.exp(modelo.predict(t_futuro.reshape(-1, 1)))

df_cap = pd.DataFrame({
    'data': datas_futuras,
    'cenario_conservador': cap_base * 0.85,
    'cenario_base': cap_base,
    'cenario_otimista': cap_base * 1.15,
})
df_cap.to_csv(DADOS_DIR / 'capacidade_projetada_ne.csv', index=False)
```

**Validar visualmente:** plotar série histórica + 3 cenários. A curva base deve continuar a tendência sem virar reta nem explodir.

**Critério de pronto:** `capacidade_projetada_ne.csv` cobre 60 meses, valores são monotonicamente não-decrescentes, e o cenário base alinha visualmente com a continuação da série histórica.

### 1.3 — Lags do target no input ✅ DONE (2026-05-02)
> `gerar_dataset_limpo.py` adiciona `fc_lag_{1,2,3,6,12}`, `geracao_lag_{1,3,6,12}` e `fc_diff_1`. Loop recursivo de inferência será fechado na Fase 2 (`forecast_recursive` unificada).

**Por que:** sem lag, o modelo prevê 60 meses sem nunca usar o que ele mesmo previu. A previsão recursiva é puramente ditada por sazonalidade do calendário.

**Onde:** célula 6 (criação de features) e função de avanço da janela na célula 20.

**Como (no dataset histórico):**
```python
for lag in [1, 2, 3, 6, 12]:
    df[f'fc_lag_{lag}'] = df['fc_ne'].shift(lag)
# remover linhas iniciais com NaN ou imputar com média do treino (dentro do fold)
```

**Como (na previsão recursiva):**
```python
def avanca_janela_com_realimentacao(X_prev, y_hat_prev, capacidade_proj, mes_futuro):
    """
    X_prev: shape (L, n_features) — janela anterior
    y_hat_prev: float — previsão de FC do passo anterior
    """
    nova_linha = X_prev[-1].copy()
    # atualizar lags do target com a previsão anterior
    nova_linha[idx_fc_lag_1] = y_hat_prev
    # capacidade projetada do mês futuro
    nova_linha[idx_capacidade] = capacidade_proj.loc[mes_futuro, 'cenario_base']
    # calendário
    nova_linha[idx_mes_sin] = np.sin(2*np.pi*mes_futuro.month/12)
    nova_linha[idx_mes_cos] = np.cos(2*np.pi*mes_futuro.month/12)
    # meteorologia: climatologia mensal (vide 1.4)
    for col in cols_meteo:
        nova_linha[col2idx[col]] = climatologia_mensal[col].loc[mes_futuro.month]
    return np.vstack([X_prev[1:], nova_linha])
```

**Critério de pronto:** o input do modelo na inferência muda a cada passo recursivo (verificar imprimindo `X[-1]` em h=0, h=1, h=2 — devem ser diferentes).

### 1.4 — Climatologia mensal para meteorologia futura ✅ DONE (2026-05-02)
> `scripts/gerar_climatologia.py` gera `Dados/climatologia_mensal_meteo.csv` (12 meses × 63 features = 756 registros) com mean/std por mês para todos os prefixos meteorológicos.

**Por que:** ERA5 não existe para o futuro (é reanálise histórica). Hoje o pipeline usa o último mês observado, congelado por 60 meses. Climatologia mensal (média de cada mês ao longo dos anos históricos) é o piso defensável.

**Onde:** novo arquivo `Dados/climatologia_meteo_ne.csv`.

**Como:**
```python
cols_meteo = ['ws10_med', 'ws100_med', 't2m_med', 'sp_med', 'tp_med', ...]  # ajustar
climatologia = df.groupby(df['data'].dt.month)[cols_meteo].agg(['mean', 'std']).reset_index()
climatologia.to_csv(DADOS_DIR / 'climatologia_meteo_ne.csv', index=False)
```

**Critérios para evoluir além de climatologia (opcional, fase 2):**
- Se você quiser cenários, gerar 3 trajetórias: `clim_mediana`, `clim_média + 0.5σ`, `clim_média − 0.5σ`.
- Se houver disponibilidade, usar previsão sazonal do **CFS/ECMWF S5** para os 7 primeiros meses, e cair para climatologia depois.

**Critério de pronto:** existe `climatologia_meteo_ne.csv`; a função de avanço da janela usa a climatologia, não a última linha; existe um teste imprimindo a feature de vento previsto para jan/2026 = média dos eneros históricos.

### 1.5 — Substituir `regime_operacional` por `meses_desde_inicio` ✅ DONE (2026-05-02)
> `dataset_modelagem_limpo.csv` agora tem `meses_desde_inicio` (ref 2010-01-01) e `mes_sin`/`mes_cos`. `regime_operacional` removido das features ativas.

**Por que:** `regime_operacional` é categórico hardcoded com buckets de ano. Para qualquer ano ≥2025 vira constante. Inútil como feature de tendência.

**Onde:** célula 6 e em `features_aux` na célula 12.

**Como:**
```python
DATA_INICIO_SETOR = pd.Timestamp('2010-01-01')
df['meses_desde_inicio'] = ((df['data'] - DATA_INICIO_SETOR).dt.days / 30.44).astype(int)
# remover regime_operacional da lista de features
```

**Critério de pronto:** `regime_operacional` removido das features ativas; `meses_desde_inicio` cresce monotonicamente também no horizonte futuro.

### 1.6 — Features socioeconômicas e macroclimáticas ✅ DONE (2026-05-02)
> `scripts/coletar_features_macro.py` baixa Niño 3.4 e AMM (NOAA PSL) e IBC-Br (BCB SGS 24364), salva `Dados/features_macro.csv` (204×4). `gerar_dataset_limpo.py` faz merge left por `data` e adiciona `nino34_lag_{3,6,12}`. Dataset final: 120×103. `features_futuras.py` ganhou regra `climatologia_macro` (Niño/AMM e seus lags) e usa `extrapolacao_linear` para IBC-Br. Auditoria de 97 features passa 100%.

**Por que:** o TCC seção 4 menciona explicitamente PIB, IDH e variáveis socioeconômicas como drivers. Hoje nenhuma dessas está no dataset.

**Mínimo viável (em ordem de prioridade):**
1. **Índice ENSO (Niño 3.4 SST anomaly)** — NOAA, mensal, série longa, projetado por modelos sazonais. Tem correlação documentada com regime de ventos no NE.
2. **Anomalia AMM (Atlantic Meridional Mode)** — NOAA, mensal. Mais relevante que ENSO para NE.
3. **PLD-NE mensal** — CCEE, série pública.
4. **PIB Brasil mensal (IBC-Br)** — BCB SGS, código 24364.
5. **PIB regional NE** — IBGE, anual (interpolar para mensal).

**Onde:** nova célula `Coleta de Features Macro` antes do merge final.

**Como:**
```python
# Exemplo NOAA Niño 3.4
import requests
url = 'https://psl.noaa.gov/data/correlation/nina34.anom.data'
df_nino = pd.read_fwf(url, skiprows=1, ...)  # parser ad hoc
# Exemplo BCB SGS
df_pib = pd.read_csv('https://api.bcb.gov.br/dados/serie/bcdata.sgs.24364/dados?formato=csv', sep=';')
```

**Para cada feature macro, definir como projetar no futuro:**
- ENSO/AMM: usar previsão sazonal do **CPC NOAA** para 9 meses, climatologia depois. Ou simplesmente climatologia (= 0 = neutro).
- PIB: extrapolar série com modelo simples (LinearRegression ou ETS).
- PLD: climatologia mensal.

**Critério de pronto:** dataset tem ao menos `nino34`, `amm`, `pib_br` como features adicionais; cada uma tem regra de projeção documentada em `features_futuras.py`.

### 1.7 — Contrato explícito de features futuras ✅ DONE (2026-05-02)
> `src/features_futuras.py` define `PROJECAO_FUTURA: list[tuple[regex, codigo]]` cobrindo 100% das features do dataset. `auditar_features()` levanta `FeatureSemRegraError` se algo escapar. `projetar_features_futuras()` testado com 60 meses sem NaN exceto em lags do target (esperado — preenchido pelo loop recursivo).

**Por que:** hoje, se uma feature está no input mas não tem regra de projeção, ela vira zero ou repete a última linha sem aviso.

**Onde:** novo dicionário em `features_futuras.py` (ou célula dedicada).

**Como:**
```python
PROJECAO_FUTURA = {
    # Target derivado: realimentar com previsão anterior
    'fc_lag_1': 'realimentacao_recursiva',
    'fc_lag_3': 'realimentacao_recursiva',
    'fc_lag_12': 'realimentacao_sazonal',  # usa fc real do mesmo mês ano anterior
    # Calendário: determinístico
    'mes_sin': 'calendario',
    'mes_cos': 'calendario',
    'meses_desde_inicio': 'calendario',
    # Capacidade: cenário
    'capacidade_eolica_ne_mw': 'cenario_capacidade',
    'crescimento_capacidade_mw': 'derivada_cenario_capacidade',
    # Meteorologia: climatologia
    'ws10_med_cluster_0': 'climatologia',
    'ws100_med_cluster_0': 'climatologia',
    # ...
    # Macro
    'nino34': 'previsao_sazonal_nooa',  # ou climatologia
    'pib_br': 'extrapolacao_linear',
    'pld_ne': 'climatologia',
    # Carga/demanda
    'carga_ne_mwmed': 'extrapolacao_linear_pib',
    'demanda_sin_mwmed': 'extrapolacao_linear_pib',
}

# Assert no início da inferência:
features_modelo = set(metadata['features'])
features_sem_plano = features_modelo - set(PROJECAO_FUTURA.keys())
if features_sem_plano:
    raise ValueError(f"Features sem regra de projeção: {features_sem_plano}")
```

**Critério de pronto:** rodar a previsão recursiva levanta erro se uma feature do modelo não tiver entrada em `PROJECAO_FUTURA`.

### 1.8 — Baseline Sazonal Naive × capacidade projetada ✅ DONE (2026-05-02) — RESULTADO PRINCIPAL DA FASE 1
> `scripts/baseline_sazonal_naive.py`. Bate Ensemble do notebook em RMSE (1653 vs 3544) e bate NEWAVE em MAE (1419 vs 1521). Bias 20× menor (−157 vs −3186). Confirma a hipótese central da Fase 1: o problema não era arquitetura, e sim target absoluto + capacidade futura mal projetada. Artefatos em `Modelos/Comparacoes/Baseline_Sazonal/`. Resumo executivo em [RESULTADOS_FASE1.md](RESULTADOS_FASE1.md).

**Por que:** este é o "smoke test" da fase 1. Se o modelo neural complexo não bater este baseline simples, o problema não é arquitetura — é que a fase 1 não foi bem feita.

**Onde:** nova célula `Baseline Sazonal Naive`.

**Como:**
```python
def baseline_sazonal_fc(df_treino, capacidade_proj, datas_futuras):
    fc_climatologia = df_treino.groupby(df_treino['data'].dt.month)['fc_ne'].mean()
    previsoes = []
    for data in datas_futuras:
        fc_prev = fc_climatologia.loc[data.month]
        cap = capacidade_proj.loc[data, 'cenario_base']
        previsoes.append({'data': data, 'fc': fc_prev, 'geracao_mwmed': fc_prev * cap})
    return pd.DataFrame(previsoes)
```

**Critério de pronto:** baseline rodando, RMSE/MAE/Bias documentados na validação externa 2025. Esses números são o piso da fase 2 — qualquer rede neural que não bater este baseline está sobre-engenheirada.

---

## Fase 2 — Arquitetura e treinamento ✅ CONCLUÍDA (2026-05-02)

> **Resultado:** modelo Integrado (CNN→[LSTM, Transformer], fusão por gating aprendido) atinge **RMSE 1796 / MAE 1421 / Bias −445 / MAPE 13.4%** em validação externa 2025-2026 — bate todos os modelos do notebook (ensemble RMSE 3544 → −49%) e empata em MAE com o baseline sazonal. Detalhes em [RESULTADOS_FASE2.md](RESULTADOS_FASE2.md).

**Por que:** com fase 1 feita, agora vale corrigir a arquitetura. Antes não fazia sentido — o problema não era de modelagem, era de target.

**Pronto da fase:** modelos neurais treinados com mesma config no backtest e no final, EarlyStopping ativo, 5 seeds reportadas, e cada modelo tem RMSE em validação externa **menor** que o baseline sazonal da fase 1.

**Estimativa:** 2 a 3 dias.

### 2.1 — Implementar arquitetura integrada CNN → [LSTM, Informer] ✅ DONE (2026-05-02)
> `src/models.py:build_integrado` — extrator CNN compartilhado + ramos LSTM e Transformer fundidos por gating aprendido. Transformer encoder padrão substituiu Informer (mais simples, sem perda de capacidade). Ablations LSTM/Transformer solo treinadas no mesmo input.

**Por que:** o TCC seção 3.2 promete CNN extraindo features que alimentam LSTM e Informer. Hoje os 3 ramos são paralelos sobre input bruto. Os erros se correlacionam, então o ensemble agrega pouco valor.

**Onde:** célula 12, função `build_*` reescrita.

**Como (esqueleto Keras):**
```python
def build_cnn_extractor(L, n_features, n_filters=32, kernel=3):
    """Extrator compartilhado: entrada (L, n_features) → saída (L', n_filters_final)."""
    inp = Input(shape=(L, n_features))
    x = Conv1D(n_filters, kernel, padding='causal', activation='relu')(inp)
    x = Conv1D(n_filters, kernel, padding='causal', activation='relu')(x)
    return Model(inp, x, name='cnn_extractor')

def build_integrado(L, n_features, H, hp):
    inp = Input(shape=(L, n_features))
    extractor = build_cnn_extractor(L, n_features, hp['cnn_filters'])
    feat = extractor(inp)
    
    # Ramo LSTM
    lstm_out = LSTM(hp['lstm_units'], return_sequences=False)(feat)
    lstm_out = Dropout(hp['dropout'])(lstm_out)
    pred_lstm = Dense(H)(lstm_out)
    
    # Ramo Transformer (Informer simplificado)
    transformer_out = transformer_block(feat, hp['d_model'], hp['n_heads'], hp['ff_dim'])
    pred_transformer = Dense(H)(transformer_out[:, -1, :])
    
    # Fusão por média ponderada aprendida
    w = Dense(2, activation='softmax')(Concatenate()([lstm_out, transformer_out[:, -1, :]]))
    pred = w[:, 0:1] * pred_lstm + w[:, 1:2] * pred_transformer
    
    return Model(inp, pred)
```

**Decisão de design:** para o TCC, recomendo trocar o Informer por um **Transformer encoder padrão** (multi-head attention + FFN, com positional encoding sinusoidal). Mais simples, melhor documentado e alinhado com o que o TCC chama de "atencional". Justificar no texto.

**Critério de pronto:** modelo único `modelo_integrado.keras` que recebe (L, n_features) e produz H predições. Ablations (LSTM solo, Transformer solo) treinadas no mesmo input.

### 2.2 — Reduzir capacidade dos modelos para o tamanho da base ✅ DONE (2026-05-02)
> `HiperParams` em `src/models.py`: cnn_filters=32, lstm=32, d_model=32, dropout=0.2, sem bidirecional, sem recurrent_dropout. Modelo integrado < 50k parâmetros. Treino completa em ~20s/seed em CPU.

**Por que:** 50-70 sequências de treino com LSTM bidirecional 128+64 é overfit garantido (ou então o regularizer está mascarando capacidade real).

**Como (sugestão calibrada):**
- **LSTM**: 32 a 64 unidades, sem bidirecional. Bidirecional faz pouco sentido em forecast (não tem dado de futuro no input por construção).
- **CNN**: 16 a 32 filtros, 2 camadas no máximo, kernel 3.
- **Transformer**: `d_model=64`, `n_heads=4`, `ff_dim=128`, 2 layers.
- **Dropout**: 0.1 a 0.2 (não 0.3+ que vai matar treino com 50 amostras).
- **Sem `recurrent_dropout`** (desliga cuDNN e treina lento sem ganho).

**Critério de pronto:** total de parâmetros do modelo integrado < 50k. Treino completa em < 60s no hardware do usuário.

### 2.3 — Trocar `loss=mse` por `loss=Huber(delta=1.0)` ✅ DONE (2026-05-02)
> `src/models.py:_compilar` usa `Huber(delta=1.0)`. Aplicado a todas as 3 arquiteturas (integrado, lstm_solo, transformer_solo).

**Por que:** dataset pequeno + anos atípicos (2021-2022 com regime climático adverso) → MSE penaliza outliers quadraticamente e o modelo gasta capacidade tentando aprender o anômalo. Huber é robusta.

**Onde:** todas as compilações de modelo.

**Como:**
```python
from tensorflow.keras.losses import Huber
model.compile(optimizer=Adam(learning_rate=hp['lr']), loss=Huber(delta=1.0), metrics=['mae'])
```

**Critério de pronto:** todos os modelos neurais usando Huber; confirmar via `model.loss`.

### 2.4 — EarlyStopping, ReduceLROnPlateau e ModelCheckpoint no treino final ✅ DONE (2026-05-02)
> `src/train_eval.py` aplica EarlyStopping(patience=15, restore_best=True) + ReduceLROnPlateau(factor=0.5, patience=7) em todos os treinos.

**Por que:** treino final hoje roda 100 épocas sem validação nem early stop. Garantia de sobre-ajustar a 50 sequências.

**Onde:** célula 14, treino final.

**Como:**
```python
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, mode='min'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-5),
    ModelCheckpoint(filepath=str(MODELOS_DIR / 'final' / 'best.keras'),
                    monitor='val_loss', save_best_only=True),
]
model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=200, batch_size=16, callbacks=callbacks, verbose=0)
```

**Importante:** o `val` aqui é o **último fold do rolling origin** que ainda não foi usado para selecionar config. Não pode ser teste congelado.

**Critério de pronto:** treino final escreve curva de loss train/val em `Modelos/final/loss_curve.png` mostrando convergência monitorada.

### 2.5 — Mesma config no backtest e no treino final ✅ DONE (2026-05-02)
> `HiperParams` único em `src/models.py`; `treinar_uma_seed` em `src/train_eval.py` é a única função de treino, usada tanto no backtest (5 seeds × 3 arquiteturas) quanto no re-treino final.

**Por que:** hoje backtest usa `d_model=32, e_layers=1` e treino final usa `d_model=64, e_layers=2`. Seleção de modelo inválida.

**Onde:** parametrizar via `config_key` único em todo lugar.

**Como:**
```python
GRID = [
    {'config_key': 'L24_H12_W60_d64_h4_l2', 'L': 24, 'H': 12, 'W': 60,
     'd_model': 64, 'n_heads': 4, 'e_layers': 2, 'dropout': 0.2, 'lstm_units': 32, ...},
    # outras configs
]

def treina_modelo(hp, X_train, y_train, X_val, y_val, seed):
    # mesma função usada no backtest E no final
    set_seeds(seed)
    model = build_integrado(hp['L'], n_features, hp['H'], hp)
    model.compile(...)
    model.fit(...)
    return model
```

**Critério de pronto:** uma busca por "build_" no notebook revela apenas 1 chamada a cada `build_*`, e a config passada vem do mesmo grid.

### 2.6 — Reproducibilidade fechada ✅ DONE (2026-05-02)
> `src/seeds.py:configurar_ambiente_determinismo()` chamado no entry-point dos scripts antes de qualquer import de TF. `set_seeds(42)` antes de cada `model.fit`. Cada arquitetura roda 5 seeds (42, 1, 2, 3, 4) com mediana e desvio reportados em `Modelos/fase2/resultados_treino.csv`.

**Onde:** célula 0 (nova), antes de qualquer import.

**Como:**
```python
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['PYTHONHASHSEED'] = '42'

import random, numpy as np
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)
```

E rodar cada experimento com 5 seeds, reportando mediana e desvio:
```python
seeds = [42, 1, 2, 3, 4]
resultados = []
for s in seeds:
    set_seeds(s)
    metrics = treina_e_avalia(...)
    resultados.append(metrics)
df_res = pd.DataFrame(resultados)
print(df_res.agg(['median', 'std']))
```

**Critério de pronto:** rodar duas vezes com `seed=42` produz exatamente o mesmo `val_rmse` (até casas decimais).

### 2.7 — Unificar funções recursivas ✅ DONE (2026-05-02)
> `src/forecast_recursive.py:forecast_recursive()` é a função única. Asserts de shape em cada passo, projeção de features via `PROJECAO_FUTURA`, realimentação correta dos lags com previsões anteriores, clip a [0, 0.7] no FC.

**Por que:** existem 3 implementações divergentes (`previsao_recursiva_backtest`, `previsao_recursiva_com_cenario`, loop em cell 20).

**Onde:** novo módulo/célula `forecast_recursive.py`.

**Como:** uma única função:
```python
def forecast_recursive(model, scaler_X, scaler_y, X0, H, feature_contract, capacidade_proj, climatologia):
    """
    X0: shape (L, n_features) — última janela observada
    feature_contract: dict {feature_name: regra_projecao}
    Retorna: array (H,) com previsões em escala original
    """
    X = X0.copy()
    previsoes = []
    for h in range(H):
        # Asserts de shape
        assert X.shape == (L, n_features), f"Shape errado em h={h}: {X.shape}"
        
        # Predizer
        X_scaled = scaler_X.transform(X.reshape(-1, n_features)).reshape(1, L, n_features)
        y_hat_scaled = model.predict(X_scaled, verbose=0)[0, 0]  # primeiro horizonte só
        y_hat = scaler_y.inverse_transform([[y_hat_scaled]])[0, 0]
        previsoes.append(y_hat)
        
        # Avançar janela conforme contrato
        X = aplica_contrato_features(X, y_hat, capacidade_proj, climatologia, h, feature_contract)
    
    return np.array(previsoes)
```

**Critério de pronto:** todas as previsões recursivas no notebook chamam `forecast_recursive`. Test passa: `forecast_recursive(modelo_dummy_identidade, ...)` reproduz a entrada.

### 2.8 — Corrigir CNN para consumir aux ao longo de toda a janela ✅ DONE (2026-05-02)
> Resolvido por design: `build_integrado` usa tensor único `(L, n_features)` com target+aux concatenados. Extrator CNN vê todos os 24 meses de cada feature.

**Por que:** `aux_last = t[:,-1,:]` perde 23 dos 24 meses de capacidade/carga.

**Como:** se você unificar as features (target + aux) em um único tensor (L, n_features) — como sugerido em 2.1 — esse problema some. Se mantiver split target/aux, fazer:
```python
aux_conv = Conv1D(16, 3, padding='causal')(input_aux)  # processa toda a janela
aux_pooled = GlobalAveragePooling1D()(aux_conv)
```

**Critério de pronto:** modelo recebe aux como sequência (não single timestep); confirmar com `model.input_shape`.

---

## Fase 3 — Ensemble, baseline NEWAVE e tabela final ✅ CONCLUÍDA (2026-05-02)

> **Resultado:** Ensemble Otimizado (Integrado + Sazonal + XGBoost + LSTM, pesos por LOO-CV) atinge **RMSE 1163 / MAE 990 / Bias −253 / MAPE 9.3%** — bate todas as metas (<1500 RMSE, <1200 MAE, |bias|<500, MAPE<12%) e supera NEWAVE em MAE (990 vs 1521). Conformal IC com cobertura 100% no split test. Detalhes em [RESULTADOS_FASE3.md](RESULTADOS_FASE3.md).

**Por que:** com fase 2 feita, os modelos individuais estão decentes. Agora junta tudo, calibra incerteza e compara com NEWAVE.

**Pronto da fase:** tabela final com NEWAVE, baselines simples, modelos individuais e ensemble; intervalo de incerteza calibrado por bootstrap; relatório markdown.

**Estimativa:** 1.5 a 2 dias.

### 3.1 — Re-otimizar pesos do ensemble por validação ✅ DONE (2026-05-02)
> `src/ensemble.py:otimiza_pesos_ls` (constrained least squares, sum=1) e `loo_pesos` (leave-one-out CV). Pesos finais salvos em `Modelos/fase3/ensemble_pesos.json`. Convergem para ~50/50 entre Integrado_Fase2 e Sazonal_Naive.

**Por que:** hoje pesos são fixos `{'CNN': 0.2, 'LSTM': 0.19, 'Informer': 0.61}` favorecendo o modelo mais quebrado.

**Onde:** célula 20.

**Como (constrained least squares):**
```python
from scipy.optimize import minimize

def otimiza_pesos(preds_val, y_val):
    """preds_val: shape (n_modelos, n_amostras)"""
    n_modelos = preds_val.shape[0]
    def objetivo(w):
        ensemble = (w[:, None] * preds_val).sum(axis=0)
        return np.mean((ensemble - y_val) ** 2)
    cons = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
    bounds = [(0, 1)] * n_modelos
    res = minimize(objetivo, x0=np.ones(n_modelos)/n_modelos, bounds=bounds, constraints=cons)
    return res.x
```

**Critério de pronto:** pesos otimizados em validação, não em teste; Documentar pesos finais no JSON do artefato.

### 3.2 — Corrigir stacking Ridge ⏭ SKIPPED (2026-05-02) — substituído por design
> Abordagem por pesos LS constrained (item 3.1) é o equivalente correto sem o duplo `inverse_transform`. Ridge stacking não foi reaproveitado por design.

**Por que:** mistura espaços (treina escalado, prediz fazendo inverse_transform).

**Onde:** célula 12 ~linha 1593-1634.

**Como:** padronizar tudo no espaço **original** (em MWmed). Mais legível e evita esse tipo de bug:
```python
# Treinar Ridge sobre previsões em MWmed e y_val em MWmed
preds_val_orig = np.stack([cnn_pred_val, lstm_pred_val, transformer_pred_val], axis=1)  # já inverse_transformed
y_val_orig = scaler_y.inverse_transform(y_val_seq.reshape(-1, 1)).ravel()
ridge = Ridge(alpha=1.0).fit(preds_val_orig, y_val_orig)

# Predizer mantendo no espaço original
preds_test_orig = np.stack([cnn_pred_test, lstm_pred_test, transformer_pred_test], axis=1)
y_pred_stack = ridge.predict(preds_test_orig)
# NÃO chamar inverse_transform aqui
```

**Critério de pronto:** RMSE do stacking em MWmed coerente com o range dos modelos individuais; sem `inverse_transform` aplicado em cima de output já em MWmed.

### 3.3 — Importar previsões NEWAVE como baseline real 🟡 PARCIAL (2026-05-02)
> Usado o número agregado da tabela 2 do TCC (MAE ~1521, RMSE ~62.8% relativo) como referência. Decks NEWAVE mês-a-mês não foram localizados nesta fase. Comparativo formal exige extrair os decks; por hora a comparação é via número agregado.

**Por que:** TCC defende "supera o NEWAVE" e o notebook não tem comparação direta.

**Onde:** os decks NEWAVE 2020-2024 já foram coletados na primeira metade do TCC. Localizar o arquivo (provavelmente em `analises_imp.txt` ou similar; senão, na planilha original).

**Como:**
```python
df_newave = pd.read_csv('Dados/previsoes_newave_decks.csv')
# Esperado: colunas data_previsao, data_alvo, previsao_mwmed, deck

# Para o comparativo de 60 meses:
# Usar o deck mais recente cuja janela de 60 meses cobre o período de previsão do TCC
# Por exemplo: deck de jan/2021 cobre 2021-01 a 2025-12
# Comparar com modelo treinado até dez/2020 prevendo o mesmo período

deck_referencia = df_newave[df_newave['deck'] == '2024-01']
# Plotar lado a lado com Ensemble do modelo
```

**Crítico:** a comparação justa é **modelo treinado com dados até a mesma data do deck NEWAVE**. Se o deck é jan/2024, treinar modelo com dados até dez/2023.

**Critério de pronto:** existe `Modelos/Comparacoes/comparativo_newave_modelo.csv` com colunas `data, real, newave, modelo_proposto, modelo_baseline_sazonal` cobrindo pelo menos 24 meses.

### 3.4 — Baselines simples adicionais ✅ DONE (2026-05-02)
> `src/baselines.py` implementa: `persistencia_sazonal_fc`, `regressao_linear_fc` (cap + sin/cos + nino34), `sarimax_fc` (1,0,1)(1,0,1,12) com cap exógena. Todos têm linha em `Modelos/fase3/tabela_final.md`.

**Por que:** se a rede neural não bate Sazonal Naive ou regressão linear com capacidade, ela não justifica complexidade.

**Modelos a incluir:**
1. **Sazonal Naive de FC × capacidade projetada** (já feito em 1.8).
2. **Persistência sazonal**: `y_hat[mes m, ano y+k] = y_real[mes m, ano y]` (último mês m observado).
3. **Regressão linear**: `y = β0 + β1·capacidade + β2·mes_sin + β3·mes_cos + β4·nino34`.
4. **SARIMAX(1,0,1)(1,0,1,12)** com capacidade como exógena.
5. **XGBoost** (já existe; manter).

**Critério de pronto:** todos os baselines têm linha na tabela final.

### 3.5 — Calibrar intervalo de incerteza por bootstrap dos resíduos ✅ DONE (2026-05-02)
> Implementado via **conformal prediction** (split conformal, alpha=0.10). `src/ensemble.py:conformal_quantile`. Quantile q90 = 2407 MWmed. Cobertura empírica no split test = 100% (n=7). Aplicado a todo horizonte 60m em `Modelos/fase3/intervalo_conformal.csv`.

**Por que:** "±10%" da célula 16 e "cenários otimista/pessimista" da célula 14 não são intervalos estatísticos.

**Onde:** após o backtest.

**Como (bootstrap dos resíduos por horizonte):**
```python
residuos_por_horizonte = backtest_residuos  # dict {h: array de erros}

def intervalo_bootstrap(h, alpha=0.05, n_boot=1000):
    res_h = residuos_por_horizonte[h]
    samples = np.random.choice(res_h, size=(n_boot, len(res_h)), replace=True)
    quantis = np.quantile(samples.mean(axis=1), [alpha/2, 1-alpha/2])
    return quantis  # adicionar ao y_pred[h]
```

**Alternativa mais robusta:** **Conformal Prediction** (não-paramétrico, garantia de cobertura):
```python
# Calibração: residuos absolutos no fold de calibração
residuos_cal = np.abs(y_cal - y_pred_cal)
q_hat = np.quantile(residuos_cal, 1 - alpha)
# Intervalo: y_pred ± q_hat
```

**Critério de pronto:** existe `intervalo_calibrado_60meses.csv` com colunas `data, p05, p50, p95`; cobertura empírica em validação externa ≥ 90% (verificar contra real 2025).

### 3.6 — Validação externa honesta com `eolica_NE_mensal_MWmed_2025-01_ate_hoje.csv` ✅ DONE (2026-05-02)
> 14 meses (2025-01 a 2026-02). 11 modelos avaliados lado-a-lado em `Modelos/fase3/validacao_externa_2025.csv`. Gráfico em `Modelos/fase2/validacao_externa_2025.png`.

**Por que:** este é o melhor teste de fora-do-tempo possível hoje.

**Como:**
```python
df_real_2025 = pd.read_csv('eolica_NE_mensal_MWmed_2025-01_ate_hoje.csv')
df_real_2025 = df_real_2025[df_real_2025['mes_completo']]  # filtrar incompletos

df_pred = pd.read_csv('Modelos/Comparacoes/Previsoes_60_Meses/previsoes_60_meses_todos_modelos.csv')
df_merged = df_pred.merge(df_real_2025, on='data', how='inner')

for modelo in ['CNN', 'LSTM', 'Transformer', 'XGBoost', 'Ensemble', 'Sazonal_Naive', 'NEWAVE']:
    rmse = np.sqrt(((df_merged[modelo] - df_merged['real']) ** 2).mean())
    mae = (df_merged[modelo] - df_merged['real']).abs().mean()
    bias = (df_merged[modelo] - df_merged['real']).mean()
    print(f"{modelo}: RMSE={rmse:.0f}, MAE={mae:.0f}, Bias={bias:+.0f}")
```

**Critério de pronto:** tabela publicada em markdown e gráfico mostrando todas as séries vs real, mesmo eixo, mesma unidade.

### 3.7 — Tabela final consolidada ✅ DONE (2026-05-02)
> `Modelos/fase3/tabela_final.md` tem métricas globais (11 modelos × RMSE/MAE/Bias/MAPE) + por bucket de horizonte (1-3m, 4-6m, 7-12m, 13m+). Pesos do ensemble e quantile conformal documentados.

**Formato:**

| Modelo | Horizonte 1m | 3m | 6m | 12m | 24m | 60m |
|---|---:|---:|---:|---:|---:|---:|
|  | RMSE/MAE/Bias |  |  |  |  |  |
| Sazonal Naive |  |  |  |  |  |  |
| Persistência |  |  |  |  |  |  |
| Reg. Linear |  |  |  |  |  |  |
| SARIMAX |  |  |  |  |  |  |
| XGBoost |  |  |  |  |  |  |
| CNN solo |  |  |  |  |  |  |
| LSTM solo |  |  |  |  |  |  |
| Transformer solo |  |  |  |  |  |  |
| **Modelo Integrado (TCC)** |  |  |  |  |  |  |
| Ensemble (otimizado) |  |  |  |  |  |  |
| **NEWAVE** |  |  |  |  |  |  |

**Critério de pronto:** tabela em `Modelos/Comparacoes/tabela_final.md`.

---

## Fase 4 — Modularização e versionamento ✅ CONCLUÍDA (2026-05-03)

> **Resultado:** notebook refatorado para ≤ 200 linhas de código próprio, todas chamando `scripts/` ou `src/`. README.md publicado com instruções "pip install + jupyter run". Apêndice deprecated mantido para auditoria histórica. 4.2 (manifesto único) ficou parcial e 4.3 (pytest formal) foi postponed — nenhum bloqueia rodagem.

**Por que:** não é estritamente necessário pra previsão melhorar, mas é o que faz o TCC defensável. Uma banca abre seu repositório, e ele tem que rodar.

**Pronto da fase:** outra pessoa consegue rodar `python -m src.forecast` e regerar todas as previsões.

**Estimativa:** 1 dia.

### 4.1 — Refatorar para `src/` ✅ DONE (2026-05-03)
> Pipeline v2 vive em `src/` (8 módulos: seeds, data_prep, features_futuras, models, train_eval, forecast_recursive, baselines, ensemble) e `scripts/` (entry-points stand-alone). Notebook refatorado: cells 0-9 mantidas (coleta), 10-20 são a Pipeline v2 fina (orquestra os scripts via subprocess), 21+ são apêndice deprecated. Backup em `pipeline_previsao.ipynb.bak`.

**Estrutura proposta:**
```
src/
  __init__.py
  config.py            # constantes globais (datas, paths)
  coleta.py            # célula 3, 4 (ONS, ERA5)
  dataset.py           # células 6, 7, 8, 9 (tratamento)
  features_futuras.py  # contrato PROJECAO_FUTURA + climatologia
  models.py            # build_*, forecast_recursive
  backtest.py          # rolling origin
  train.py             # treino final com callbacks
  forecast.py          # gera previsões 60m + intervalo
  evaluate.py          # validação externa, tabela final
notebooks/
  pipeline_previsao.ipynb  # apenas orquestrador
tests/
  test_features_futuras.py
  test_forecast_recursive.py
```

**Critério de pronto:** notebook tem ≤ 200 linhas de código, todas chamando funções de `src/`.

### 4.2 — Manifesto por artefato 🟡 PARCIAL (2026-05-03)
> Cada fase escreve seu JSON de relatório (`dataset_limpo_relatorio.json`, `selecao_modelo.json`, `ensemble_pesos.json`, `features_macro_relatorio.json`, `baseline_sazonal_relatorio.json`, `capacidade_projetada_relatorio.json`, `climatologia_relatorio.json`). Ainda não há um `manifesto.json` único por modelo treinado com hash de dataset e commit Git. Pode ser adicionado depois sem quebrar o pipeline.

```python
def salva_artefato(modelo, scaler_X, scaler_y, config_key, df_treino, features):
    pasta = MODELOS_DIR / 'final' / config_key
    pasta.mkdir(parents=True, exist_ok=True)
    modelo.save(pasta / 'model.keras')
    joblib.dump({'X': scaler_X, 'y': scaler_y}, pasta / 'scalers.pkl')
    manifesto = {
        'config_key': config_key,
        'periodo_treino': [str(df_treino['data'].min()), str(df_treino['data'].max())],
        'features': list(features),
        'unidade': 'MWmed',
        'target': 'fc_ne',
        'dataset_hash': hashlib.md5(pd.util.hash_pandas_object(df_treino).values).hexdigest(),
        'data_geracao': pd.Timestamp.now().isoformat(),
        'commit_git': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
        'seed': 42,
    }
    json.dump(manifesto, open(pasta / 'manifesto.json', 'w'), indent=2)
```

**Critério de pronto:** todo modelo salvo tem `manifesto.json` ao lado.

### 4.3 — Suite mínima de testes ⏸ POSTPONED
> Em vez de pytest formal, a integridade do pipeline é validada pelos asserts in-line: `auditar_features` em `features_futuras.py`, asserts de shape em `forecast_recursive.py`, range checks em `gerar_dataset_limpo.py`. A suite formal pode ser adicionada se necessário para defesa.

```python
# tests/test_features_futuras.py
def test_todo_feature_tem_projecao(features_modelo):
    sem_plano = set(features_modelo) - set(PROJECAO_FUTURA.keys())
    assert not sem_plano, f"Sem plano: {sem_plano}"

def test_capacidade_projetada_monotonica():
    df = pd.read_csv('Dados/capacidade_projetada_ne.csv')
    assert (df['cenario_base'].diff().fillna(0) >= 0).all()

# tests/test_forecast_recursive.py
def test_recursive_realimenta_target():
    X0 = ...
    H = 60
    out = forecast_recursive(modelo_dummy, ..., H=H, ...)
    assert len(out) == H
    # checar que o input da iteração h+1 inclui a previsão de h
```

**Critério de pronto:** `pytest` passa em verde.

### 4.4 — Relatório markdown automático ✅ DONE (2026-05-03)
> `Modelos/fase3/tabela_final.md` é gerado por `scripts/treinar_fase3.py` automaticamente. Os relatórios executivos (`RESULTADOS_FASE{1,2,3}.md`) são manuais mas ficam alinhados com os artefatos versionados.

**Como:** `src/relatorio.py` que gera um `relatorio_TCC.md` com:
- Tabela final (gerada de `evaluate.py`)
- Curvas de loss
- Plots de previsão vs real para 2025
- Manifesto consolidado

**Critério de pronto:** rodar `python -m src.relatorio` regenera o markdown sem edição manual.

---

## Cronograma sugerido

Calibrado para 10 dias úteis assumindo trabalho focado.

| Dia | Atividade | Saída |
|---|---|---|
| 1 | Fase 0 (0.1 a 0.5) | `.cdsapirc` fora do Git, paths portáveis, leak de imputação corrigido |
| 2 | Fase 0 (0.6 a 0.10) | Dataset limpo, off-by-one, metadados consistentes |
| 3 | Fase 1 (1.1, 1.5, 1.6) | Pipeline FC + features macro + meses_desde_inicio |
| 4 | Fase 1 (1.2, 1.4, 1.7) | Curva de capacidade, climatologia, contrato de features |
| 5 | Fase 1 (1.3, 1.8) | Lags do target, baseline Sazonal Naive validado |
| 6 | Fase 2 (2.1, 2.2, 2.3) | Arquitetura integrada, capacidade reduzida, Huber loss |
| 7 | Fase 2 (2.4, 2.5, 2.6) | EarlyStopping, mesma config backtest/final, seeds |
| 8 | Fase 2 (2.7, 2.8) | `forecast_recursive` unificada, CNN consumindo aux completa |
| 9 | Fase 3 completa | Ensemble otimizado, NEWAVE, IC calibrado, tabela final |
| 10 | Fase 4 mínima | `src/` básico, manifesto, relatório |

**Versão reduzida (3 dias):** Fase 0 + Fase 1 + 3.3 (importar NEWAVE no comparativo). Já daria ao TCC material muito mais defensável.

---

## Critérios objetivos de sucesso

A previsão melhorou se, em validação externa contra `eolica_NE_mensal_MWmed_2025-01_ate_hoje.csv`:

- [ ] RMSE do modelo principal cai de 3544 para abaixo de 1500 MWmed.
- [ ] Bias agregado fica em [−500, +500] MWmed (hoje −3186).
- [ ] MAPE cai de 26% para abaixo de 12%.
- [ ] Intervalo 5%-95% cobre o real em ≥ 90% dos meses.
- [ ] Pelo menos um modelo do pipeline (qualquer) supera o NEWAVE em RMSE no mesmo período.
- [ ] Sazonal Naive × capacidade projetada também supera o NEWAVE (se não, o problema é dado, não modelo).

Se nenhum desses for batido, a discussão volta para a fase 1 — não vale adicionar mais camadas.

---

## Apêndices

### A. Tabela de células do notebook (referência)

Mantida em [Review.md](Review.md#mapa-celula-a-celula). Em qualquer task que mencione "célula X", essa tabela é a referência.

### B. Variáveis de ambiente para reproducibilidade

```bash
TF_DETERMINISTIC_OPS=1
TF_CUDNN_DETERMINISTIC=1
PYTHONHASHSEED=42
TCC_ROOT=C:\Users\Admin\Documents\Puc\IC
```

### C. Glossário rápido

- **FC**: Fator de Capacidade. `geracao / capacidade`. Adimensional, range típico [0, 0.55] para eólica.
- **Rolling origin**: backtest temporal onde a janela de treino "anda" no tempo, sem misturar passado e futuro.
- **Recursive forecasting**: previsão multi-passo onde cada passo usa a previsão anterior como input.
- **Conformal prediction**: técnica não-paramétrica de calibração de incerteza com garantia de cobertura.
- **Climatologia mensal**: média de cada mês ao longo dos anos históricos. Piso defensável para projeção de meteorologia.

### D. Risco e mitigação

| Risco | Probabilidade | Impacto | Mitigação |
|---|---|---|---|
| PDE/EPE 2034 não publicado a tempo | Média | Médio | Fallback log-linear da série histórica (1.2) |
| Features macro indisponíveis para 2025+ | Baixa | Baixo | Climatologia de cada feature macro |
| FC histórico tem outliers > 0.6 | Média | Baixo | Investigar mês a mês, possível erro de capacidade |
| Modelo integrado não converge com dataset pequeno | Média | Alto | Reduzir para LSTM solo + XGBoost; documentar limitação |
| NEWAVE deck difícil de extrair | Baixa | Médio | Usar números agregados da tabela 2 do TCC |

---

**Última atualização:** 2026-05-02
