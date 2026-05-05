# Review do `pipeline_previsao.ipynb`

Este review foi feito em modo somente leitura sobre `pipeline_previsao.ipynb` e `TCC1.pdf`. O notebook tem 22 celulas. A numeracao abaixo e a numeracao real do notebook, contando celulas Markdown e codigo.

Tambem usei quatro subagentes especializados, um para cada area pedida: coleta, tratamento/dataset, treinamento/experimentacao e geracao de previsoes. Consolidei os achados deles com validacoes locais adicionais, incluindo uma comparacao parcial entre previsoes salvas e valores reais de 2025-01 a 2026-02.

## Contexto do TCC

Pelo `TCC1.pdf`, o plano e:

- mostrar que o NEWAVE tem limitacoes para previsao de geracao eolica de longo prazo;
- focar especialmente no subsistema Nordeste;
- usar dados reais do ONS e variaveis meteorologicas do ERA5;
- montar um modelo neural com componentes CNN, LSTM e Informer, alem de um XGBoost como baseline/controle;
- avaliar a capacidade de previsao no horizonte de 60 meses, comparavel ao horizonte dos decks do NEWAVE.

Isso importa porque o notebook hoje treina e seleciona modelos principalmente em horizonte de 12 meses e depois estende a previsao para 60 meses de forma recursiva. Esse ponto e o principal desvio entre o objetivo cientifico e o codigo atual.

## Mapa das celulas por area

| Area | Celulas | Papel |
|---|---:|---|
| Coleta | 1, 2, 3, 4, 22 | Setup de caminhos/dependencias, coleta ONS, coleta ERA5, coleta ONS recente ad hoc. |
| Tratamento de dados e montagem do dataset | 5, 6, 7, 8, 9, 10, parte da 13 | Processamento ONS, extracao ERA5 por parque, agregacao, clustering, merge, EDA, limpeza para modelagem. |
| Treinamento do modelo e plano de experimentacao | 11, 12, 13, 14, 15, 16, 17, 18, 19 | Ensemble inicial, grid/backtesting, treino final, XGBoost baseline, selecao do modelo final, comparativos. |
| Geracao de previsoes | parte da 14, parte da 16, 20, 21, apoio da 22 | Previsoes de 60 meses por cenarios, previsao XGBoost, previsao todos os modelos, coleta real posterior para checagem. |

## Mapa celula a celula

| Celula | Tipo | Classificacao | Descricao |
|---:|---|---|---|
| 1 | Codigo | Setup / apoio a todas as areas | Importa bibliotecas, cria diretorios, define caminhos, anos, UFs do Nordeste, status operacional. |
| 2 | Markdown | Coleta | Cabecalho "Coleta de Dados". |
| 3 | Codigo | Coleta | Baixa arquivos ONS de balanco, carga e curva de carga. |
| 4 | Codigo | Coleta + pre-processamento de coleta | Monta mapeamento parque/celula ERA5, busca capacidade local, baixa ERA5 mensal por celula. |
| 5 | Markdown | Tratamento | Cabecalho "Tratamento de Dados e Geracao de Estatisticas". |
| 6 | Codigo | Tratamento ONS | Processa ONS, gera serie mensal, capacidade, carga, demanda e features derivadas. |
| 7 | Codigo | Tratamento ERA5 | Abre NetCDF/ZIP ERA5, extrai variaveis por parque e mes. |
| 8 | Codigo | Tratamento ERA5 | Agrega ERA5 por media ponderada por capacidade e cria features meteorologicas regionais. |
| 9 | Codigo | Montagem dataset | Faz clustering espacial, pivot por cluster, merge ONS + ERA5 e salva `dataset_final_modelagem`. |
| 10 | Codigo | EDA / validacao exploratoria | Analisa dataset final, correlacoes, clusters e estatisticas. |
| 11 | Markdown | Treinamento | Cabecalho "Pipeline dos Modelos". |
| 12 | Markdown | Treinamento | Cabecalho "Pipeline Ensemble". |
| 13 | Codigo | Treinamento exploratorio + limpeza | Prepara dados, remove algumas colunas de vazamento, treina CNN/LSTM/Informer em split 70/15/15 e cria ensemble inicial. |
| 14 | Codigo | Plano de experimentacao + treino final + previsao | Motor de backtesting, grid de hiperparametros, selecao, treino final, cenarios e previsao de 60 meses. |
| 15 | Markdown | Baseline | Cabecalho "Pipeline XGBoost - Baseline/Controle". |
| 16 | Codigo | Treinamento XGBoost + previsao | Backtesting XGBoost, selecao de janela, treino final e previsao recursiva de 60 meses. |
| 17 | Markdown | Selecao | Texto de "Selecao do Modelo Final do TCC". |
| 18 | Codigo | Selecao final | Carrega rankings, compara Ensemble/XGBoost e copia artefatos finais. |
| 19 | Codigo | Comparacao experimental | Consolida previsoes e metricas, cria comparativos estatisticos entre modelos. |
| 20 | Markdown | Previsao | Texto de "Previsao 60 Meses - Todos os Modelos". |
| 21 | Codigo | Geracao de previsoes | Carrega modelos/scalers e gera CSV/JSON/PNG de 60 meses para CNN, LSTM, Informer, XGBoost e Ensemble. |
| 22 | Codigo | Coleta/validacao posterior | Baixa ONS real recente e gera `eolica_NE_mensal_MWmed_2025-01_ate_hoje.csv`; nao alimenta o dataset final. |

## Resumo executivo dos maiores riscos

1. **A previsao de 60 meses nao esta validada como previsao de 60 meses.** O ranking final efetivo so contem `H=12`; o backtesting de 60 meses nao executou por falta de dados. Hoje a previsao de 60 meses e uma extensao recursiva de modelos selecionados para 12 meses.

2. **As covariaveis futuras estao mal definidas.** ERA5 e reanalise historica; nao existe ERA5 real para 2025-2029. A celula 14 tenta cenarios, mas usa nomes de colunas que nao existem para capacidade/carga. A celula 21 praticamente congela meteorologia, capacidade e carga, mudando quase so calendario/regime.

3. **Ha vazamento temporal e risco de vazamento de target.** A celula 13 imputa NaNs com media calculada no dataset inteiro antes do split. O dataset bruto salvo contem colunas derivadas do target (`capacity_factor`, `penetracao`, `corte`), que alguns fluxos removem depois, mas o artefato continua perigoso.

4. **A selecao de modelos usa informacao de teste em pontos importantes.** O XGBoost escolhe melhor janela por `test_rmse`, nao por validacao. No grid neural, o ranking tambem e baseado em metricas de teste das rodadas, enquanto o teste deveria ser reservado para avaliacao final.

5. **Unidades e artefatos estao inconsistentes.** O target e `MWmed`, mas graficos e textos usam `GWh` em varios pontos. O `metadata_inferencia.json` em `Modelos/TCC_Modelo_Final` aponta `L48_H12_W72`, enquanto a configuracao final atual aponta `L24_H12_W60`.

6. **Existe risco real de credencial exposta.** `.cdsapirc` esta rastreado pelo Git e contem chave CDS. Nao imprimi a chave, mas esse arquivo deve ser tratado como segredo.

## Achados detalhados: Coleta

### C1 - Setup, caminhos e dependencias

- **Critico: `.cdsapirc` rastreado no Git.** O arquivo existe na raiz e `git ls-files` confirma que esta versionado. Como ele contem chave da CDS API, isso deve ser rotacionado e removido do versionamento.
  - Recomendacao: rotacionar a chave CDS, manter `.cdsapirc` no perfil do usuario ou variavel de ambiente, e impedir novo versionamento do arquivo.

- **Caminho absoluto fragil.** `pasta_central = r'C:\\Users\\Admin\\Documents\\Puc\\IC'` prende o notebook a uma maquina.
  - Recomendacao: usar `Path.cwd()` ou um arquivo de configuracao local.

- **Recorte temporal centralizado de forma fraca.** `anos = range(2010, 2025)` significa 2010-2024, mas outras celulas processam 2025 ou usam `date.today()`.
  - Recomendacao: criar `DATA_CORTE_DADOS` e `ULTIMO_MES_COMPLETO` por fonte.

- **Dependencias sem versao fixa e arquivo com nome inconsistente.** O arquivo local e `requiremets.txt`, mas o comentario fala em `requirements.txt`. Tambem nao ha pins de versao.
  - Recomendacao: padronizar para `requirements.txt` e fixar versoes principais de `pandas`, `xarray`, `tensorflow`, `xgboost`, `cdsapi`, `netCDF4`.

### C3 - Coleta ONS

- **`CAPACIDADE_GERACAO.csv` e dependencia oculta.** A celula baixa `BALANCO_ENERGIA_SUBSISTEMA`, `CARGA_ENERGIA` e `CURVA_CARGA`, mas as celulas 4, 6 e 7 dependem de `CAPACIDADE_GERACAO.csv`.
  - Recomendacao: baixar capacidade explicitamente e salvar metadados de origem, data de extracao e schema.

- **A aparencia de paralelismo e enganosa.** A celula abre `ThreadPoolExecutor`, mas chama `downloadFile(...)` diretamente, sem `ex.submit`. Alem disso, `results` nunca e atualizado.
  - Recomendacao: submeter cada arquivo/ano como job, capturar excecoes, atualizar contadores reais e gerar manifest.

- **Downloads pulam arquivo apenas por existencia.** Se um CSV estiver parcial ou corrompido, sera reutilizado.
  - Recomendacao: validar tamanho minimo, colunas esperadas, numero de meses/linhas e hash.

- **Um 404 pode encerrar a serie daquele alvo cedo demais.** `return "not_found"` sai da funcao inteira.
  - Recomendacao: registrar o 404 por ano e continuar nos demais anos quando fizer sentido.

### C4 - Coleta ERA5

- **Baixa muitas celulas que nao entram no treino historico.** O mapeamento tem 225 celulas, mas parte vem de parques futuros, sem data ou nao operacionais no periodo historico. Isso aumenta custo e mistura footprint futuro com coleta historica.
  - Recomendacao: separar modo historico observado de modo cenario futuro.

- **Download ERA5 nao usa arquivo temporario validado.** O `retrieve` grava direto no destino. Se interromper, um `.nc` ruim pode ficar e ser pulado depois.
  - Recomendacao: baixar em `.part`, abrir/validar NetCDF, conferir variaveis e 12 meses, depois `os.replace`.

- **Sem manifest de coleta.** Nao ficam registrados hash, bytes, data da coleta, variaveis, bbox ou status por arquivo.
  - Recomendacao: salvar `manifest_coleta_era5.csv/json`.

- **Dados futuros do ERA5 nao existem.** Para previsao 2025-2029, ERA5 nao e fonte futura; e reanalise historica.
  - Recomendacao: definir cientificamente se as covariaveis futuras virao de climatologia mensal, cenarios, previsao sazonal/climatica ou serao removidas do modelo de longo prazo.

### C22 - Coleta ONS recente ad hoc

- **A celula esta solta do pipeline.** Ela baixa real 2025-2026 e salva na raiz, mas nao atualiza `dataset_final_modelagem`.
  - Recomendacao: integrar como etapa formal de avaliacao externa ou mover para um script/notebook de validacao.

- **Usa `date.today().year`.** Isso muda o resultado conforme o dia em que roda e pode incluir meses parciais.
  - Recomendacao: usar ultimo mes completo explicitamente e filtrar meses incompletos.

## Achados detalhados: Tratamento e montagem do dataset

### C6 - Processamento ONS

- **ONS processa 2015-2025, mas o dataset final termina em 2024.** O ERA5 coletado vai ate 2024; a juncao por `inner merge` na celula 9 derruba 2025 silenciosamente.
  - Recomendacao: antes do merge, imprimir e salvar quais meses foram descartados por fonte.

- **Colunas derivadas do target entram no dataset bruto.** `corte_eolica_ne_mwmed`, `capacity_factor_ne`, `penetracao_eolica_ne` e `penetracao_eolica_sin` usam `geracao_eolica_ne_mwmed`.
  - Recomendacao: salvar dois datasets: `dataset_raw_com_diagnosticos` e `dataset_modelagem_sem_vazamento`.

- **`corte_eolica_ne_mwmed` e uma heuristica, nao curtailment real.** A regra `max(0, capacidade * 0.45 - geracao)` pode ensinar ao modelo uma relacao artificial.
  - Recomendacao: se nao houver dado real de restricao/curtailment, manter essa variavel fora do treino e rotular como diagnostico exploratorio.

- **Capacidade mensal usa uma regra de mes inteiro pouco precisa.** A logica considera ativo a partir do dia 1 do mes. Para MWmed mensal, usinas que entram no meio do mes deveriam contribuir proporcionalmente aos dias ativos.
  - Recomendacao: calcular capacidade media ponderada por dias ativos no mes.

### C7 e C8 - Processamento ERA5

- **A extracao por parque e boa para preservar peso por capacidade, mas gera duplicacao pesada.** A celula 7 expande dados para centenas de milhares de linhas antes da agregacao.
  - Recomendacao: para performance, considerar extrair por celula e depois agregar pesos por celula/mes sem repetir todos os parques quando possivel.

- **NaN e pesos precisam de validacao mais forte.** A celula 8 filtra ativos, mas `weighted_average` nao trata explicitamente soma de pesos igual a zero.
  - Recomendacao: assertar `peso_capacidade_mw > 0` para parques ativos ou tratar soma zero como NaN controlado.

- **Sem indicador de imputacao/atividade.** Quando um cluster ainda nao tinha parque ativo, isso aparece como NaN depois.
  - Recomendacao: criar flags como `cluster_ativo` e `meteo_imputada` para o modelo saber quando a informacao e estruturalmente ausente.

- **Confirmar semantica de `tp`.** A conversao de precipitacao para mm depende da semantica do produto ERA5 mensal.
  - Recomendacao: documentar se `tp` e acumulado mensal, media diaria acumulada ou outra unidade do produto CDS escolhido.

### C9 - Juncao ONS + ERA5 com clustering espacial

- **O dataset final salvo tem NaNs.** Verificacao local: `Dados/dataset_final_modelagem.csv` tem 120 linhas, 95 colunas, periodo 2015-01 a 2024-12, mas 800 NaNs em 24 colunas de clusters.
  - Recomendacao: salvar um artefato final realmente pronto para modelagem ou deixar claro que a limpeza posterior e obrigatoria.

- **O `inner merge` mascara perda temporal.** ONS 2025 e descartado porque ERA5 nao existe para 2025 no dataset final.
  - Recomendacao: usar auditoria de cobertura por fonte antes/depois do merge.

- **Clustering usa a visao espacial de todo o periodo.** Para backtesting causal estrito, usar celulas/parques que so aparecem no futuro pode ser conhecimento futuro.
  - Recomendacao: ou declarar o grid espacial como conhecimento exogeno fixo, ou recalcular/fixar clusters apenas com informacao disponivel no cutoff.

- **SST de clusters continentais e preenchido pelo vizinho costeiro.** Isso elimina NaN, mas pode criar sinal fisicamente fraco para parques interiores.
  - Recomendacao: adicionar `sst_imputada` e distancia ao vizinho, ou remover SST de clusters onde SST nao tem interpretacao local.

### C10 - EDA

- **A EDA mostra um sinal importante:** capacidade vs geracao tem correlacao alta, e melhor cluster de vento tem correlacao alta. Isso sugere que crescimento estrutural de capacidade e tao importante quanto meteorologia.
  - Recomendacao: tratar capacidade futura como cenario central, nao como climatologia ou valor congelado.

## Achados detalhados: Treinamento e plano de experimentacao

### C13 - Ensemble inicial

- **Deve ser tratado como exploratorio/deprecated.** A saida salva mostra CNN com RMSE de aproximadamente 140 mil MWmed e ensemble em torno de 35 mil MWmed, muito pior que os outros modelos. Isso contamina comparativos se for misturado com resultados da celula 14.
  - Recomendacao: marcar a celula 13 como exploratoria ou remover seus artefatos dos comparativos finais.

- **Imputacao com vazamento temporal.** A celula preenche NaNs de features meteorologicas com a media da coluna calculada no dataset inteiro antes do split.
  - Recomendacao: criar imputador por rodada/fold, com `fit` apenas no treino.

- **`mes_sin` e `mes_cos` sao escalados aqui.** A celula 14 corrige isso mantendo time features cruas, mas a celula 13 nao.
  - Recomendacao: centralizar a logica de scaling e reutilizar nas celulas de treino e inferencia.

- **Split 70/15/15 gera avaliacao rolling dentro do bloco.** Isso pode ser valido, mas nao equivale a prever o bloco inteiro a partir de um unico cutoff.
  - Recomendacao: nomear como rolling-origin ou preferir a logica de uma origem por rodada da celula 14.

### C14 - Motor de experimentos, grid e treino final

- **O grid declarado nao foi efetivo para `H=24`.** O arquivo `ranking_configuracoes_fase3.csv` tem 16 linhas, todas com `H=12`. Nao ha resultado `H=24` no ranking final.
  - Recomendacao: reportar que a selecao final foi validada para 12 meses, ou ajustar o grid/dados para que `H=24` tenha rodadas validas.

- **Backtesting de 60 meses nao executou.** A saida indica "Backtesting de 60 meses nao executado (dados insuficientes)".
  - Recomendacao: nao apresentar o modelo como validado para 60 meses. Apresentar como forecast recursivo de 12 meses ate haver protocolo de 60 meses.

- **RMSE por horizonte com uma amostra por rodada vira erro absoluto daquela rodada.** Como cada rodada tem uma origem por horizonte, o RMSE de uma rodada/horizonte e numericamente igual ao absoluto do erro. Depois o codigo tira medias desses RMSEs.
  - Recomendacao: acumular erros de todas as origens e calcular `sqrt(mean(error^2))` no agregado.

- **Selecao usa metricas de teste das rodadas.** O ranking global vem de `round_result['metrics']`, que sao metricas de teste. Teste deve ser avaliacao final, nao criterio de escolha.
  - Recomendacao: escolher configuracao por `metrics_val` e guardar teste congelado para a tabela final.

- **Treino final tem poucas sequencias e nao usa validacao/callbacks.** Na configuracao vencedora `L=24`, `H=12`, `W=60`, o treino final produz cerca de 25 sequencias. Depois treina 100 epocas sem early stopping.
  - Recomendacao: usar validacao final, `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint` e repeticoes por seeds.

- **Seeds nao fecham reprodutibilidade.** `TF_DETERMINISTIC_OPS` e definido depois de TensorFlow ja ter sido importado em outros pontos; `train_models_for_round_v2` aceita `seed`, mas `run_backtesting_v2` nao passa.
  - Recomendacao: setar variaveis de ambiente antes de importar TensorFlow, passar seed por rodada e rodar 3-5 seeds por configuracao.

- **Arquitetura diverge do texto do TCC.** O PDF descreve uma arquitetura integrada onde CNN alimenta ramos LSTM/Informer, e tambem menciona ablations como CNN->LSTM e CNN->Informer. O notebook implementa ramos paralelos CNN, LSTM e Informer.
  - Recomendacao: alinhar texto e codigo. Ou o TCC vira "ensemble paralelo", ou o notebook implementa as ablations descritas.

- **Baseline sazonal esta no lugar errado.** A secao 9.6 roda dentro do `else` de dados insuficientes para backtesting 60m. Se no futuro houver dados suficientes, essa comparacao pode deixar de rodar.
  - Recomendacao: calcular baseline sempre, por origem de backtesting e no teste congelado.

- **Comparar com NEWAVE ainda nao esta no protocolo final.** O TCC reporta metricas do NEWAVE; o notebook precisa comparar os modelos no mesmo protocolo de origem/horizonte.
  - Recomendacao: incluir NEWAVE como baseline real no mesmo formato de avaliacao.

### C16 - XGBoost baseline

- **A melhor janela e escolhida por `test_rmse`.** Isso usa teste para selecao.
  - Recomendacao: escolher por `val_rmse` e reportar `test_rmse` apenas depois.

- **A previsao de 60 meses congela/copía covariaveis.** O XGBoost atualiza lags do target, calendario e algumas colunas, mas meteorologia/capacidade/carga continuam essencialmente copiadas da ultima linha.
  - Recomendacao: usar o mesmo contrato de covariaveis futuras da celula 14.

- **Banda de incerteza `+-10%` e arbitraria.** Isso nao e intervalo estatistico.
  - Recomendacao: renomear como faixa ilustrativa ou calibrar com residuos/backtesting.

### C18 - Selecao final

- **Artefatos finais tem metadados inconsistentes.** `artefato_final_tcc.json` aponta `Informer`, `L24_H12_W60`; `metadata_inferencia.json` aponta `config: L48_H12_W72`, `lookback: 48`.
  - Recomendacao: regenerar metadados a partir de `configuracao_final.json` e salvar hash/data/config junto de cada modelo.

- **A selecao "final" ainda e por horizonte curto.** O vencedor Informer tem `H=12`, nao uma validacao direta de 60 meses.
  - Recomendacao: deixar isso explicito no TCC e no nome dos artefatos.

### C19 - EDA comparativa

- **Pode misturar resultados antigos/incompativeis.** A celula tenta puxar `ENSEMBLE_RESULTS` e resultados XGBoost do estado em memoria. Se a celula 13 foi executada antes da 14, os comparativos podem incluir o ensemble exploratorio ruim.
  - Recomendacao: carregar comparativos apenas de artefatos versionados da celula 14 e XGBoost, nao de variaveis globais soltas.

- **Teste Diebold-Mariano e metricas estatisticas precisam de cuidado com dependencia.** Previsoes multi-horizonte e rolling origins podem ter erros autocorrelacionados e sobrepostos.
  - Recomendacao: documentar o protocolo estatistico e usar erros alinhados por origem/horizonte.

## Achados detalhados: Geracao de previsoes

### C14 - Cenarios de 60 meses

- **Off-by-one no mes futuro.** Em `gerar_cenarios_futuros_tcc`, o primeiro passo usa `ultimo_mes.month + h`, entao quando `h=0` repete o mes do ultimo dado historico. Se o ultimo dado e 2024-12, o primeiro cenario usa dezembro, mas a previsao salva comeca em 2025-01.
  - Recomendacao: usar `future_date = ultimo_mes + pd.DateOffset(months=h+1)`.

- **Features tendenciais usam nomes que nao existem no dataset.** A lista usa `capacidade_instalada_mw`, `carga_mwmed`, `demanda_max_mw`. As colunas reais sao `capacidade_eolica_ne_mw`, `carga_ne_mwmed`, `demanda_sin_mwmed`, `demanda_max_sin_mw`, `crescimento_capacidade_mw`.
  - Recomendacao: criar um dicionario explicito `feature -> metodo_de_projecao` e abortar se alguma feature ficar sem classificacao.

- **Faixa de incerteza nao e incerteza probabilistica.** O grafico usa cenarios otimista/pessimista e chama de faixa de incerteza.
  - Recomendacao: chamar de "faixa de cenarios" ou calibrar intervalos com bootstrap, quantis, MC dropout, conformal prediction ou residuos de backtesting.

- **Rotulos em GWh estao errados.** O target e `geracao_eolica_ne_mwmed`.
  - Recomendacao: manter MWmed ou converter para GWh com `MWmed * horas_do_mes / 1000`.

### C21 - Previsao 60 meses todos os modelos

- **Congelamento de covariaveis futuras.** `advance_window_no_roll` copia a ultima linha e atualiza quase so calendario/regime. Como `colunas_features` nao inclui `ano`/`mes` na lista final, na pratica mudam `mes_sin`, `mes_cos` e `regime_operacional`; meteorologia, capacidade, carga e demanda ficam congeladas.
  - Recomendacao: usar os cenarios da celula 14 ou um gerador unico de covariaveis futuras.

- **Risco de modelo e scaler incompativeis.** A celula carrega `scalers_finais.pkl`, mas prioriza modelos em memoria de `ENSEMBLE_RESULTS` antes de `FINAL_MODELS`.
  - Recomendacao: carregar modelo + scaler do mesmo bundle e validar `config_key`, lista de colunas e shapes.

- **Features ausentes viram zero silenciosamente.** Ao montar `X_full`, se uma feature esperada nao existe no dataframe, fica zero.
  - Recomendacao: abortar com erro listando features faltantes, exceto colunas com regra de projecao explicita.

- **O markdown promete mais do que a celula entrega.** A celula 20 fala em todos os modelos e modelo final TCC, mas a celula 21 nao carrega explicitamente `modelo_final_tcc.keras`; ela usa CNN/LSTM/Informer/XGBoost/Ensemble.
  - Recomendacao: alinhar texto e codigo, ou carregar o artefato final do TCC como modelo separado.

- **JSON de previsao perde contexto.** O JSON salva listas por modelo, mas nao inclui datas por ponto, unidade, cenario, config/model hash, nem origem do scaler.
  - Recomendacao: salvar registros com `{data, horizonte, modelo, previsao, unidade, cenario, config_key}`.

### Comparacao parcial com valores reais ja disponiveis

Usei o CSV local `eolica_NE_mensal_MWmed_2025-01_ate_hoje.csv` contra `Modelos/Comparacoes/Previsoes_60_Meses/previsoes_60_meses_todos_modelos.csv`. Ha 14 meses em comum, de 2025-01 a 2026-02.

| Modelo | RMSE | MAE | Bias | MAPE |
|---|---:|---:|---:|---:|
| CNN | 4204.9 | 3602.6 | -3602.2 | 31.4% |
| LSTM | 2627.0 | 2192.5 | -1898.7 | 17.6% |
| Informer | 3830.4 | 3443.4 | -3375.2 | 27.4% |
| XGBoost | 2098.0 | 1829.1 | -1260.5 | 15.1% |
| Ensemble | 3544.4 | 3221.6 | -3186.7 | 26.0% |

Leitura educativa: esse teste nao e o protocolo final, porque usa dados reais posteriores que nao estao integrados ao pipeline. Mesmo assim, ele mostra um sinal claro: as previsoes neurais e o ensemble estao subestimando forte 2025, especialmente de maio a outubro. Isso combina com o problema de capacidade/covariaveis futuras congeladas ou mal projetadas.

## Must-fix antes de usar resultados no TCC

1. **Rotacionar/remover a chave `.cdsapirc` do Git.**

2. **Criar um contrato unico de dados e datas.** Definir corte temporal por fonte, ultimo mes completo, manifest de coleta e meses descartados no merge.

3. **Salvar dataset de modelagem sem vazamento.** Remover colunas derivadas do target do artefato de treino, ou pelo menos criar asserts que impeçam essas colunas em `FEATURES_*`.

4. **Mover imputacao/scaling para dentro de cada rodada temporal.** Nenhum `mean`, scaler ou imputador pode aprender com validacao/teste.

5. **Separar validacao de selecao e teste final.** Selecionar hiperparametros por validacao; usar teste congelado so uma vez, para relatorio.

6. **Corrigir cenarios de covariaveis futuras.** Capacidade, carga, demanda e meteorologia precisam de regras explicitas e auditaveis. A previsao de 60 meses nao pode depender da ultima linha congelada.

7. **Corrigir off-by-one de meses na celula 14.**

8. **Validar ou renomear a ambicao de 60 meses.** Hoje o modelo e selecionado para 12 meses e estendido recursivamente. Para afirmar "modelo de 60 meses", precisa de backtest de 60 meses ou de uma limitacao explicita.

9. **Comparar contra NEWAVE no mesmo protocolo.** O TCC discute NEWAVE; entao o baseline principal deve entrar na mesma tabela de origem/horizonte/unidade.

10. **Padronizar unidade como MWmed ou converter corretamente para GWh.**

## Melhorias que podem aumentar a qualidade da previsao

- **Modelar capacidade futura como driver central.** A correlacao capacidade vs geracao e alta. Use cenarios PDE/EPE/ANEEL ou pelo menos curvas conservadora/base/otimista calibradas.

- **Adicionar baselines fortes e simples.** Sazonal naive com tendencia, persistencia sazonal, regressao linear com capacidade, SARIMAX/ETS e XGBoost corretamente validado ajudam a saber se a rede neural esta realmente agregando valor.

- **Fazer ablation real.** Se o TCC promete LSTM, Informer, CNN->LSTM, CNN->Informer, Ensemble e XGBoost, cada ablation precisa existir com a mesma avaliacao.

- **Repetir por seeds.** Com poucas sequencias, uma unica seed pode mudar muito o resultado. Reportar media e desvio por 3-5 seeds.

- **Criar um pipeline modular.** Separar `coleta.py`, `dataset.py`, `features_futuras.py`, `backtest.py`, `train.py`, `forecast.py` ou funcoes equivalentes. O notebook pode virar orquestrador, mas a logica critica deve ser testavel.

- **Registrar artefatos com versao.** Todo modelo/scaler/previsao deve carregar `config_key`, periodo de treino, lista de features, hash do dataset, unidade e data de geracao.

- **Usar o real 2025-2026 como validacao externa.** O arquivo da celula 22 pode virar uma avaliacao honesta dos primeiros meses previstos, desde que integrado e filtrado para meses completos.

## Conclusao

O pipeline ja tem uma base ambiciosa e varios componentes bons: coleta ONS/ERA5, ponderacao por capacidade, clustering espacial, modelos neurais variados, XGBoost baseline e tentativa de backtesting temporal. O principal problema nao e falta de modelo; e o contrato temporal.

Para previsao de 60 meses, o ganho mais importante agora nao e adicionar mais camadas na rede. E garantir que cada feature usada no futuro exista de forma causal, que a selecao nao veja teste, que o horizonte de avaliacao bata com o horizonte defendido no TCC, e que os artefatos finais sejam coerentes entre si.

---

## Achados complementares (segunda passagem)

Esta secao complementa o review acima com pontos que passaram batido na primeira leitura. Foram confirmados por inspecao direta do notebook (celulas 12, 13, 20, 21) e cruzados com o que o TCC promete na secao 3.2 (Arquitetura Integrada Proposta) e secao 4.1 (Plano de Acao).

### A1 - A arquitetura integrada CNN -> LSTM/Informer NAO existe

O TCC promete (secao 3.2): "extrator convolucional 1D (CNN), cujo papel e aprender filtros locais [...] entregando mapas de atributos mais informativos e compactos para os previsores temporais. A partir desses mapas, a arquitetura ramifica-se em dois ramos complementares" - LSTM e Informer recebendo features da CNN.

No notebook, os tres ramos rodam **paralelos sobre o mesmo input bruto**. CNN, LSTM e Informer sao construidos como modelos independentes (`build_cnn`, `build_lstm`, `build_informer` na celula 12) e treinados separadamente. Os "embeddings" intermediarios da CNN e do LSTM sao salvos em `.npy` mas nunca consumidos por outro ramo. A "fusao" do ensemble e media ponderada das tres previsoes finais, nao fusao de representacoes.

- **Impacto direto na previsao**: o que esta sendo chamado de "Ensemble CNN+(LSTM+Informer)" e na verdade um voting de tres modelos independentes treinados nos mesmos dados, com correlacao alta entre eles. O valor agregado de um ensemble vem de erros descorrelacionados; aqui os erros tendem a se mover juntos, e por isso o ensemble esta com bias proximo ao da CNN/Informer (ver tabela do RMSE 2025 no review original: Ensemble bias -3186 contra Informer -3375).
- **Recomendacao**: ou implementar a arquitetura prometida (CNN extrai features -> alimenta LSTM e Informer), ou redefinir o TCC para "ensemble paralelo de CNN, LSTM e Informer" e justificar no texto.

### A2 - A previsao recursiva NAO realimenta o target

Em `advance_window_no_roll` (celula 20, ~linha 229) o avanco da janela copia a ultima linha observada e atualiza apenas calendario/regime. O alvo predito (`y_hat`) **nao volta como feature** no proximo passo, porque nao existe feature `geracao_lag_*` no input. O modelo prevê 60 meses como se cada mês começasse de jan/2025 sem saber o que ele mesmo previu.

Combinado com o congelamento de covariaveis ja apontado, isso explica o quadro de subestimacao crescente: o modelo nao tem nem dinamica auto-regressiva nem cenario futuro coerente.

- **Recomendacao**: incluir `geracao_lag_1, geracao_lag_3, geracao_lag_6, geracao_lag_12` no input de treino, e no loop recursivo substituir o lag pela previsao do passo anterior.

### A3 - Falta o feature mais importante: NAO modelar fator de capacidade

O target e `geracao_eolica_ne_mwmed` em valor absoluto. A capacidade instalada cresceu ~5x no periodo. Logo, mais da metade do "sinal" que o modelo precisa aprender e simplesmente seguir a curva de capacidade. Modelar o **fator de capacidade** (`geracao / capacidade`) e re-multiplicar por capacidade futura na inferencia separa o problema em duas partes muito mais tratáveis: (1) cenario de capacidade, que e exogeno e conhecido com baixa incerteza dos planos PDE/EPE; (2) FC, que e estacionario (entre 0 e ~0.55) e tem sazonalidade clara.

A coluna `capacity_factor_ne` foi banida do treino por suspeita de vazamento (celula 12 ~linha 56). Isso e exagero - FC historico nao vaza target futuro: ele e calculado apenas com dados que ja eram observados na linha. Banir foi fechar a porta de uma das melhores melhorias possiveis.

- **Recomendacao**: criar um pipeline alternativo `pipeline_fc.py` que treina sobre FC, valida no mesmo protocolo e reconstroi geracao na saida. Comparar honestamente com o pipeline atual.

### A4 - Implementacao do Informer enfraquece o que o TCC promete

Tres problemas concretos:

1. **ProbSparse mal implementada** (celula 12, ~linha 1073-1141): calcula softmax full e zera valores abaixo de `1/L_K` *depois*. Isso e o oposto do paper - o ganho do Informer vem de selecionar Top-u queries via amostragem KL-divergence *antes* do softmax, reduzindo complexidade. Como esta, nao ha ganho de eficiencia nem de inducao.

2. **Distilling desativado** (celula 12 ~linha 1288): todas as camadas com `distilling=False`. O componente que da nome ao Informer nao roda.

3. **Configuracao do backtest difere do treino final**. No backtest (celula 13 ~linha 962-966): `d_model=32, n_heads=2, e_layers=1, d_ff=64`. No treino final (celula 12 ~linha 1326-1331): `d_model=64, n_heads=4, e_layers=2, d_ff=128, dropout=0.2`. **Isso invalida a selecao de modelo** - voce escolheu o Informer porque ganhou no backtest, mas treinou um modelo diferente no final.

- **Impacto**: pesar o Informer com 0.61 no ensemble (celula 20 ~linha 189) sendo que ele esta mal implementado e provavelmente a causa direta da subestimacao do ensemble agregado.
- **Recomendacao**: simplificar para um Transformer encoder-decoder padrao com sinusoidal/learned positional encoding, ou implementar ProbSparse Top-u corretamente. Recalibrar pesos do ensemble por validacao apos a correcao.

### A5 - O stacking Ridge mistura espacos e produz numero invalido

Na celula 12 (~linha 1593-1634) o meta-regressor Ridge e treinado em `meta_y_val` que e y de validacao **escalado** (`y_val_seq.flatten()`), mas no teste aplica `inverse_transform` na predicao como se ela ja viesse escalada. Treina em escala A, prevê e desfaz escala como se fosse escala B - o RMSE publicado para o stacking e numericamente invalido.

- **Recomendacao**: padronizar todo o stacking no espaco original (inverse-transform antes de empilhar) ou no espaco escalado (sem inverse-transform). Documentar explicitamente.

### A6 - CNN ignora 23 dos 24 meses de aux

Em `build_cnn` (celula 12 ~linha 433-438): `aux_last = Lambda(lambda t: t[:, -1, :])(input_aux)`. A CNN olha a serie inteira do target (24 meses) mas das auxiliares (capacidade, carga, demanda, crescimento) so usa o ultimo timestep. Justamente as features que carregam a tendencia estrutural sao truncadas.

- **Recomendacao**: passar aux por uma Conv1D paralela ou concatenar canal-a-canal com o target antes da CNN principal.

### A7 - `regime_operacional` vira constante no futuro

Definido por buckets de ano hardcoded (celula 12 ~linha 168-179): `<2017:0, <2020:1, <2022:2, >=2022:3`. Para qualquer ano >=2025 o valor e fixo em 3. O modelo recebe uma feature constante por 60 meses - inutil e potencialmente confunde o ensemble.

- **Recomendacao**: substituir por `tempo_desde_inicio_setor` (numero de meses desde 2010) ou remover.

### A8 - Hiperparametros mal dimensionados para o tamanho da base

Sao 120 linhas mensais (2015-2024). Apos rolling origin com lookback 24, sobram em torno de 50-70 sequencias para treino. As redes neurais usam capacidade massiva: LSTM bidirecional 128+64 (celula 12 ~linha 408-430), CNN com filtros 64->128->256.

- Loss sempre `mse` - com escala comprimida pelo `RobustScaler` e dataset pequeno, **Huber/MAE** sao mais robustos a outliers de 2021-2022 (que sao o evento mais informativo da base e estao sendo penalizados quadraticamente).
- Learning rates fixos: CNN/LSTM 1e-3, Informer 5e-4. Sem warmup, sem cosine decay.
- `recurrent_dropout=0.1` no Bidirectional LSTM (celula 12 ~linha 683) **desliga o kernel cuDNN** silenciosamente - treino lento sem ganho.

- **Recomendacao**: reduzir capacidade (LSTM 32-64, CNN ate 64), trocar loss para Huber, adicionar `ReduceLROnPlateau` e treinar com 5 seeds reportando mediana e desvio.

### A9 - Stacking com `min_samples align` indica bug de shape

Na celula 12 ~linha 1527-1537 ha um alinhamento por `min_samples` antes de empilhar previsoes. Isso indica que CNN/LSTM/Informer estao retornando arrays com tamanhos diferentes em alguns folds - provavelmente porque um deles cuspe `(N,)` e outro `(N,1)`, ou porque H_base nao divide H_total no recursivo. Sintoma de fragilidade nas funcoes recursivas (existem **tres** delas - `previsao_recursiva_backtest`, `previsao_recursiva_com_cenario` e o loop em cell 20 - cada uma com propria logica de re-alimentacao).

- **Recomendacao**: unificar em uma unica funcao `forecast_recursive(model, scaler, X0, H, feature_contract)` testada com asserts de shape em cada passo.

### A10 - Cenarios `Pessimista` colapsam quando chave nao existe

Na celula 13 ~linha 2464-2467 a definicao usa `cenarios.get('Conservador_Desfavoravel', list(cenarios.values())[0])`. Se a chave nao existir (e em algum momento foi renomeada), os tres cenarios viram o mesmo array. A "faixa de incerteza" plotada some sem aviso.

### A11 - Cell 21 quebra reproducibilidade

Usa `date.today().year` para definir `ano_fim` da coleta de validacao externa. Rodar o notebook em meses diferentes muda o resultado e o periodo coberto.

### A12 - NEWAVE NAO aparece em lugar nenhum como baseline comparavel

O TCC inteiro defende que o trabalho vem para superar o NEWAVE. Mas no notebook nao existe nenhuma serie de previsao do NEWAVE para comparar. O `baseline_sazonal` (celula 13 ~linha 722) e media mensal historica - util, mas nao e o NEWAVE. Sem isso, a tese central do TCC ("este modelo supera o NEWAVE") nao tem evidencia.

- **Recomendacao**: importar as previsoes dos decks NEWAVE 2020-2024 (que voce ja extraiu para a primeira metade do TCC) e plotar lado a lado com os modelos neurais e XGBoost no mesmo eixo, mesmo periodo, mesma unidade.

### A13 - Pesos do ensemble fixos privilegiam o modelo mais quebrado

`ensemble_weights = {'CNN': 0.2, 'LSTM': 0.19, 'Informer': 0.61}` (celula 20 ~linha 189). Pesar 61% no Informer com a implementacao da secao A4 e provavel causa raiz dos -3186 MWmed de bias agregado em 2025. LSTM, que esta menos ferido, tem o menor peso (0.19) mas a melhor performance no backtest 2025 (RMSE 2627, bias -1898).

- **Recomendacao**: re-otimizar pesos por minimizacao de MSE em validacao (constrained least squares com `w_i >= 0` e `sum(w_i) = 1`). Apos a correcao do Informer, repetir.

### A14 - Imputacao de NaN com media global vaza tudo

Celula 12 ~linha 122-126: `df_clean[col].fillna(df_clean[col].mean())` sobre o dataset inteiro. Aplica a todas as features meteorologicas. Isso e leak para validacao e teste e foi mencionado no review original mas merece reforco - e provavelmente a fonte mais comum de "boas metricas no backtest, ruins na inferencia real".

### A15 - Faltam features que o proprio TCC menciona

O texto da secao 4 menciona explicitamente que geracao eolica depende de "PIB, IDH" e variaveis socioeconomicas. Nada disso esta no dataset. O modelo recebe so meteorologia + capacidade + carga + calendario.

- **Recomendacao**: adicionar PIB Nordeste, IDH (anual/mensal interpolado), preco de energia (PLD), e idealmente um indice climatico macro (ENSO/AMM/MEI) que e disponivel publico e ajuda muito horizonte longo.

### A16 - Sumario das 5 acoes com maior ROI esperado

1. **Modelar fator de capacidade** (target = geracao/capacidade, multiplicar por capacidade projetada na saida). Sozinha, esta acao deve cortar o bias de longo prazo a quase zero.
2. **Adicionar lags do target** + corrigir loop recursivo para realimentar a previsao no input.
3. **Cenario de capacidade futura crescente** (curva PDE/EPE ou fit linear da tendencia historica) em vez de copiar a ultima linha.
4. **Corrigir o Informer** (ProbSparse Top-u correto + distilling=True + mesma config no backtest e no final) e re-otimizar pesos do ensemble por validacao.
5. **Incluir NEWAVE como baseline real** no comparativo final.

---

## Plano de desenvolvimento para melhorar previsoes

Plano em quatro fases. As fases sao executaveis em paralelo dentro de cada uma, mas a ordem entre fases importa porque cada fase depende do contrato temporal/dados da anterior. Estimativa de prazo assumindo 2-3 dias de trabalho focado por fase.

### Fase 0 - Higiene (bloqueante para qualquer comparacao confiavel)

Sem isso, qualquer numero novo continua sendo apple-vs-orange contra os antigos.

| # | Acao | Arquivo/celula | Criterio de pronto |
|---|---|---|---|
| 0.1 | Rotacionar chave CDS e remover `.cdsapirc` do Git | raiz | `.gitignore` cobre, `git ls-files` nao retorna |
| 0.2 | Substituir `pasta_central` hardcoded por `Path(__file__).parent` ou `Path.cwd()` | celula 1 | notebook roda em outra maquina |
| 0.3 | Renomear `requiremets.txt` -> `requirements.txt` e fixar versoes principais | raiz | `pip install -r` reproduz |
| 0.4 | Mover imputacao de NaN, scaling e clustering para **dentro** de cada fold de backtest | celulas 12, 13 | nenhum `.fit()` de scaler/imputer em dataset full |
| 0.5 | Remover do dataset de modelagem todas as colunas derivadas do target (ou criar 2 datasets: `raw_diagnostico` e `modelagem_clean`) | celula 6, 9 | `assert` em `prepare_data` rejeita feature derivada de target |
| 0.6 | Off-by-one do mes futuro: `future_date = ultimo_mes + DateOffset(months=h+1)` | celula 13 | primeiro ponto previsto = 2025-01 e nao 2024-12 |
| 0.7 | Padronizar unidade em MWmed em todos os graficos e CSVs | celulas 12, 16, 19, 20 | nenhum rotulo "GWh" sem conversao |
| 0.8 | Validar metadados consistentes (`L`, `H`, `W`, `config_key` iguais em `metadata_inferencia.json` e `configuracao_final.json`) | celula 17 | regenerar metadados a partir do treino vencedor |

**Pronto da fase**: pipeline reproduzivel em outra maquina, sem leak na imputacao/scaling, sem feature do target no input.

### Fase 1 - Reformular o target e as features futuras (alto impacto)

Aqui entra a maior parte do ganho de qualidade.

| # | Acao | Detalhe |
|---|---|---|
| 1.1 | Trocar target para **fator de capacidade** (`fc = geracao_mwmed / capacidade_mw`) | manter pipeline em paralelo com target absoluto para comparacao A/B |
| 1.2 | Criar gerador unico de **covariaveis futuras** com regras explicitas por feature | `meteorologia: climatologia_mensal`, `capacidade: curva_projetada`, `carga: extrapolacao_linear`, `regime: tempo_desde_2010` |
| 1.3 | Adicionar lags do target ao input: `geracao_lag_{1,3,6,12}` ou `fc_lag_{1,3,6,12}` | gerados via `df.shift()` em celula 6 |
| 1.4 | Adicionar features de mercado/macro mensais: PIB regional, PLD-NE, indice ENSO/AMM, preco do petroleo | fontes: IBGE, CCEE, NOAA, BCB |
| 1.5 | Adicionar `meses_desde_inicio_setor` como tendencia continua | substitui `regime_operacional` |
| 1.6 | Curva de capacidade futura calibrada (3 cenarios: PDE base, conservador, otimista) | usa Plano Decenal de Energia 2034 ou fit linear da serie historica |
| 1.7 | Garantir que toda feature usada no input tenha plano explicito de projecao no futuro - abortar se faltar | adicionar `assert feature in PROJECAO_FUTURA` em `gerar_cenarios_futuros_tcc` |

**Pronto da fase**: rodar uma previsao de 60 meses **so com Sazonal Naive + capacidade projetada** (sem rede neural). Se essa baseline ja bate o NEWAVE, voce ganhou metade do TCC sem treinar nada.

### Fase 2 - Corrigir arquitetura e treinamento

| # | Acao | Detalhe |
|---|---|---|
| 2.1 | Implementar arquitetura integrada: CNN -> [LSTM, Informer] (em paralelo, recebendo features da CNN) | substitui ramos paralelos atuais |
| 2.2 | Corrigir Informer: ProbSparse Top-u correto OU substituir por Transformer padrao com positional encoding | recomendado: Transformer padrao por simplicidade |
| 2.3 | Garantir mesma config de hiperparametros entre backtest e treino final | parametrizar via `config_key` |
| 2.4 | Reduzir capacidade das redes para o tamanho da base (LSTM 32-64, CNN ate 64) | evitar overfit |
| 2.5 | Trocar `loss=mse` por `loss=Huber(delta=1.0)` em todos os modelos neurais | mais robusto a outliers |
| 2.6 | Adicionar `EarlyStopping(patience=15) + ReduceLROnPlateau + ModelCheckpoint` no treino final | hoje nao tem |
| 2.7 | Treinar com 5 seeds, reportar mediana e desvio | reduzir variancia da estimativa |
| 2.8 | Mover `os.environ['TF_DETERMINISTIC_OPS']='1'` para **antes** de qualquer `import tensorflow` | reproducibilidade |
| 2.9 | Unificar as 3 funcoes recursivas em `forecast_recursive(model, scaler, X0, H, feature_contract)` | adicionar asserts de shape por passo |
| 2.10 | Corrigir CNN para consumir aux ao longo de toda a janela (Conv1D paralela ou concat de canais) | em vez de `aux_last` |

**Pronto da fase**: backtest com `H=12` e `H=24` rodando, criterio de selecao por `val_rmse` (nao test_rmse), teste congelado relatado uma vez no fim.

### Fase 3 - Selecao de modelo, ensemble e baseline NEWAVE

| # | Acao | Detalhe |
|---|---|---|
| 3.1 | Re-otimizar pesos do ensemble por validacao (constrained least squares: `w_i>=0`, `sum=1`) | substitui pesos fixos atuais |
| 3.2 | Corrigir stacking Ridge para operar 100% em um unico espaco (recomendo escalado) | hoje mistura escalado + original |
| 3.3 | Importar previsoes NEWAVE 2020-2024 e plotar mesmo eixo, mesma unidade, mesmo periodo | usar dados que voce ja extraiu na primeira metade do TCC |
| 3.4 | Adicionar baselines simples ao comparativo: Sazonal Naive, Persistencia Sazonal, SARIMAX/ETS, regressao linear com capacidade | mostra que a rede agrega valor |
| 3.5 | Calibrar intervalo de incerteza de verdade: bootstrap dos residuos do backtest, OU MC dropout, OU conformal prediction | substitui `+-10%` arbitrario |
| 3.6 | Usar `eolica_NE_mensal_MWmed_2025-01_ate_hoje.csv` como **validacao externa honesta** | filtrar so meses completos, comparar previsao vs real para cada modelo |
| 3.7 | Tabela final: `Modelo x Horizonte (1m, 6m, 12m, 24m, 60m)` com RMSE, MAE, bias, MAPE | inclui NEWAVE, Sazonal, XGBoost, CNN, LSTM, Informer, Ensemble |

**Pronto da fase**: tabela comparativa robusta, intervalo de incerteza calibrado, NEWAVE como baseline real.

### Fase 4 - Modularizacao e versionamento (manutencao)

| # | Acao | Detalhe |
|---|---|---|
| 4.1 | Refatorar notebook para `src/`: `coleta.py`, `dataset.py`, `features_futuras.py`, `backtest.py`, `train.py`, `forecast.py` | notebook fica como orquestrador |
| 4.2 | Salvar manifesto de cada artefato com `config_key, periodo_treino, lista_features, hash_dataset, unidade, data_geracao, seed` | json ao lado de cada `.keras`/`.pkl` |
| 4.3 | Adicionar suite minima de testes: shape do output, range do FC (`[0, 0.55]`), nao-NaN, lag funcionando | pytest |
| 4.4 | Pipeline de geracao de relatorio markdown final automatico | metricas + figuras + tabelas em um `relatorio_TCC.md` versionado |

**Pronto da fase**: alguem que nao seja voce consegue rodar `python -m src.forecast` e regerar todas as previsoes.

---

### Cronograma sugerido (10 dias uteis)

```
Dia 1-2  Fase 0 (higiene)
Dia 3-5  Fase 1 (target FC + features futuras + cenarios capacidade)
Dia 6-8  Fase 2 (arquitetura + treinamento corrigido)
Dia 9    Fase 3 (ensemble + NEWAVE + tabela final)
Dia 10   Fase 4 (modularizacao minima + relatorio)
```

A fase 1 e a que tem maior chance de mover o ponteiro nas previsoes. Se voce so tiver 3 dias, faca **fase 0 + fase 1 + 3.3 (importar NEWAVE)** e ja tera material muito mais defensavel para o TCC do que o estado atual.

### Criterios objetivos de melhoria

A previsao melhorou se em validacao externa (CSV 2025-01 ate hoje):

- RMSE do Ensemble cai de 3544 para abaixo de 1500 MWmed (melhora >2x).
- Bias agregado fica em [-500, +500] MWmed (hoje -3186).
- MAPE cai de 26% para abaixo de 12%.
- Intervalo de incerteza (5%-95%) cobre o real em pelo menos 90% dos meses.
- Pelo menos um modelo (qualquer um) supera o NEWAVE no mesmo periodo, em pelo menos uma das metricas RMSE/MAE.

Se nao bater nenhum desses, o problema nao e de modelagem - e de dados, e a discussao volta para a fase 1.
