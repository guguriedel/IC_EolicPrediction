"""
Fase 1.7: Contrato de features futuras.

Cada feature usada como input do modelo precisa ter uma regra explicita de como
sera projetada no horizonte futuro. Sem isso, o pipeline de inferencia
escolhe silenciosamente um default (copiar ultima linha, zero, etc) e a
previsao recursiva fica enviesada de forma invisivel.

Contrato:
    PROJECAO_FUTURA: dict[str, str]
        Mapeia regex/nome de feature -> codigo de regra.

Regras suportadas (codigo: descricao):
    'realimentacao_recursiva'  Lag do target. Substituir pelo y_hat anterior.
    'realimentacao_sazonal'    Lag de 12 meses. Substituir pelo y_hat de 12 passos atras.
    'calendario'               Funcao deterministica do tempo. mes_sin, mes_cos, meses_desde_inicio.
    'cenario_capacidade'       Le coluna do CSV capacidade_projetada_ne.csv.
    'derivada_capacidade'      Diff da projecao de capacidade (crescimento_capacidade_mw).
    'climatologia'             Le do CSV climatologia_mensal_meteo.csv (media por mes).
    'extrapolacao_linear'      Fit linear na serie historica e extrapola.
    'congela_ultimo'           Mantem ultimo valor observado (uso desencorajado).
    'sera_zerada'              Feature foi removida; defina valor=0.

Uso (esqueleto):
    from src.features_futuras import PROJECAO_FUTURA, projetar_features_futuras

    df_futuro = projetar_features_futuras(
        features_modelo=lista_de_colunas,
        df_historico=df_treino,
        horizonte=60,
        cenario='base',  # 'conservador' | 'base' | 'otimista'
    )
    # df_futuro tem mesmas colunas que features_modelo, com regra aplicada.
"""
from __future__ import annotations
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DADOS = ROOT / "Dados"

CSV_CAPACIDADE = DADOS / "capacidade_projetada_ne.csv"
CSV_CLIMATOLOGIA = DADOS / "climatologia_mensal_meteo.csv"

# Inicio simbolico do setor eolico (mesmo valor de gerar_dataset_limpo.py)
DATA_INICIO_SETOR = pd.Timestamp("2010-01-01")

# Prefixos meteorologicos (mesma lista de gerar_climatologia.py)
PREFIXOS_METEO = (
    "vel_vento_100m_",
    "t2m_",
    "sp_",
    "tp_",
    "sst_",
    "zust_",
    "densidade_ar_",
)

# === Contrato ===
# Cada item: (regex_da_feature, codigo_regra). Avaliado em ordem; primeiro match vence.
# Mantenha lista pequena e auditavel.
PROJECAO_FUTURA: list[tuple[str, str]] = [
    # Lags do target
    (r"^fc_lag_\d+$", "realimentacao_recursiva"),
    (r"^geracao_lag_\d+$", "realimentacao_recursiva"),
    (r"^fc_diff_\d+$", "realimentacao_recursiva"),
    # Calendario / tendencia deterministica
    (r"^mes_sin$|^mes_cos$|^meses_desde_inicio$|^mes$|^ano$|^trimestre$", "calendario"),
    # Capacidade NE total e derivada
    (r"^capacidade_eolica_ne_mw$", "cenario_capacidade"),
    (r"^crescimento_capacidade_mw$", "derivada_capacidade"),
    # Capacidade por cluster ERA5
    (r"^capacidade_cluster_mw_", "cenario_capacidade_cluster"),
    # Carga / demanda - extrapolacao linear simples (PIB seria melhor, fica para 1.6)
    (r"^carga_ne_mwmed$|^demanda_sin_mwmed$|^demanda_max_sin_mw$", "extrapolacao_linear"),
    # Meteorologia: climatologia mensal
    (r"^vel_vento_100m_|^t2m_|^sp_|^tp_|^sst_|^zust_|^densidade_ar_", "climatologia"),
    # Subsistema: hardcoded NE (string)
    (r"^subsistema$", "constante_str_NE"),
    # === Fase 1.6: features macro ===
    # Nino 3.4 e AMM: indices climaticos. Usar climatologia mensal historica
    # (anomalias medias por mes ~ 0, mas captura sazonalidade residual).
    (r"^nino34_anom$|^amm_sst$", "climatologia_macro"),
    # Lags de Nino 3.4: idem climatologia (lag de uma anomalia ainda e anomalia).
    (r"^nino34_lag_\d+$", "climatologia_macro"),
    # IBC-Br: proxy mensal de PIB. Extrapolacao linear.
    (r"^ibc_br$", "extrapolacao_linear"),
]


class FeatureSemRegraError(ValueError):
    pass


def regra_da_feature(nome: str) -> str | None:
    for padrao, codigo in PROJECAO_FUTURA:
        if re.match(padrao, nome):
            return codigo
    return None


def auditar_features(features: Iterable[str]) -> dict:
    """Para cada feature, retorna a regra ou None. Levanta se houver feature
    sem regra (nao pode ir para inferencia)."""
    out = {f: regra_da_feature(f) for f in features}
    sem_regra = [f for f, r in out.items() if r is None]
    if sem_regra:
        raise FeatureSemRegraError(
            f"Features sem regra de projecao em PROJECAO_FUTURA: {sem_regra}. "
            f"Adicione um padrao em src/features_futuras.PROJECAO_FUTURA ou remova do input do modelo."
        )
    return out


def carregar_capacidade_projetada(cenario: str = "base") -> pd.Series:
    if not CSV_CAPACIDADE.exists():
        raise FileNotFoundError(
            f"{CSV_CAPACIDADE} nao existe. Rode scripts/gerar_capacidade_projetada.py."
        )
    df = pd.read_csv(CSV_CAPACIDADE, parse_dates=["data"]).set_index("data")
    col = f"cenario_{cenario}"
    if col not in df.columns:
        raise ValueError(f"Cenario '{cenario}' invalido. Disponiveis: {list(df.columns)}")
    return df[col]


def carregar_climatologia() -> pd.DataFrame:
    """Retorna DataFrame indexado por (mes, feature) com colunas mean, std."""
    if not CSV_CLIMATOLOGIA.exists():
        raise FileNotFoundError(
            f"{CSV_CLIMATOLOGIA} nao existe. Rode scripts/gerar_climatologia.py."
        )
    return pd.read_csv(CSV_CLIMATOLOGIA).set_index(["mes", "feature"])


def projetar_calendario(datas: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=datas)
    out["mes"] = datas.month
    out["ano"] = datas.year
    out["trimestre"] = datas.quarter
    out["mes_sin"] = np.sin(2 * np.pi * datas.month / 12)
    out["mes_cos"] = np.cos(2 * np.pi * datas.month / 12)
    out["meses_desde_inicio"] = (
        ((datas - DATA_INICIO_SETOR).days / 30.4375).round().astype(int)
    )
    return out


def projetar_climatologia(datas: pd.DatetimeIndex, features_meteo: list[str]) -> pd.DataFrame:
    clim = carregar_climatologia()
    out = pd.DataFrame(index=datas, columns=features_meteo, dtype=float)
    for d in datas:
        m = d.month
        for f in features_meteo:
            try:
                out.at[d, f] = clim.loc[(m, f), "mean"]
            except KeyError:
                out.at[d, f] = np.nan
    return out


def projetar_capacidade_total(datas: pd.DatetimeIndex, cenario: str = "base") -> pd.Series:
    serie = carregar_capacidade_projetada(cenario)
    return serie.reindex(datas)


def projetar_capacidade_clusters(
    datas: pd.DatetimeIndex,
    cluster_features: list[str],
    capacidade_total_futura: pd.Series,
    df_historico: pd.DataFrame,
) -> pd.DataFrame:
    """Projeta capacidade por cluster mantendo a PROPORCAO observada na ultima
    linha historica e escalando pela capacidade total futura."""
    cap_total_hist = df_historico["capacidade_eolica_ne_mw"].iloc[-1]
    proporcoes = df_historico[cluster_features].iloc[-1] / cap_total_hist

    out = pd.DataFrame(index=datas, columns=cluster_features, dtype=float)
    for d in datas:
        cap_total_d = capacidade_total_futura.loc[d]
        out.loc[d] = (proporcoes.values * cap_total_d).astype(float)
    return out


def extrapolacao_linear(
    df_historico: pd.DataFrame, coluna: str, datas_futuras: pd.DatetimeIndex
) -> pd.Series:
    from sklearn.linear_model import LinearRegression
    s = df_historico[[coluna]].dropna()
    t_hist = np.arange(len(s)).reshape(-1, 1)
    m = LinearRegression().fit(t_hist, s.values.ravel())
    t_fut = np.arange(len(s), len(s) + len(datas_futuras)).reshape(-1, 1)
    return pd.Series(m.predict(t_fut), index=datas_futuras, name=coluna)


def climatologia_macro(
    df_historico: pd.DataFrame, coluna: str, datas_futuras: pd.DatetimeIndex
) -> pd.Series:
    """Media mensal historica de uma coluna macro (Nino, AMM, etc).

    Diferente de `projetar_climatologia`, calcula direto do df_historico em
    vez de ler CSV pre-computado. Isso permite incluir features novas
    (como lags de Nino) sem ter que regerar arquivo de climatologia.
    """
    df = df_historico[["data", coluna]].dropna()
    df["data"] = pd.to_datetime(df["data"])
    media_por_mes = df.groupby(df["data"].dt.month)[coluna].mean()
    out = pd.Series(
        [media_por_mes.get(d.month, np.nan) for d in datas_futuras],
        index=datas_futuras,
        name=coluna,
    )
    return out


def projetar_features_futuras(
    features_modelo: list[str],
    df_historico: pd.DataFrame,
    horizonte: int,
    cenario_capacidade: str = "base",
    primeiro_mes: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Gera DataFrame com 'horizonte' linhas e colunas == features_modelo.

    Atencao: lags do target (`realimentacao_recursiva`) ficam preenchidos com
    NaN aqui. Eles devem ser preenchidos pelo loop recursivo de inferencia,
    que conhece y_hat de cada passo.
    """
    auditar_features(features_modelo)

    if primeiro_mes is None:
        ultimo = pd.to_datetime(df_historico["data"].iloc[-1])
        primeiro_mes = ultimo + pd.DateOffset(months=1)
    datas = pd.date_range(primeiro_mes, periods=horizonte, freq="MS")

    out = pd.DataFrame(index=datas, columns=features_modelo, dtype=object)

    cal = projetar_calendario(datas)
    cap_total = projetar_capacidade_total(datas, cenario=cenario_capacidade)

    cluster_cols = [f for f in features_modelo if f.startswith("capacidade_cluster_mw_")]
    if cluster_cols:
        cluster_proj = projetar_capacidade_clusters(
            datas, cluster_cols, cap_total, df_historico
        )
        for c in cluster_cols:
            out[c] = cluster_proj[c]

    meteo_cols = [
        f for f in features_modelo
        if any(f.startswith(p) for p in PREFIXOS_METEO) and f not in cluster_cols
    ]
    if meteo_cols:
        meteo_proj = projetar_climatologia(datas, meteo_cols)
        for c in meteo_cols:
            out[c] = meteo_proj[c]

    # Aplica regras restantes
    for f in features_modelo:
        regra = regra_da_feature(f)
        if regra == "calendario":
            if f in cal.columns:
                out[f] = cal[f].values
        elif regra == "cenario_capacidade":
            out[f] = cap_total.values
        elif regra == "derivada_capacidade":
            cap_extendido = pd.concat(
                [df_historico.set_index("data")[["capacidade_eolica_ne_mw"]].iloc[-1:]
                 .rename(columns={"capacidade_eolica_ne_mw": f}),
                 pd.Series(cap_total.values, index=cap_total.index, name=f).to_frame()]
            )
            out[f] = cap_extendido[f].diff().reindex(datas).values
        elif regra == "extrapolacao_linear":
            out[f] = extrapolacao_linear(df_historico, f, datas).values
        elif regra == "climatologia_macro":
            out[f] = climatologia_macro(df_historico, f, datas).values
        elif regra == "constante_str_NE":
            out[f] = "NE"
        elif regra == "realimentacao_recursiva":
            out[f] = np.nan  # preenchido pelo loop recursivo
        # 'cenario_capacidade_cluster' e 'climatologia' ja preenchidos acima

    out.index.name = "data"
    return out
