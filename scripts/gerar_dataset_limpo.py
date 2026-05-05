"""
Gera dataset_modelagem_limpo.csv a partir do bruto.

Mudancas vs dataset_final_modelagem.csv:
  Fase 0.6 - Remove colunas DERIVADAS DO TARGET (vazamento estrutural):
      corte_eolica_ne_mwmed, capacity_factor_ne (substituido por fc_ne),
      penetracao_eolica_ne, penetracao_eolica_sin
  Fase 0.6 - Adiciona fc_ne em fracao [0,1] (target alternativo da Fase 1).
  Fase 0.6 - Detecta clusters com >=50% de NaN e remove (cluster fantasma).
  Fase 1.3 - Adiciona fc_lag_{1,2,3,6,12}, geracao_lag_{1,3,6,12}, fc_diff_1.
  Fase 1.5 - Adiciona meses_desde_inicio (substitui regime_operacional).
  Fase 1.5 - Adiciona mes_sin / mes_cos (caso ja nao existam).

  Salva tambem dataset_diagnostico_completo.csv com TODAS as colunas
  (incluindo derivadas do target) para EDA exploratoria - NUNCA modelagem.

Uso:
    python scripts/gerar_dataset_limpo.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DADOS = ROOT / "Dados"
INPUT_CSV = DADOS / "dataset_final_modelagem.csv"
INPUT_MACRO = DADOS / "features_macro.csv"  # Fase 1.6 - opcional
OUT_DIAGNOSTICO = DADOS / "dataset_diagnostico_completo.csv"
OUT_LIMPO = DADOS / "dataset_modelagem_limpo.csv"
OUT_RELATORIO = DADOS / "dataset_limpo_relatorio.json"

COLS_DERIVADAS_DO_TARGET = [
    "corte_eolica_ne_mwmed",       # heuristica usando geracao
    "capacity_factor_ne",          # proporcional ao target
    "penetracao_eolica_ne",        # proporcional ao target
    "penetracao_eolica_sin",       # proporcional ao target
]

# Limite de missing por coluna para considerar removivel (cluster fantasma)
LIMITE_NAN_PCT = 0.50

# Inicio simbolico do setor eolico brasileiro de larga escala (PROINFA, ~2010)
DATA_INICIO_SETOR = pd.Timestamp("2010-01-01")

# Lags adicionados como features (em meses)
LAGS_FC = (1, 2, 3, 6, 12)
LAGS_GERACAO = (1, 3, 6, 12)


def detectar_clusters_problematicos(df: pd.DataFrame) -> list[str]:
    """Retorna prefixos de clusters com >=LIMITE_NAN_PCT NaN no periodo todo."""
    nan_pct = df.isna().mean()
    suspeitas = nan_pct[nan_pct >= LIMITE_NAN_PCT].index.tolist()
    # Extrair sufixos lat*_lon* dos suspeitos
    sufixos = set()
    for col in suspeitas:
        # Pega tudo apos o ultimo "_lat" - heuristica para clusters
        if "_lat" in col:
            sufixo = "_lat" + col.split("_lat", 1)[1]
            sufixos.add(sufixo)
    return sorted(sufixos)


def main() -> int:
    if not INPUT_CSV.exists():
        print(f"ERRO: {INPUT_CSV} nao encontrado", file=sys.stderr)
        return 1
    df = pd.read_csv(INPUT_CSV)
    df["data"] = pd.to_datetime(df["data"])
    df = df.sort_values("data").reset_index(drop=True)

    relatorio: dict = {
        "input": str(INPUT_CSV),
        "linhas": len(df),
        "colunas_originais": len(df.columns),
        "periodo": [str(df["data"].min().date()), str(df["data"].max().date())],
    }

    # === Diagnostico completo (mantem tudo, marca origem) ===
    df.to_csv(OUT_DIAGNOSTICO, index=False)
    relatorio["diagnostico_path"] = str(OUT_DIAGNOSTICO)

    # === Versao limpa para modelagem ===
    df_limpo = df.copy()

    # 1. Adiciona fc_ne em fracao (target alternativo da Fase 1)
    fc_em_fracao = df_limpo["geracao_eolica_ne_mwmed"] / df_limpo["capacidade_eolica_ne_mw"]
    if not fc_em_fracao.between(0, 1).all():
        out_of_range = fc_em_fracao[~fc_em_fracao.between(0, 1)]
        print(f"AVISO: {len(out_of_range)} meses com FC fora de [0,1]:")
        print(out_of_range.head())
    df_limpo.insert(
        df_limpo.columns.get_loc("geracao_eolica_ne_mwmed") + 1,
        "fc_ne",
        fc_em_fracao,
    )
    relatorio["fc_ne_stats"] = {
        "min": float(fc_em_fracao.min()),
        "max": float(fc_em_fracao.max()),
        "mean": float(fc_em_fracao.mean()),
        "std": float(fc_em_fracao.std()),
    }

    # 2. Remove colunas derivadas do target
    cols_removidas_alvo = [c for c in COLS_DERIVADAS_DO_TARGET if c in df_limpo.columns]
    df_limpo = df_limpo.drop(columns=cols_removidas_alvo)
    relatorio["colunas_removidas_derivadas_target"] = cols_removidas_alvo

    # 3. Detecta e remove clusters com NaN excessivo
    sufixos_problematicos = detectar_clusters_problematicos(df)
    cols_cluster_remover = []
    for suf in sufixos_problematicos:
        cols_cluster_remover.extend([c for c in df_limpo.columns if c.endswith(suf)])
    df_limpo = df_limpo.drop(columns=cols_cluster_remover)
    relatorio["clusters_removidos_nan_excessivo"] = sufixos_problematicos
    relatorio["colunas_cluster_removidas"] = sorted(cols_cluster_remover)

    # === Fase 1.3: lags do target (FC e geracao absoluta) ===
    for lag in LAGS_FC:
        df_limpo[f"fc_lag_{lag}"] = df_limpo["fc_ne"].shift(lag)
    for lag in LAGS_GERACAO:
        df_limpo[f"geracao_lag_{lag}"] = df_limpo["geracao_eolica_ne_mwmed"].shift(lag)
    # diferenca m-a-m do FC (sinal de variacao)
    df_limpo["fc_diff_1"] = df_limpo["fc_ne"].diff(1)
    relatorio["lags_adicionados"] = {
        "fc_lags": list(LAGS_FC),
        "geracao_lags": list(LAGS_GERACAO),
        "outros": ["fc_diff_1"],
    }

    # === Fase 1.5: meses_desde_inicio (substitui regime_operacional categorico) ===
    delta_dias = (df_limpo["data"] - DATA_INICIO_SETOR).dt.days
    df_limpo["meses_desde_inicio"] = (delta_dias / 30.4375).round().astype(int)
    relatorio["meses_desde_inicio"] = {
        "min": int(df_limpo["meses_desde_inicio"].min()),
        "max": int(df_limpo["meses_desde_inicio"].max()),
        "ref_inicio": str(DATA_INICIO_SETOR.date()),
    }

    # === Calendario ciclico (Fase 1.5) - so adiciona se ainda nao existe ===
    if "mes_sin" not in df_limpo.columns:
        df_limpo["mes_sin"] = np.sin(2 * np.pi * df_limpo["data"].dt.month / 12)
        df_limpo["mes_cos"] = np.cos(2 * np.pi * df_limpo["data"].dt.month / 12)
        relatorio["calendario_ciclico_adicionado"] = True
    else:
        relatorio["calendario_ciclico_adicionado"] = False

    # === Fase 1.6: merge features macro (climaticas/socioeconomicas) ===
    if INPUT_MACRO.exists():
        df_macro = pd.read_csv(INPUT_MACRO, parse_dates=["data"])
        antes = df_limpo.columns.tolist()
        df_limpo = df_limpo.merge(df_macro, on="data", how="left")
        novas = [c for c in df_limpo.columns if c not in antes]
        relatorio["macro_merge"] = {
            "fonte": str(INPUT_MACRO),
            "colunas_adicionadas": novas,
            "nan_por_coluna": {c: int(df_limpo[c].isna().sum()) for c in novas},
        }
        # Lags de Nino 3.4 (impacto defasado em regimes de vento)
        if "nino34_anom" in df_limpo.columns:
            for lag in (3, 6, 12):
                df_limpo[f"nino34_lag_{lag}"] = df_limpo["nino34_anom"].shift(lag)
            relatorio["macro_merge"]["lags_nino34"] = [3, 6, 12]
    else:
        relatorio["macro_merge"] = {
            "fonte": str(INPUT_MACRO),
            "skipped": "arquivo nao existe - rode scripts/coletar_features_macro.py",
        }

    # 4. NaN restante (deve ser pequeno) - apenas reporta. Imputacao fica nos folds.
    nan_restante = df_limpo.isna().sum()
    nan_restante = nan_restante[nan_restante > 0]
    relatorio["nan_restante_por_coluna"] = nan_restante.to_dict()
    relatorio["nan_restante_total"] = int(df_limpo.isna().sum().sum())

    # 5. Salva
    df_limpo.to_csv(OUT_LIMPO, index=False)
    relatorio["limpo_path"] = str(OUT_LIMPO)
    relatorio["limpo_linhas"] = len(df_limpo)
    relatorio["limpo_colunas"] = len(df_limpo.columns)

    OUT_RELATORIO.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False))
    print(json.dumps(relatorio, indent=2, ensure_ascii=False))
    print(f"\nOK: {OUT_LIMPO} ({len(df_limpo)} linhas, {len(df_limpo.columns)} colunas)")
    print(f"OK: {OUT_DIAGNOSTICO} (preservado para EDA)")
    print(f"OK: {OUT_RELATORIO}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
