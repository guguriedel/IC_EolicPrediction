"""
Fase 1.8: Baseline Sazonal Naive de FC x capacidade projetada.

Este e o "smoke test" da Fase 1: se a baseline simples ja bate o Ensemble do
notebook (RMSE 3544 MWmed em 2025) entao a hipotese central da Fase 1 esta
correta - o problema do pipeline atual NAO e arquitetura, e tratamento do
target e de features futuras.

Modelo:
    fc_climatologico[mes] = media historica de fc_ne quando data.month == mes
    geracao_prev[t] = fc_climatologico[t.month] * capacidade_projetada[t, cenario]

Avaliacao: contra eolica_NE_mensal_MWmed_2025-01_ate_hoje.csv (14 meses).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DADOS = ROOT / "Dados"
MODELOS_COMP = ROOT / "Modelos" / "Comparacoes"
OUT_DIR = MODELOS_COMP / "Baseline_Sazonal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_HIST = DADOS / "dataset_modelagem_limpo.csv"
CSV_REAL_2025 = ROOT / "eolica_NE_mensal_MWmed_2025-01_ate_hoje.csv"
CSV_CAPACIDADE = DADOS / "capacidade_projetada_ne.csv"
CSV_PREVISOES_NOTEBOOK = MODELOS_COMP / "Previsoes_60_Meses" / "previsoes_60_meses_todos_modelos.csv"


def calcular_metricas(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    return {
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(np.abs(err).mean()),
        "bias": float(err.mean()),
        "mape": float((np.abs(err) / y_true).mean() * 100),
        "n": int(len(y_true)),
    }


def main() -> int:
    df_hist = pd.read_csv(CSV_HIST, parse_dates=["data"]).sort_values("data").reset_index(drop=True)
    df_cap = pd.read_csv(CSV_CAPACIDADE, parse_dates=["data"]).set_index("data")
    df_real = pd.read_csv(CSV_REAL_2025)
    df_real["data"] = pd.to_datetime(df_real["mes"] + "-01")
    df_real = df_real.rename(columns={"mwmed": "real"})[["data", "real"]].set_index("data")

    # === Climatologia FC ===
    df_hist["mes"] = df_hist["data"].dt.month
    fc_clim = df_hist.groupby("mes")["fc_ne"].agg(["mean", "std", "count"])
    print("Climatologia FC por mes (historico 2015-2024):")
    print(fc_clim.round(4))

    # === Previsoes para todos os 60 meses, todos os 3 cenarios ===
    datas_previstas = pd.date_range("2025-01-01", periods=60, freq="MS")
    previsoes = pd.DataFrame(index=datas_previstas)
    previsoes.index.name = "data"
    for cenario in ["conservador", "base", "otimista"]:
        cap = df_cap[f"cenario_{cenario}"]
        meses = previsoes.index.month
        fc_prev = fc_clim.loc[meses, "mean"].values
        cap_prev = cap.reindex(datas_previstas).values
        previsoes[f"sazonal_{cenario}"] = fc_prev * cap_prev

    # === Avaliacao contra real 2025 (14 meses disponiveis) ===
    eval_df = previsoes.join(df_real, how="inner")
    print(f"\nAvaliacao em {len(eval_df)} meses ({eval_df.index.min()} a {eval_df.index.max()}):")
    metricas = {}
    for col in ["sazonal_conservador", "sazonal_base", "sazonal_otimista"]:
        m = calcular_metricas(eval_df["real"].values, eval_df[col].values)
        metricas[col] = m
        print(f"  {col}:  RMSE={m['rmse']:7.1f}  MAE={m['mae']:7.1f}  Bias={m['bias']:+7.1f}  MAPE={m['mape']:5.1f}%")

    # === Comparativo com modelos do notebook ===
    if CSV_PREVISOES_NOTEBOOK.exists():
        df_nb = pd.read_csv(CSV_PREVISOES_NOTEBOOK, parse_dates=["Data"]).set_index("Data")
        df_nb.index.name = "data"
        comparativo = eval_df.join(df_nb, how="inner")
        print(f"\nComparativo com modelos do notebook ({len(comparativo)} meses):")
        for col in ["CNN", "LSTM", "Informer", "XGBoost", "Ensemble"]:
            if col in comparativo.columns:
                m = calcular_metricas(comparativo["real"].values, comparativo[col].values)
                metricas[f"notebook_{col}"] = m
                print(f"  {col:25s}:  RMSE={m['rmse']:7.1f}  MAE={m['mae']:7.1f}  Bias={m['bias']:+7.1f}  MAPE={m['mape']:5.1f}%")

    # === Salvar ===
    previsoes.to_csv(OUT_DIR / "previsoes_baseline_sazonal_60m.csv")
    eval_df.to_csv(OUT_DIR / "avaliacao_baseline_sazonal_2025.csv")
    pd.DataFrame(metricas).T.to_csv(OUT_DIR / "metricas_baseline_vs_notebook.csv")

    # === Plot ===
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(eval_df.index, eval_df["real"], label="Real (ONS)", color="black", linewidth=2.5, marker="o")
    for col, color in [("sazonal_conservador", "C2"), ("sazonal_base", "C0"), ("sazonal_otimista", "C1")]:
        ax.plot(eval_df.index, eval_df[col], label=col, color=color, alpha=0.8, marker="s", markersize=4)
    if CSV_PREVISOES_NOTEBOOK.exists():
        for col, color in [("Ensemble", "C3"), ("XGBoost", "C5"), ("LSTM", "C4")]:
            if col in comparativo.columns:
                ax.plot(comparativo.index, comparativo[col], label=f"Notebook {col}", linestyle="--", color=color, alpha=0.6)
    ax.set_title("Baseline Sazonal Naive vs modelos do notebook (validacao externa 2025-2026)")
    ax.set_xlabel("Data")
    ax.set_ylabel("Geracao eolica NE (MWmed)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "baseline_vs_notebook.png", dpi=120)
    plt.close(fig)

    json.dump(
        {
            "n_meses_eval": int(len(eval_df)),
            "periodo_eval": [str(eval_df.index.min().date()), str(eval_df.index.max().date())],
            "metricas": metricas,
            "fc_climatologia": fc_clim["mean"].to_dict(),
        },
        (OUT_DIR / "baseline_sazonal_relatorio.json").open("w"),
        indent=2,
        default=str,
    )

    print(f"\nOK: artefatos em {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
