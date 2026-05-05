"""Fase 2 - Orquestrador completo.

Para cada arquitetura {lstm_solo, transformer_solo, integrado}:
  1. Treina 5 seeds em (train=ate 2022, val=2023, test=2024)
  2. Calcula RMSE/MAE em escala original FC E reconstruido em MWmed
  3. Seleciona melhor seed por VAL (Fase 0.5/2.5: nunca por test)
  4. Re-treina o melhor modelo em train+val+test (todo historico) para forecast
  5. Gera forecast recursivo de 60 meses 2025-2029 com cenario conservador
  6. Compara contra real 2025-2026 e contra baseline_sazonal

Saidas em Modelos/fase2/:
  resultados_treino.csv         - todas as seeds, todas as arquiteturas
  selecao_modelo.json            - melhor por val
  forecast_60meses.csv           - previsoes recursivas
  validacao_externa_2025.csv     - comparativo modelos x real x baseline
  validacao_externa_2025.png     - plot

Uso:
    .venv/Scripts/python.exe scripts/treinar_fase2.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

# Determinismo ANTES de import tensorflow
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.seeds import configurar_ambiente_determinismo
configurar_ambiente_determinismo()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_prep import preparar_tudo, listar_features, carregar_dataset, fitar_scalers, imputar_treino_e_aplicar
from src.models import HiperParams
from src.train_eval import treinar_multi_seed, treinar_uma_seed, MODELOS_DIR
from src.features_futuras import projetar_features_futuras, carregar_capacidade_projetada
from src.forecast_recursive import forecast_recursive

OUT_DIR = MODELOS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)
DADOS = ROOT / "Dados"
CSV_REAL_2025 = ROOT / "eolica_NE_mensal_MWmed_2025-01_ate_hoje.csv"
CSV_BASELINE = ROOT / "Modelos" / "Comparacoes" / "Baseline_Sazonal" / "previsoes_baseline_sazonal_60m.csv"

ARQUITETURAS = ["lstm_solo", "transformer_solo", "integrado"]
SEEDS = [42, 1, 2, 3, 4]
HP = HiperParams(L=24, H=12)
FIM_TREINO = pd.Timestamp("2022-12-01")
FIM_VAL = pd.Timestamp("2023-12-01")
CENARIO_CAP = "conservador"


def reconstroi_geracao_mwmed(fc_scaled, scaler_y, datas, capacidade_serie):
    fc = scaler_y.inverse_transform(fc_scaled.reshape(-1, 1)).ravel()
    fc = np.clip(fc, 0.0, 0.7)
    cap = np.array([capacidade_serie.loc[d] for d in datas])
    return fc * cap


def metricas(y_true, y_pred):
    err = y_pred - y_true
    return dict(
        rmse=float(np.sqrt(np.mean(err ** 2))),
        mae=float(np.mean(np.abs(err))),
        bias=float(np.mean(err)),
        mape=float(np.mean(np.abs(err / y_true)) * 100),
    )


def main() -> int:
    print("=" * 70)
    print("FASE 2 - Treino com FC, arquiteturas integradas, multi-seed")
    print("=" * 70)

    # === 1. Preparar dados ===
    print(f"\n[1] Preparando dados (L={HP.L}, H={HP.H}, fim_treino={FIM_TREINO.date()}, fim_val={FIM_VAL.date()})")
    split, seqs = preparar_tudo(FIM_TREINO, FIM_VAL, HP.L, HP.H)
    print(f"    features ativas: {len(split.features)}")
    print(f"    seqs train/val/test: {len(seqs['X_train'])} / {len(seqs['X_val'])} / {len(seqs['X_test'])}")
    if len(seqs['X_train']) < 10:
        print(f"    AVISO: poucas sequencias de treino")

    # === 2. Treino multi-seed por arquitetura ===
    todas_resultados = []
    melhores_modelos = {}  # arquitetura -> (modelo, resultado_df_row)
    for arq in ARQUITETURAS:
        print(f"\n[2] Treinando {arq} (5 seeds)")
        modelos, df_res = treinar_multi_seed(arq, HP, seqs, SEEDS, verbose=0)
        df_res["arquitetura"] = arq
        todas_resultados.append(df_res)
        idx_best = df_res["val_rmse_scaled"].idxmin()
        melhores_modelos[arq] = (modelos[idx_best], df_res.loc[idx_best].to_dict())
        print(f"    -> melhor seed por val: {df_res.loc[idx_best, 'seed']} "
              f"(val_rmse={df_res.loc[idx_best, 'val_rmse_scaled']:.4f})")

    df_resultados = pd.concat(todas_resultados, ignore_index=True)
    df_resultados.to_csv(OUT_DIR / "resultados_treino.csv", index=False)
    print(f"\n[OK] resultados_treino.csv -> {OUT_DIR}")

    # === 3. Avaliacao agregada ===
    print("\n[3] Resumo (mediana de 5 seeds, escalado por RobustScaler):")
    resumo = (df_resultados.groupby("arquitetura")
              .agg(val_rmse_med=("val_rmse_scaled", "median"),
                   val_rmse_std=("val_rmse_scaled", "std"),
                   test_rmse_med=("test_rmse_scaled", "median"),
                   test_rmse_std=("test_rmse_scaled", "std")))
    print(resumo.round(4).to_string())

    # === 4. Seleciona arquitetura vencedora por val ===
    arq_vencedora = resumo["val_rmse_med"].idxmin()
    print(f"\n[4] Arquitetura vencedora (menor val_rmse mediano): {arq_vencedora}")

    # === 5. Re-treina vencedora em TODO historico (sem split) com seed vencedora ===
    print("\n[5] Re-treinando vencedora em todo o historico (2015-2024) para forecast 60m")
    df_full = carregar_dataset()
    features = listar_features(df_full)
    df_full_imp = imputar_treino_e_aplicar(df_full, df_full.iloc[:0], df_full.iloc[:0], features)[0]
    scaler_X_full, scaler_y_full = fitar_scalers(df_full_imp, features)

    # Constroi sequencias com TODO historico para treino final
    from src.data_prep import construir_sequencias
    X_full, y_full, _ = construir_sequencias(df_full_imp, features, scaler_X_full, scaler_y_full, HP.L, HP.H)
    seqs_full = dict(
        X_train=X_full, y_train=y_full, datas_train=[],
        X_val=np.empty((0, HP.L, len(features))), y_val=np.empty((0, HP.H)), datas_val=[],
        X_test=np.empty((0, HP.L, len(features))), y_test=np.empty((0, HP.H)), datas_test=[],
    )
    seed_vencedora = int(melhores_modelos[arq_vencedora][1]["seed"])
    modelo_final, _ = treinar_uma_seed(arq_vencedora, HP, seqs_full, seed_vencedora, verbose=0)
    modelo_final.save(OUT_DIR / f"modelo_final_{arq_vencedora}.keras")
    print(f"    salvo: modelo_final_{arq_vencedora}.keras (seed={seed_vencedora})")

    # === 6. Forecast recursivo 60 meses ===
    print("\n[6] Gerando forecast recursivo 60 meses (cenario capacidade conservador)")
    df_futuro_base = projetar_features_futuras(
        features_modelo=features,
        df_historico=df_full_imp,
        horizonte=60,
        cenario_capacidade=CENARIO_CAP,
    )
    cap_serie = carregar_capacidade_projetada(CENARIO_CAP)
    resultado = forecast_recursive(
        model=modelo_final,
        scaler_X=scaler_X_full,
        scaler_y=scaler_y_full,
        df_historico=df_full_imp,
        df_futuro_base=df_futuro_base,
        features=features,
        L=HP.L,
        H_total=60,
        capacidade_projetada=cap_serie,
    )
    df_forecast = pd.DataFrame({
        "data": resultado.datas,
        "fc_previsto": resultado.fc_previsto,
        "geracao_mwmed_previsto": resultado.geracao_mwmed_previsto,
        "capacidade_mw_usada": resultado.capacidade_usada,
        "modelo": arq_vencedora,
        "seed": seed_vencedora,
    })
    df_forecast.to_csv(OUT_DIR / "forecast_60meses.csv", index=False)
    print(f"    forecast_60meses.csv ({len(df_forecast)} meses)")

    # === 7. Validacao externa contra real 2025 ===
    print("\n[7] Validacao externa contra real 2025-2026")
    if not CSV_REAL_2025.exists():
        print(f"    AVISO: {CSV_REAL_2025} nao encontrado, pulando.")
        return 0

    df_real = pd.read_csv(CSV_REAL_2025)
    # Normalizar nome de coluna de data
    col_data = next((c for c in df_real.columns if "data" in c.lower() or "mes" in c.lower()), df_real.columns[0])
    df_real[col_data] = pd.to_datetime(df_real[col_data])
    col_val = next((c for c in df_real.columns if "mwmed" in c.lower() or "geracao" in c.lower()), df_real.columns[1])
    df_real = df_real.rename(columns={col_data: "data", col_val: "real_mwmed"})[["data", "real_mwmed"]]

    df_merge = df_forecast.merge(df_real, on="data", how="inner")
    if CSV_BASELINE.exists():
        df_base = pd.read_csv(CSV_BASELINE, parse_dates=["data"])
        col_base = next((c for c in df_base.columns if "conservador" in c.lower()), None)
        if col_base:
            df_merge = df_merge.merge(df_base[["data", col_base]].rename(columns={col_base: "baseline_sazonal"}),
                                       on="data", how="left")
            print(f"    baseline_sazonal: coluna usada = {col_base}")

    print(f"    meses em comum: {len(df_merge)}")
    if len(df_merge) > 0:
        m_modelo = metricas(df_merge["real_mwmed"].values, df_merge["geracao_mwmed_previsto"].values)
        print(f"    {arq_vencedora:20s}  RMSE={m_modelo['rmse']:7.1f}  MAE={m_modelo['mae']:7.1f}  Bias={m_modelo['bias']:+7.1f}  MAPE={m_modelo['mape']:.1f}%")
        if "baseline_sazonal" in df_merge.columns:
            m_base = metricas(df_merge["real_mwmed"].values, df_merge["baseline_sazonal"].values)
            print(f"    baseline_sazonal     RMSE={m_base['rmse']:7.1f}  MAE={m_base['mae']:7.1f}  Bias={m_base['bias']:+7.1f}  MAPE={m_base['mape']:.1f}%")

        df_merge.to_csv(OUT_DIR / "validacao_externa_2025.csv", index=False)

        # Plot
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(df_merge["data"], df_merge["real_mwmed"], "k-", lw=2, label="Real (ONS)")
        ax.plot(df_merge["data"], df_merge["geracao_mwmed_previsto"], "b-", label=f"{arq_vencedora} (Fase 2)")
        if "baseline_sazonal" in df_merge.columns:
            ax.plot(df_merge["data"], df_merge["baseline_sazonal"], "g--", label="Baseline Sazonal (Fase 1)")
        ax.set_ylabel("Geracao eolica NE (MWmed)")
        ax.set_title(f"Validacao externa 2025-2026 - Fase 2 ({arq_vencedora}, seed={seed_vencedora})")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "validacao_externa_2025.png", dpi=120)
        plt.close()
        print(f"    plot: validacao_externa_2025.png")

    # === 8. Selecao final salva ===
    selecao = {
        "arquitetura_vencedora": arq_vencedora,
        "seed_vencedora": seed_vencedora,
        "criterio": "menor val_rmse mediano (5 seeds)",
        "hp": HP.__dict__,
        "split": dict(fim_treino=str(FIM_TREINO.date()), fim_val=str(FIM_VAL.date())),
        "cenario_capacidade": CENARIO_CAP,
        "n_features": len(features),
    }
    (OUT_DIR / "selecao_modelo.json").write_text(json.dumps(selecao, indent=2, ensure_ascii=False))
    print(f"\n[OK] tudo salvo em: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
