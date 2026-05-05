"""Fase 3 - Ensemble otimizado, baselines simples, conformal IC, tabela final.

Saidas em Modelos/fase3/:
  - previsoes_60m_todos.csv         (todos os modelos lado a lado)
  - validacao_externa_2025.csv      (real vs cada modelo)
  - metricas_global.csv             (RMSE/MAE/Bias/MAPE por modelo, validacao 2025-26)
  - metricas_por_horizonte.csv      (mesmas metricas por bucket de horizonte)
  - ensemble_pesos.json             (pesos LOO + LS in-sample)
  - intervalo_conformal.csv         (mes, p05, mediana, p95)
  - tabela_final.md
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.baselines import persistencia_sazonal_fc, regressao_linear_fc, sarimax_fc, fc_para_mwmed
from src.ensemble import otimiza_pesos_ls, loo_pesos, conformal_quantile, metricas

OUT = ROOT / "Modelos" / "fase3"
OUT.mkdir(parents=True, exist_ok=True)

DATA_INICIO = pd.Timestamp("2025-01-01")
HORIZONTE = 60
DATAS_60M = pd.date_range(DATA_INICIO, periods=HORIZONTE, freq="MS")


def carregar_real() -> pd.DataFrame:
    """Carrega ONS mensal e descarta meses incompletos.

    Filtros:
      - mwmed < 1000 MWmed: clarissimamente parcial (NE nunca gera tao pouco em mes inteiro);
      - data >= mes corrente: o ONS so consolida apos o mes terminar.
    """
    df = pd.read_csv(ROOT / "eolica_NE_mensal_MWmed_2025-01_ate_hoje.csv")
    df["data"] = pd.to_datetime(df["mes"] + "-01")
    n_orig = len(df)

    # Filtro 1: meses com geracao absurdamente baixa (parcial)
    LIMITE_MIN = 1000.0
    parciais_baixos = df[df["mwmed"] < LIMITE_MIN]
    if len(parciais_baixos) > 0:
        print(f"    [aviso] descartando {len(parciais_baixos)} meses com mwmed < {LIMITE_MIN:.0f} (parciais):")
        for _, r in parciais_baixos.iterrows():
            print(f"        {r['mes']}: {r['mwmed']:.1f} MWmed")
        df = df[df["mwmed"] >= LIMITE_MIN]

    # Filtro 2: mes corrente (ONS so consolida apos o mes acabar)
    inicio_mes_corrente = pd.Timestamp.today().to_period("M").to_timestamp()
    futuros = df[df["data"] >= inicio_mes_corrente]
    if len(futuros) > 0:
        print(f"    [aviso] descartando {len(futuros)} meses >= {inicio_mes_corrente.date()} (mes corrente nao consolidado)")
        df = df[df["data"] < inicio_mes_corrente]

    print(f"    {len(df)}/{n_orig} meses considerados como ground truth.")
    return df[["data", "mwmed"]].rename(columns={"mwmed": "real"})


def carregar_modelos_existentes() -> pd.DataFrame:
    """Coleta todas as previsoes ja existentes em wide format (data x modelo).

    Cada fonte e' opcional: se o CSV nao existir, apenas avisa e segue. O ensemble
    LOO-CV depois filtra candidatos ausentes via `candidatos = [c for c in candidatos
    if c in val.columns]`, entao rodar com um subconjunto e' valido.
    """
    out = pd.DataFrame({"data": DATAS_60M})

    # Notebook (CNN, LSTM, Informer, XGBoost, Ensemble notebook) - cells 23-31 do notebook antigo
    p_nb_path = ROOT / "Modelos/Comparacoes/Previsoes_60_Meses/previsoes_60_meses_todos_modelos.csv"
    if p_nb_path.exists():
        p_nb = pd.read_csv(p_nb_path, parse_dates=["Data"]).rename(columns={"Data": "data"})
        out = out.merge(p_nb, on="data", how="left")
        out = out.rename(columns={"Ensemble": "Ensemble_Notebook"})
    else:
        print(f"    [aviso] {p_nb_path.name} ausente - pulando previsoes do notebook (CNN/LSTM/Informer/XGBoost/Ensemble)")

    # Baseline sazonal (Fase 1)
    p_bs_path = ROOT / "Modelos/Comparacoes/Baseline_Sazonal/previsoes_baseline_sazonal_60m.csv"
    if p_bs_path.exists():
        p_bs = pd.read_csv(p_bs_path, parse_dates=["data"])
        out = out.merge(
            p_bs[["data", "sazonal_conservador"]].rename(columns={"sazonal_conservador": "Sazonal_Naive"}),
            on="data", how="left"
        )
    else:
        print(f"    [aviso] {p_bs_path.name} ausente - pulando Sazonal_Naive")

    # Integrado (Fase 2)
    p_int_path = ROOT / "Modelos/fase2/forecast_60meses.csv"
    if p_int_path.exists():
        p_int = pd.read_csv(p_int_path, parse_dates=["data"])
        out = out.merge(
            p_int[["data", "geracao_mwmed_previsto"]].rename(columns={"geracao_mwmed_previsto": "Integrado_Fase2"}),
            on="data", how="left"
        )
    else:
        print(f"    [aviso] {p_int_path.name} ausente - pulando Integrado_Fase2")

    return out


def gerar_baselines_fase3(df_hist: pd.DataFrame) -> pd.DataFrame:
    """Persistencia sazonal, regressao linear, SARIMAX."""
    out = pd.DataFrame({"data": DATAS_60M})

    print("  - Persistencia sazonal..."); fc = persistencia_sazonal_fc(df_hist, DATAS_60M)
    out["Persistencia_Sazonal"] = fc_para_mwmed(fc).values

    print("  - Regressao linear..."); fc = regressao_linear_fc(df_hist, DATAS_60M)
    out["Regressao_Linear"] = fc_para_mwmed(fc).values

    print("  - SARIMAX..."); fc = sarimax_fc(df_hist, DATAS_60M)
    out["SARIMAX"] = fc_para_mwmed(fc).values
    return out


def main():
    print("[1] Carregando dados...")
    df_hist = pd.read_csv(ROOT / "Dados/dataset_modelagem_limpo.csv", parse_dates=["data"])
    df_real = carregar_real()
    print(f"    historico: {len(df_hist)} meses ate {df_hist['data'].max().date()}")
    print(f"    real externo: {len(df_real)} meses {df_real['data'].min().date()}..{df_real['data'].max().date()}")

    print("[2] Coletando previsoes existentes (notebook + Sazonal + Integrado)...")
    pred_existentes = carregar_modelos_existentes()
    print(f"    colunas: {[c for c in pred_existentes.columns if c != 'data']}")

    print("[3] Gerando baselines novos (Fase 3)...")
    pred_novos = gerar_baselines_fase3(df_hist)

    todos = pred_existentes.merge(pred_novos, on="data", how="left")
    todos.to_csv(OUT / "previsoes_60m_todos.csv", index=False)

    # ----- Validacao externa -----
    val = todos.merge(df_real, on="data", how="inner").sort_values("data").reset_index(drop=True)
    val.to_csv(OUT / "validacao_externa_2025.csv", index=False)
    print(f"[4] Validacao externa: {len(val)} meses sobrepostos com real")

    modelos_avaliar = [c for c in val.columns if c not in ("data", "real")]

    # ----- Metricas globais -----
    print("[5] Calculando metricas globais...")
    rows = []
    for m in modelos_avaliar:
        d = val.dropna(subset=[m, "real"])
        if len(d) == 0: continue
        met = metricas(d["real"].values, d[m].values)
        met["Modelo"] = m; met["N"] = len(d)
        rows.append(met)
    df_glob = pd.DataFrame(rows)[["Modelo", "N", "RMSE", "MAE", "Bias", "MAPE"]].sort_values("RMSE")
    df_glob.to_csv(OUT / "metricas_global.csv", index=False)
    print(df_glob.to_string(index=False))

    # ----- Metricas por horizonte -----
    print("\n[6] Metricas por bucket de horizonte...")
    val["h"] = ((val["data"] - DATA_INICIO).dt.days / 30.44).round().astype(int) + 1
    buckets = [("1-3m", (1, 3)), ("4-6m", (4, 6)), ("7-12m", (7, 12)), ("13m+", (13, 999))]
    rows_h = []
    for m in modelos_avaliar:
        for label, (lo, hi) in buckets:
            d = val[(val["h"] >= lo) & (val["h"] <= hi)].dropna(subset=[m, "real"])
            if len(d) == 0: continue
            met = metricas(d["real"].values, d[m].values)
            met.update({"Modelo": m, "Bucket": label, "N": len(d)})
            rows_h.append(met)
    df_hor = pd.DataFrame(rows_h)[["Modelo", "Bucket", "N", "RMSE", "MAE", "Bias", "MAPE"]]
    df_hor.to_csv(OUT / "metricas_por_horizonte.csv", index=False)

    # ----- Ensemble otimizado -----
    print("\n[7] Otimizando ensemble (LS + LOO-CV)...")
    candidatos = ["Integrado_Fase2", "Sazonal_Naive", "XGBoost", "LSTM"]
    candidatos = [c for c in candidatos if c in val.columns]
    val_ens = val.dropna(subset=candidatos + ["real"])
    P = val_ens[candidatos].values.T  # (n_mod, n)
    y = val_ens["real"].values

    w_ls = otimiza_pesos_ls(P, y)
    w_loo, yhat_loo = loo_pesos(P, y)
    yhat_ls = w_ls @ P

    pesos_dict = {
        "candidatos": candidatos,
        "pesos_ls_insample": dict(zip(candidatos, w_ls.tolist())),
        "pesos_loo_medios": dict(zip(candidatos, w_loo.tolist())),
        "metricas_ls_insample": metricas(y, yhat_ls),
        "metricas_loo": metricas(y, yhat_loo),
    }
    json.dump(pesos_dict, open(OUT / "ensemble_pesos.json", "w"), indent=2)
    print(f"    LS in-sample weights: {dict(zip(candidatos, np.round(w_ls, 3).tolist()))}")
    print(f"    LOO mean weights:     {dict(zip(candidatos, np.round(w_loo, 3).tolist()))}")
    print(f"    LOO RMSE: {pesos_dict['metricas_loo']['RMSE']:.1f}  MAE: {pesos_dict['metricas_loo']['MAE']:.1f}")

    # Adiciona ensemble nas metricas globais
    df_glob_full = pd.concat([
        df_glob,
        pd.DataFrame([{**metricas(y, yhat_loo), "Modelo": "Ensemble_Otimizado_LOO", "N": len(y)}])
    ])[["Modelo", "N", "RMSE", "MAE", "Bias", "MAPE"]].sort_values("RMSE").reset_index(drop=True)
    df_glob_full.to_csv(OUT / "metricas_global.csv", index=False)

    # ----- Conformal prediction (split: 1a metade calibra, 2a avalia cobertura) -----
    print("\n[8] Conformal prediction (IC 5%-95%)...")
    n = len(val_ens)
    n_cal = n // 2
    res_cal = (val_ens["real"].values[:n_cal] - yhat_loo[:n_cal])
    q90 = conformal_quantile(res_cal, alpha=0.10)

    # Aplica nos meses futuros: precisamos do ensemble previsto (todo horizonte 60m)
    pred_fut = todos.dropna(subset=candidatos)[["data"] + candidatos].set_index("data")
    yhat_60m = (pred_fut.values @ w_loo)

    df_ic = pd.DataFrame({
        "data": pred_fut.index,
        "p05": yhat_60m - q90,
        "mediana": yhat_60m,
        "p95": yhat_60m + q90,
    })
    df_ic.to_csv(OUT / "intervalo_conformal.csv", index=False)

    # Cobertura empirica na 2a metade
    yhat_test = yhat_loo[n_cal:]; y_test = val_ens["real"].values[n_cal:]
    cobertura = float(np.mean(np.abs(y_test - yhat_test) <= q90))
    print(f"    q90 = {q90:.1f} MWmed | cobertura empirica (split test): {cobertura:.0%}")

    # ----- Adiciona Ensemble_Otimizado_LOO aos CSVs principais -----
    # Re-escreve previsoes_60m_todos.csv com a coluna do ensemble
    df_ens_60m = pd.DataFrame({"data": pred_fut.index, "Ensemble_Otimizado_LOO": yhat_60m})
    todos = todos.merge(df_ens_60m, on="data", how="left")
    todos.to_csv(OUT / "previsoes_60m_todos.csv", index=False)

    # Re-escreve validacao_externa_2025.csv com a coluna do ensemble
    val_with_ens = val.merge(df_ens_60m, on="data", how="left")
    val_with_ens.drop(columns=["h"], errors="ignore").to_csv(OUT / "validacao_externa_2025.csv", index=False)

    # ----- CSV consolidado pronto para planilha externa -----
    consol = todos.merge(
        df_ic[["data", "p05", "p95"]].rename(columns={"p05": "IC_p05", "p95": "IC_p95"}),
        on="data", how="left",
    ).merge(
        df_real.rename(columns={"real": "ONS_real"}),
        on="data", how="left",
    )
    cols_pref = ["data"] + [c for c in [
        "Sazonal_Naive", "Integrado_Fase2", "Regressao_Linear",
        "Persistencia_Sazonal", "SARIMAX", "XGBoost", "LSTM",
        "Ensemble_Otimizado_LOO", "IC_p05", "IC_p95", "ONS_real",
    ] if c in consol.columns]
    consol = consol[cols_pref]
    consol.to_csv(OUT / "previsoes_60m_consolidado.csv", index=False, float_format="%.2f")
    print(f"    [OK] previsoes_60m_consolidado.csv: {len(consol)} meses, {len(consol.columns)-1} colunas (incluindo Ensemble + IC + ONS_real)")

    # ----- Tabela final markdown -----
    print("\n[9] Gerando tabela final markdown...")
    linhas = ["# Tabela final — Fase 3", "", f"Validacao externa: {len(val)} meses ({val['data'].min().date()} a {val['data'].max().date()})", ""]
    linhas.append("## Metricas globais (todos os horizontes agregados)")
    linhas.append("")
    linhas.append("| Modelo | N | RMSE | MAE | Bias | MAPE |")
    linhas.append("|---|---:|---:|---:|---:|---:|")
    for _, r in df_glob_full.iterrows():
        linhas.append(f"| {r['Modelo']} | {int(r['N'])} | {r['RMSE']:.1f} | {r['MAE']:.1f} | {r['Bias']:+.1f} | {r['MAPE']:.1f}% |")
    linhas.append("| **NEWAVE (TCC tabela 2)** | n/d | n/d | ~1521 | ~-806 | n/d |")
    linhas.append("")
    linhas.append("## Metricas por bucket de horizonte")
    linhas.append("")
    linhas.append("| Modelo | Bucket | N | RMSE | MAE | Bias |")
    linhas.append("|---|---|---:|---:|---:|---:|")
    for _, r in df_hor.iterrows():
        linhas.append(f"| {r['Modelo']} | {r['Bucket']} | {int(r['N'])} | {r['RMSE']:.1f} | {r['MAE']:.1f} | {r['Bias']:+.1f} |")
    linhas.append("")
    linhas.append("## Ensemble otimizado")
    linhas.append("")
    linhas.append(f"- Candidatos: {', '.join(candidatos)}")
    linhas.append(f"- Pesos LOO-CV (medios): {dict(zip(candidatos, np.round(w_loo, 3).tolist()))}")
    linhas.append(f"- Pesos LS in-sample: {dict(zip(candidatos, np.round(w_ls, 3).tolist()))}")
    linhas.append("")
    linhas.append(f"## Conformal prediction (alpha=0.10)")
    linhas.append("")
    linhas.append(f"- q90 (residuos calibracao) = {q90:.1f} MWmed")
    linhas.append(f"- Cobertura empirica no split test (2a metade): {cobertura:.0%}")
    linhas.append(f"- Intervalo aplicado a todo horizonte 60m em `intervalo_conformal.csv`")
    (OUT / "tabela_final.md").write_text("\n".join(linhas), encoding="utf-8")

    print(f"\n[OK] Saidas em {OUT}")


if __name__ == "__main__":
    main()
