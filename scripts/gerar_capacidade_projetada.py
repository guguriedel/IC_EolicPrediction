"""
Fase 1.2: Gera curva de capacidade eolica instalada projetada para
DATA_PRIMEIRO_MES_PREVISTO ate +HORIZONTE_PREVISAO_MESES, com 3 cenarios
calibrados em modelos DIFERENTES da serie historica (nao apenas multiplicador):

  - cenario_conservador  : linear sobre os ultimos 24 meses  (expansao
                           desacelerou nos anos recentes)
  - cenario_base         : linear sobre a serie inteira      (slope medio
                           historico, ~constante em MW/ano)
  - cenario_otimista     : log-linear sobre serie inteira    (mantem taxa
                           percentual historica - agressivo)

Quando substituir por dados oficiais (PDE/EPE 2034), basta sobrescrever o CSV
de saida mantendo as 4 colunas: data, cenario_conservador, cenario_base,
cenario_otimista.

Inputs:
    Dados/dataset_modelagem_limpo.csv  (capacidade_eolica_ne_mw historica)

Outputs:
    Dados/capacidade_projetada_ne.csv  (data + 3 cenarios)
    Dados/capacidade_projetada_ne.png  (plot historico + projecao)
    Dados/capacidade_projetada_relatorio.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DADOS = ROOT / "Dados"
INPUT = DADOS / "dataset_modelagem_limpo.csv"
OUT_CSV = DADOS / "capacidade_projetada_ne.csv"
OUT_PNG = DADOS / "capacidade_projetada_ne.png"
OUT_JSON = DADOS / "capacidade_projetada_relatorio.json"

# Constantes do contrato de datas (devem refletir cell 0 do notebook)
ULTIMO_MES_COMPLETO = pd.Timestamp("2024-12-01")
HORIZONTE_PREVISAO_MESES = 60
JANELA_CONSERVADOR_MESES = 24  # ultimos N meses para fit conservador


def main() -> int:
    if not INPUT.exists():
        print(f"ERRO: {INPUT} nao encontrado. Rode gerar_dataset_limpo.py primeiro.", file=sys.stderr)
        return 1

    df = pd.read_csv(INPUT, parse_dates=["data"]).sort_values("data").reset_index(drop=True)
    cap_hist = df[["data", "capacidade_eolica_ne_mw"]].copy()
    cap_hist["t"] = np.arange(len(cap_hist))
    y_hist = cap_hist["capacidade_eolica_ne_mw"].values
    X_full = cap_hist[["t"]].values

    # === Modelo 1: linear sobre serie inteira (BASE) ===
    m_lin = LinearRegression().fit(X_full, y_hist)
    a_lin = float(m_lin.intercept_)
    b_lin = float(m_lin.coef_[0])

    # === Modelo 2: linear sobre ultimos N meses (CONSERVADOR) ===
    cap_recent = cap_hist.iloc[-JANELA_CONSERVADOR_MESES:]
    m_lin_recent = LinearRegression().fit(cap_recent[["t"]].values, cap_recent["capacidade_eolica_ne_mw"].values)
    a_lin_r = float(m_lin_recent.intercept_)
    b_lin_r = float(m_lin_recent.coef_[0])

    # === Modelo 3: log-linear sobre serie inteira (OTIMISTA) ===
    m_log = LinearRegression().fit(X_full, np.log(y_hist))
    a_log = float(m_log.intercept_)
    b_log = float(m_log.coef_[0])

    # Diagnostico de fit historico
    fit_hist = {
        "linear_total": {
            "y_pred": m_lin.predict(X_full),
            "rmse": float(np.sqrt(((m_lin.predict(X_full) - y_hist) ** 2).mean())),
            "slope_mw_mes": b_lin,
            "slope_mw_ano": b_lin * 12,
        },
        "linear_recente": {
            "y_pred": m_lin_recent.predict(X_full),
            "rmse": float(np.sqrt(((m_lin_recent.predict(X_full) - y_hist) ** 2).mean())),
            "slope_mw_mes": b_lin_r,
            "slope_mw_ano": b_lin_r * 12,
            "janela_meses": JANELA_CONSERVADOR_MESES,
        },
        "log_linear": {
            "y_pred": np.exp(m_log.predict(X_full)),
            "rmse": float(np.sqrt(((np.exp(m_log.predict(X_full)) - y_hist) ** 2).mean())),
            "taxa_anual_pct": float((np.exp(12 * b_log) - 1) * 100),
        },
    }

    # Projecao
    datas_futuras = pd.date_range(
        ULTIMO_MES_COMPLETO + pd.DateOffset(months=1),
        periods=HORIZONTE_PREVISAO_MESES,
        freq="MS",
    )
    t_futuro = np.arange(len(cap_hist), len(cap_hist) + len(datas_futuras))
    Xf = t_futuro.reshape(-1, 1)

    candidatos = {
        "linear_total": m_lin.predict(Xf),
        "linear_recente": m_lin_recent.predict(Xf),
        "log_linear": np.exp(m_log.predict(Xf)),
    }
    # Ordena por valor final ascendente: o menor vira conservador, intermediario base, maior otimista.
    # Mantem semantica do nome independente de qual modelo ficou em qual papel.
    ordenados = sorted(candidatos.items(), key=lambda kv: kv[1][-1])
    cap_conservador = ordenados[0][1]
    cap_base = ordenados[1][1]
    cap_otimista = ordenados[2][1]
    modelo_de = {
        "conservador": ordenados[0][0],
        "base": ordenados[1][0],
        "otimista": ordenados[2][0],
    }
    print(f"Atribuicao de modelos -> cenarios: {modelo_de}")

    df_proj = pd.DataFrame({
        "data": datas_futuras,
        "cenario_conservador": cap_conservador,
        "cenario_base": cap_base,
        "cenario_otimista": cap_otimista,
    })

    for col in ["cenario_conservador", "cenario_base", "cenario_otimista"]:
        if (df_proj[col].diff().dropna() < 0).any():
            print(f"AVISO: {col} nao e monotonicamente nao-decrescente", file=sys.stderr)

    df_proj.to_csv(OUT_CSV, index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(cap_hist["data"], y_hist, label="Historico (ONS)", color="black", linewidth=2)
    ax.plot(cap_hist["data"], fit_hist["linear_total"]["y_pred"], label="Fit linear total", color="C0", linestyle="--", alpha=0.5)
    ax.plot(cap_hist["data"], fit_hist["log_linear"]["y_pred"], label="Fit log-linear", color="C1", linestyle="--", alpha=0.5)

    ax.plot(df_proj["data"], df_proj["cenario_conservador"], label=f"Conservador ({modelo_de['conservador']})", color="C2")
    ax.plot(df_proj["data"], df_proj["cenario_base"], label=f"Base ({modelo_de['base']})", color="C0")
    ax.plot(df_proj["data"], df_proj["cenario_otimista"], label=f"Otimista ({modelo_de['otimista']})", color="C1")
    ax.fill_between(df_proj["data"], df_proj["cenario_conservador"], df_proj["cenario_otimista"], alpha=0.15, color="grey")

    ax.axvline(ULTIMO_MES_COMPLETO, color="grey", linestyle=":", label="Cutoff historico")
    ax.set_xlabel("Data")
    ax.set_ylabel("Capacidade instalada NE (MW)")
    ax.set_title("Capacidade eolica NE: 3 cenarios de projecao (Fase 1.2)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)

    relatorio = {
        "input": str(INPUT),
        "n_meses_historico": len(cap_hist),
        "modelos": modelo_de,
        "fit_historico": {k: {kk: vv for kk, vv in v.items() if kk != "y_pred"} for k, v in fit_hist.items()},
        "projecao": {
            "primeiro_mes": str(datas_futuras[0].date()),
            "ultimo_mes": str(datas_futuras[-1].date()),
            "n_meses": len(datas_futuras),
            "cap_historico_ultimo_mw": float(y_hist[-1]),
            "fim_horizonte_mw": {
                "conservador": float(df_proj["cenario_conservador"].iloc[-1]),
                "base": float(df_proj["cenario_base"].iloc[-1]),
                "otimista": float(df_proj["cenario_otimista"].iloc[-1]),
            },
        },
        "outputs": {"csv": str(OUT_CSV), "png": str(OUT_PNG)},
    }
    OUT_JSON.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False))
    print(json.dumps(relatorio, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
