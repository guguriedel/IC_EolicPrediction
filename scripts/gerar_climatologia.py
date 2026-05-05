"""
Fase 1.4: Gera climatologia mensal das features meteorologicas para uso na
projecao recursiva (substituir o "congelar a ultima linha" por climatologia).

Para cada coluna meteorologica e cada mes do ano (1-12), calcula media e desvio
padrao no historico. A climatologia e o piso defensavel para projetar features
no futuro - tipicamente menos enviesado que extrapolacao temporal.

Inputs:
    Dados/dataset_modelagem_limpo.csv

Outputs:
    Dados/climatologia_mensal_meteo.csv  (long format: mes, feature, mean, std, n_amostras)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DADOS = ROOT / "Dados"
INPUT = DADOS / "dataset_modelagem_limpo.csv"
OUT_CSV = DADOS / "climatologia_mensal_meteo.csv"
OUT_JSON = DADOS / "climatologia_relatorio.json"

# Prefixos das features meteorologicas (definidos pelo nome de coluna no dataset)
PREFIXOS_METEO = (
    "vel_vento_100m_",
    "t2m_",
    "sp_",
    "tp_",
    "sst_",
    "zust_",
    "densidade_ar_",
)

# Capacidade do cluster NAO entra na climatologia: deve seguir cenario futuro
# (ela cresce no tempo). Calendario (mes_sin/cos) tambem nao - e deterministico.


def main() -> int:
    if not INPUT.exists():
        print(f"ERRO: {INPUT} nao encontrado.", file=sys.stderr)
        return 1
    df = pd.read_csv(INPUT, parse_dates=["data"])
    df["mes"] = df["data"].dt.month

    cols_meteo = [c for c in df.columns if c.startswith(PREFIXOS_METEO)]
    if not cols_meteo:
        print("ERRO: nenhuma coluna meteorologica encontrada", file=sys.stderr)
        return 1

    registros = []
    for col in cols_meteo:
        for mes, grupo in df.groupby("mes"):
            vals = grupo[col].dropna()
            if len(vals) == 0:
                continue
            registros.append({
                "mes": int(mes),
                "feature": col,
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "n_amostras": int(len(vals)),
            })

    df_clim = pd.DataFrame(registros)
    df_clim.to_csv(OUT_CSV, index=False)

    relatorio = {
        "input": str(INPUT),
        "n_features_meteorologicas": len(cols_meteo),
        "features": cols_meteo,
        "n_meses_amostras": int(df["mes"].nunique()),
        "n_anos_amostras": int(df["data"].dt.year.nunique()),
        "output": str(OUT_CSV),
        "exemplo_jan_vento": df_clim[
            (df_clim["mes"] == 1) & df_clim["feature"].str.startswith("vel_vento_100m_")
        ].head(3).to_dict(orient="records"),
    }
    OUT_JSON.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False))
    print(f"OK: {OUT_CSV} ({len(df_clim)} registros: {df['mes'].nunique()} meses x {len(cols_meteo)} features)")
    print(f"Exemplo (Janeiro, vento 100m, primeiros 3 clusters):")
    for r in relatorio["exemplo_jan_vento"]:
        print(f"  {r['feature']:50s} mean={r['mean']:.3f}  std={r['std']:.3f}  n={r['n_amostras']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
