"""Fase 3.4 - Baselines simples adicionais.

Modela FC e multiplica por capacidade projetada para gerar geracao MWmed.

Modelos:
  - persistencia_sazonal: y[m, ano+k] = y[m, ultimo ano observado]
  - regressao_linear:     FC ~ capacidade + mes_sin + mes_cos + nino34
  - sarimax:              SARIMAX(1,0,1)(1,0,1,12) com capacidade exogena
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent.parent
DADOS = ROOT / "Dados"


def _capacidade_proj(cenario: str = "conservador") -> pd.Series:
    cap = pd.read_csv(DADOS / "capacidade_projetada_ne.csv", parse_dates=["data"])
    return cap.set_index("data")[f"cenario_{cenario}"]


def persistencia_sazonal_fc(df_hist: pd.DataFrame, datas_futuras: pd.DatetimeIndex) -> pd.Series:
    """y_hat[ano, mes] = y_real[mes] no ultimo ano completo."""
    df = df_hist.copy().sort_values("data")
    ultimo_ano = df["data"].dt.year.max()
    ref = df[df["data"].dt.year == ultimo_ano].set_index(df[df["data"].dt.year == ultimo_ano]["data"].dt.month)["fc_ne"]
    fc = pd.Series(index=datas_futuras, dtype=float)
    for d in datas_futuras:
        fc.loc[d] = ref.loc[d.month] if d.month in ref.index else df["fc_ne"].mean()
    return fc


def regressao_linear_fc(df_hist: pd.DataFrame, datas_futuras: pd.DatetimeIndex) -> pd.Series:
    """FC = b0 + b1*capacidade + b2*sin(mes) + b3*cos(mes) + b4*nino34."""
    df = df_hist.dropna(subset=["fc_ne", "capacidade_eolica_ne_mw"]).copy()
    df["mes_sin"] = np.sin(2 * np.pi * df["data"].dt.month / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["data"].dt.month / 12)
    nino_col = "nino34_anom" if "nino34_anom" in df.columns else None
    feats = ["capacidade_eolica_ne_mw", "mes_sin", "mes_cos"]
    if nino_col:
        df = df.dropna(subset=[nino_col])
        feats.append(nino_col)
    modelo = LinearRegression().fit(df[feats].values, df["fc_ne"].values)

    cap_proj = _capacidade_proj("conservador")
    nino_clima = df.groupby(df["data"].dt.month)[nino_col].mean() if nino_col else None

    rows = []
    for d in datas_futuras:
        x = [cap_proj.loc[d] if d in cap_proj.index else df["capacidade_eolica_ne_mw"].iloc[-1],
             np.sin(2 * np.pi * d.month / 12),
             np.cos(2 * np.pi * d.month / 12)]
        if nino_col:
            x.append(nino_clima.loc[d.month])
        rows.append(modelo.predict([x])[0])
    return pd.Series(rows, index=datas_futuras).clip(0, 0.7)


def sarimax_fc(df_hist: pd.DataFrame, datas_futuras: pd.DatetimeIndex) -> pd.Series:
    """SARIMAX(1,0,1)(1,0,1,12) sobre FC com capacidade como exogena."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    df = df_hist.dropna(subset=["fc_ne", "capacidade_eolica_ne_mw"]).sort_values("data")
    y = df["fc_ne"].values
    exog_hist = df[["capacidade_eolica_ne_mw"]].values

    cap_proj = _capacidade_proj("conservador")
    exog_fut = np.array([[cap_proj.loc[d] if d in cap_proj.index else exog_hist[-1, 0]]
                         for d in datas_futuras])

    modelo = SARIMAX(y, exog=exog_hist, order=(1, 0, 1), seasonal_order=(1, 0, 1, 12),
                     enforce_stationarity=False, enforce_invertibility=False)
    fit = modelo.fit(disp=False)
    pred = fit.forecast(steps=len(datas_futuras), exog=exog_fut)
    return pd.Series(np.clip(pred, 0, 0.7), index=datas_futuras)


def fc_para_mwmed(fc: pd.Series, cenario: str = "conservador") -> pd.Series:
    cap = _capacidade_proj(cenario)
    out = pd.Series(index=fc.index, dtype=float)
    for d in fc.index:
        c = cap.loc[d] if d in cap.index else cap.iloc[-1]
        out.loc[d] = fc.loc[d] * c
    return out
