"""Fase 2 - Preparacao de dados para modelagem.

Responsavel por:
  - Carregar dataset_modelagem_limpo
  - Remover linhas iniciais sem lag_12
  - Definir features ativas (auditadas pelo contrato de features_futuras)
  - Split temporal train / val / test
  - Fitar scalers SOMENTE no treino (Fase 0.5 anti-leak)
  - Construir sequencias (X, y) com janela L e horizonte H
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from .features_futuras import auditar_features

ROOT = Path(__file__).resolve().parent.parent
DADOS = ROOT / "Dados"
CSV_LIMPO = DADOS / "dataset_modelagem_limpo.csv"

TARGET = "fc_ne"
COLS_NAO_FEATURE = {
    "data", "ano", "mes", "trimestre", "subsistema",
    "geracao_eolica_ne_mwmed", "fc_ne",
}


@dataclass
class SplitTemporal:
    df_treino: pd.DataFrame
    df_val: pd.DataFrame
    df_teste: pd.DataFrame
    features: list[str]
    scaler_X: RobustScaler
    scaler_y: RobustScaler


def carregar_dataset() -> pd.DataFrame:
    df = pd.read_csv(CSV_LIMPO, parse_dates=["data"]).sort_values("data").reset_index(drop=True)
    # Remove linhas iniciais sem lag_12 (12 primeiros meses)
    df = df.dropna(subset=["fc_lag_12"]).reset_index(drop=True)
    return df


def listar_features(df: pd.DataFrame) -> list[str]:
    feats = [c for c in df.columns if c not in COLS_NAO_FEATURE]
    auditar_features(feats)  # erro duro se alguma feature nao tem regra futura
    return feats


def split_temporal(
    df: pd.DataFrame,
    fim_treino: pd.Timestamp,
    fim_val: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_treino = df[df["data"] <= fim_treino].copy()
    df_val = df[(df["data"] > fim_treino) & (df["data"] <= fim_val)].copy()
    df_teste = df[df["data"] > fim_val].copy()
    return df_treino, df_val, df_teste


def fitar_scalers(
    df_treino: pd.DataFrame, features: list[str]
) -> tuple[RobustScaler, RobustScaler]:
    scaler_X = RobustScaler().fit(df_treino[features].values)
    scaler_y = RobustScaler().fit(df_treino[[TARGET]].values)
    return scaler_X, scaler_y


def imputar_treino_e_aplicar(
    df_treino: pd.DataFrame, df_val: pd.DataFrame, df_teste: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Mediana fitada SO no treino, aplicada nos tres."""
    medianas = df_treino[features].median()
    return (
        df_treino.assign(**{f: df_treino[f].fillna(medianas[f]) for f in features}),
        df_val.assign(**{f: df_val[f].fillna(medianas[f]) for f in features}),
        df_teste.assign(**{f: df_teste[f].fillna(medianas[f]) for f in features}),
    )


def construir_sequencias(
    df: pd.DataFrame, features: list[str], scaler_X: RobustScaler, scaler_y: RobustScaler,
    L: int, H: int,
) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]:
    """Retorna X shape (N, L, F), y shape (N, H), datas_alvo (lista de Timestamps
    da PRIMEIRA observacao do horizonte y[i, 0])."""
    X_full = scaler_X.transform(df[features].values)
    y_full = scaler_y.transform(df[[TARGET]].values).ravel()
    datas = df["data"].values

    X_seq, y_seq, datas_alvo = [], [], []
    for i in range(len(df) - L - H + 1):
        X_seq.append(X_full[i : i + L])
        y_seq.append(y_full[i + L : i + L + H])
        datas_alvo.append(pd.Timestamp(datas[i + L]))
    if not X_seq:
        return np.empty((0, L, len(features))), np.empty((0, H)), []
    return np.array(X_seq), np.array(y_seq), datas_alvo


def preparar_tudo(
    fim_treino: pd.Timestamp,
    fim_val: pd.Timestamp,
    L: int,
    H: int,
) -> tuple[SplitTemporal, dict]:
    """Pipeline completo: carrega, split, imputa, fita scaler, gera sequencias.

    Retorna (SplitTemporal, dict com X_train/y_train/X_val/y_val/X_test/y_test/datas_*).
    """
    df = carregar_dataset()
    features = listar_features(df)
    df_tr, df_v, df_te = split_temporal(df, fim_treino, fim_val)
    df_tr, df_v, df_te = imputar_treino_e_aplicar(df_tr, df_v, df_te, features)
    scaler_X, scaler_y = fitar_scalers(df_tr, features)

    # Para val/test precisamos de janela de contexto a partir do treino,
    # entao concatenamos o tail do treino para gerar sequencias completas.
    df_v_ctx = pd.concat([df_tr.tail(L), df_v]).reset_index(drop=True)
    df_te_ctx = pd.concat([df_tr.tail(L // 2), df_v, df_te]).reset_index(drop=True)

    X_tr, y_tr, dt_tr = construir_sequencias(df_tr, features, scaler_X, scaler_y, L, H)
    X_v, y_v, dt_v = construir_sequencias(df_v_ctx, features, scaler_X, scaler_y, L, H)
    X_te, y_te, dt_te = construir_sequencias(df_te_ctx, features, scaler_X, scaler_y, L, H)

    # Filtra sequencias do val/test cujo PRIMEIRO mes do horizonte cai no
    # split correto (assim nao vazamos val em test, etc).
    def _filtra(X, y, dt, lo, hi):
        mask = [(d > lo) and (d <= hi) for d in dt]
        return X[mask], y[mask], [d for d, m in zip(dt, mask) if m]

    X_v, y_v, dt_v = _filtra(X_v, y_v, dt_v, fim_treino, fim_val)
    X_te, y_te, dt_te = _filtra(X_te, y_te, dt_te, fim_val, pd.Timestamp("2099-01-01"))

    split = SplitTemporal(df_tr, df_v, df_te, features, scaler_X, scaler_y)
    seqs = dict(
        X_train=X_tr, y_train=y_tr, datas_train=dt_tr,
        X_val=X_v, y_val=y_v, datas_val=dt_v,
        X_test=X_te, y_test=y_te, datas_test=dt_te,
    )
    return split, seqs
