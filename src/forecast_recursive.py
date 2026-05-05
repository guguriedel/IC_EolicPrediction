"""Fase 2.7 - Funcao recursiva UNICA de inferencia multi-horizonte.

Substitui as 3 implementacoes divergentes do notebook (`previsao_recursiva_backtest`,
`previsao_recursiva_com_cenario`, loop em cell 20).

Estrategia: a cada passo, prediz APENAS o primeiro horizonte (h=0) do modelo,
usa essa previsao como o `fc` realimentado, atualiza a janela conforme o
contrato de features futuras, e repete H_total vezes.

Por que so o primeiro horizonte:
  - Mantem coerencia com o contrato de features futuras (capacidade,
    climatologia etc estao definidos por mes)
  - Permite asserts de shape simples (1 step = 1 mes)
  - Trade-off: mais lento (60 calls vs 5 calls com H=12), mas correto.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


@dataclass
class ResultadoForecast:
    datas: pd.DatetimeIndex
    fc_previsto: np.ndarray            # shape (H_total,) em fracao [0, 1]
    geracao_mwmed_previsto: np.ndarray # shape (H_total,) reconstruido
    capacidade_usada: np.ndarray       # shape (H_total,) MW


def _aplica_lags_target(
    df_futuro: pd.DataFrame, fc_historico: list[float], h: int,
) -> pd.DataFrame:
    """No instante h, preencher os lags do FC olhando para historico+previsoes."""
    # fc_historico = lista crescente: ate ultimo valor real, depois ja preenche com previsoes
    # No passo h, fc_lag_k em df_futuro.iloc[h] vem de fc_historico[-(k - h)]? Nao:
    # se h=0, fc_lag_1 = ultimo real; se h=1, fc_lag_1 = fc previsto em h=0.
    # historico_acumulado[-1] = ultimo conhecido ANTES do passo atual.
    for col in df_futuro.columns:
        if col.startswith("fc_lag_"):
            k = int(col.split("_")[-1])
            idx = -k  # k passos atras
            if abs(idx) <= len(fc_historico):
                df_futuro.at[df_futuro.index[h], col] = fc_historico[idx]
        elif col.startswith("geracao_lag_"):
            # Sera reconstruido depois pelo caller (precisa de capacidade)
            pass
        elif col == "fc_diff_1":
            if len(fc_historico) >= 2:
                df_futuro.at[df_futuro.index[h], col] = fc_historico[-1] - fc_historico[-2]
    return df_futuro


def _aplica_lags_geracao(
    df_futuro: pd.DataFrame, geracao_historico: list[float], h: int,
) -> pd.DataFrame:
    for col in df_futuro.columns:
        if col.startswith("geracao_lag_"):
            k = int(col.split("_")[-1])
            idx = -k
            if abs(idx) <= len(geracao_historico):
                df_futuro.at[df_futuro.index[h], col] = geracao_historico[idx]
    return df_futuro


def forecast_recursive(
    model,
    scaler_X: RobustScaler,
    scaler_y: RobustScaler,
    df_historico: pd.DataFrame,
    df_futuro_base: pd.DataFrame,
    features: list[str],
    L: int,
    H_total: int,
    capacidade_projetada: pd.Series,
) -> ResultadoForecast:
    """
    df_historico: linhas observadas (com 'data', 'fc_ne', 'geracao_eolica_ne_mwmed', features)
                  ja imputado e completo.
    df_futuro_base: saida de projetar_features_futuras (index=datas, colunas=features).
                   Lags do target estao com NaN (serao preenchidos aqui).
    capacidade_projetada: serie indexada por data com capacidade futura em MW.
    """
    H_total = min(H_total, len(df_futuro_base))
    df_fut = df_futuro_base.copy()

    fc_hist = df_historico["fc_ne"].tolist()
    ger_hist = df_historico["geracao_eolica_ne_mwmed"].tolist()
    janela_hist = df_historico[features].tail(L).copy().reset_index(drop=True)

    fc_previsto = np.zeros(H_total)
    ger_previsto = np.zeros(H_total)
    cap_usada = np.zeros(H_total)

    for h in range(H_total):
        df_fut = _aplica_lags_target(df_fut, fc_hist, h)
        df_fut = _aplica_lags_geracao(df_fut, ger_hist, h)

        # Monta janela = ultimas L linhas combinando historico + futuras ja preenchidas
        if h == 0:
            janela = janela_hist.values
        else:
            # tail do historico + linhas futuras 0..h-1 ja completas
            futuras_ate_h = df_fut.iloc[:h][features].values.astype(float)
            n_hist = L - h
            if n_hist > 0:
                janela = np.vstack([janela_hist.tail(n_hist).values, futuras_ate_h])
            else:
                janela = futuras_ate_h[-L:]

        assert janela.shape == (L, len(features)), f"Shape errado em h={h}: {janela.shape}"

        # NaN sanity: se algum NaN sobrou (feature sem cobertura), fillna com mediana da janela_hist
        if np.isnan(janela).any():
            mediana_col = np.nanmedian(janela_hist.values, axis=0)
            mask = np.isnan(janela)
            janela = np.where(mask, mediana_col[None, :], janela)

        X_scaled = scaler_X.transform(janela).reshape(1, L, len(features))
        y_hat_scaled = model.predict(X_scaled, verbose=0)[0, 0]
        y_hat = float(scaler_y.inverse_transform([[y_hat_scaled]])[0, 0])
        y_hat = float(np.clip(y_hat, 0.0, 0.7))  # FC fisicamente plausivel

        data_h = df_fut.index[h]
        cap = float(capacidade_projetada.loc[data_h])
        ger_h = y_hat * cap

        fc_previsto[h] = y_hat
        ger_previsto[h] = ger_h
        cap_usada[h] = cap

        fc_hist.append(y_hat)
        ger_hist.append(ger_h)

    return ResultadoForecast(
        datas=pd.DatetimeIndex(df_fut.index[:H_total], name="data"),
        fc_previsto=fc_previsto,
        geracao_mwmed_previsto=ger_previsto,
        capacidade_usada=cap_usada,
    )
