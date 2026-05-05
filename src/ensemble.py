"""Fase 3.1, 3.5 - Ensemble otimizado e conformal prediction."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def otimiza_pesos_ls(preds: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Constrained least squares: w_i >= 0, sum(w_i) = 1.

    preds: shape (n_modelos, n_amostras).
    """
    n = preds.shape[0]

    def obj(w):
        return float(np.mean((w @ preds - y) ** 2))

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bnds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n
    res = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-9})
    return res.x


def loo_pesos(preds: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-out: para cada amostra, ajusta pesos no resto e prediz.
    Retorna (pesos_medios, predicoes_loo).
    """
    n_mod, n = preds.shape
    Ws = np.zeros((n, n_mod))
    yhat = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        w = otimiza_pesos_ls(preds[:, mask], y[mask])
        Ws[i] = w
        yhat[i] = w @ preds[:, i]
    return Ws.mean(axis=0), yhat


def conformal_quantile(residuos_calib: np.ndarray, alpha: float = 0.10) -> float:
    """Quantil empirico para IC simetrico (1-alpha) — split conformal."""
    n = len(residuos_calib)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return float(np.quantile(np.abs(residuos_calib), q_level))


def metricas(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-6)) * 100)
    return {"RMSE": rmse, "MAE": mae, "Bias": bias, "MAPE": mape}
