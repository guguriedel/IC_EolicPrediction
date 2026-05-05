"""Fase 2.4, 2.5, 2.6 - Treino com callbacks, multi-seed, mesma config sempre."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from .models import HiperParams, CONSTRUTORES
from .seeds import set_seeds


ROOT = Path(__file__).resolve().parent.parent
MODELOS_DIR = ROOT / "Modelos" / "fase2"
MODELOS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ResultadoTreino:
    seed: int
    arquitetura: str
    val_rmse_scaled: float
    val_mae_scaled: float
    test_rmse_scaled: float
    test_mae_scaled: float
    n_epocas_efetivas: int


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def treinar_uma_seed(
    arquitetura: str,
    hp: HiperParams,
    seqs: dict,
    seed: int,
    verbose: int = 0,
) -> tuple[tf.keras.Model, ResultadoTreino]:
    set_seeds(seed)
    n_features = seqs["X_train"].shape[2]
    model = CONSTRUTORES[arquitetura](n_features, hp)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, mode="min"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5),
    ]

    if len(seqs["X_val"]) > 0:
        val_data = (seqs["X_val"], seqs["y_val"])
    else:
        val_data = None
        callbacks = []

    hist = model.fit(
        seqs["X_train"], seqs["y_train"],
        validation_data=val_data,
        epochs=hp.max_epochs,
        batch_size=min(hp.batch_size, len(seqs["X_train"])),
        callbacks=callbacks,
        verbose=verbose,
        shuffle=False,  # series temporais
    )

    y_val_pred = model.predict(seqs["X_val"], verbose=0) if len(seqs["X_val"]) else np.empty(0)
    y_te_pred = model.predict(seqs["X_test"], verbose=0) if len(seqs["X_test"]) else np.empty(0)

    res = ResultadoTreino(
        seed=seed,
        arquitetura=arquitetura,
        val_rmse_scaled=_rmse(seqs["y_val"], y_val_pred) if len(seqs["X_val"]) else float("nan"),
        val_mae_scaled=_mae(seqs["y_val"], y_val_pred) if len(seqs["X_val"]) else float("nan"),
        test_rmse_scaled=_rmse(seqs["y_test"], y_te_pred) if len(seqs["X_test"]) else float("nan"),
        test_mae_scaled=_mae(seqs["y_test"], y_te_pred) if len(seqs["X_test"]) else float("nan"),
        n_epocas_efetivas=len(hist.history.get("loss", [])),
    )
    return model, res


def treinar_multi_seed(
    arquitetura: str,
    hp: HiperParams,
    seqs: dict,
    seeds: list[int] = (42, 1, 2, 3, 4),
    verbose: int = 0,
) -> tuple[list[tf.keras.Model], pd.DataFrame]:
    modelos, resultados = [], []
    for s in seeds:
        m, r = treinar_uma_seed(arquitetura, hp, seqs, s, verbose=verbose)
        modelos.append(m)
        resultados.append(r.__dict__)
        print(f"  seed={s:>2}  val_rmse={r.val_rmse_scaled:.4f}  test_rmse={r.test_rmse_scaled:.4f}  ep={r.n_epocas_efetivas}")
    df_res = pd.DataFrame(resultados)
    return modelos, df_res
