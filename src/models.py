"""Fase 2.1, 2.2, 2.3 - Arquiteturas de modelo.

Tres arquiteturas:
  - LSTM solo            (ablation, baseline neural)
  - Transformer solo     (ablation, atencional puro)
  - Integrado            (CNN extrator -> [LSTM, Transformer] -> fusao aprendida)

Princípios (conforme PLAN.md secao 2.2):
  - Capacidade pequena (dataset ~70 sequencias)
  - Loss Huber (robusta a outliers)
  - Sem recurrent_dropout (mata cuDNN)
  - Sem bidirectional (nao faz sentido em forecasting)
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Conv1D, LSTM, Dense, Dropout, Concatenate, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Add, Lambda, Multiply,
)
from tensorflow.keras import ops as kops
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam


@dataclass(frozen=True)
class HiperParams:
    L: int = 24
    H: int = 12
    cnn_filters: int = 32
    cnn_kernel: int = 3
    lstm_units: int = 32
    d_model: int = 32  # = cnn_filters apos extractor (precisa bater)
    n_heads: int = 4
    ff_dim: int = 64
    dropout: float = 0.15
    lr: float = 1e-3
    batch_size: int = 16
    max_epochs: int = 200


def _positional_encoding(L: int, d: int) -> tf.Tensor:
    pos = np.arange(L)[:, None]
    i = np.arange(d)[None, :]
    angles = pos / np.power(10000.0, (2 * (i // 2)) / d)
    pe = np.zeros((L, d), dtype=np.float32)
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.constant(pe[None, :, :])


def _bloco_transformer(x, hp: HiperParams):
    pe = _positional_encoding(hp.L, hp.d_model)
    x = x + pe
    attn = MultiHeadAttention(num_heads=hp.n_heads, key_dim=hp.d_model // hp.n_heads,
                              dropout=hp.dropout)(x, x)
    x1 = LayerNormalization(epsilon=1e-6)(Add()([x, attn]))
    ff = Dense(hp.ff_dim, activation="relu")(x1)
    ff = Dense(hp.d_model)(ff)
    x2 = LayerNormalization(epsilon=1e-6)(Add()([x1, ff]))
    return x2


def _compilar(model: Model, hp: HiperParams) -> Model:
    model.compile(optimizer=Adam(learning_rate=hp.lr), loss=Huber(delta=1.0), metrics=["mae"])
    return model


def build_lstm_solo(n_features: int, hp: HiperParams) -> Model:
    inp = Input(shape=(hp.L, n_features), name="entrada")
    x = LSTM(hp.lstm_units)(inp)
    x = Dropout(hp.dropout)(x)
    out = Dense(hp.H, name="saida_h")(x)
    return _compilar(Model(inp, out, name="lstm_solo"), hp)


def build_transformer_solo(n_features: int, hp: HiperParams) -> Model:
    inp = Input(shape=(hp.L, n_features), name="entrada")
    x = Dense(hp.d_model)(inp)  # projeta para d_model
    x = _bloco_transformer(x, hp)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(hp.dropout)(x)
    out = Dense(hp.H, name="saida_h")(x)
    return _compilar(Model(inp, out, name="transformer_solo"), hp)


def build_integrado(n_features: int, hp: HiperParams) -> Model:
    """Arquitetura prometida no TCC secao 3.2:
       CNN extrai mapas locais -> ramos LSTM e Transformer recebem esses mapas
       -> fusao por pesos aprendidos."""
    assert hp.cnn_filters == hp.d_model, "cnn_filters deve igualar d_model"
    inp = Input(shape=(hp.L, n_features), name="entrada")
    # === Extrator CNN (compartilhado) ===
    feat = Conv1D(hp.cnn_filters, hp.cnn_kernel, padding="causal", activation="relu")(inp)
    feat = Conv1D(hp.cnn_filters, hp.cnn_kernel, padding="causal", activation="relu")(feat)
    # === Ramo LSTM ===
    lstm_h = LSTM(hp.lstm_units)(feat)
    lstm_h = Dropout(hp.dropout)(lstm_h)
    pred_lstm = Dense(hp.H, name="pred_lstm")(lstm_h)
    # === Ramo Transformer ===
    trans = _bloco_transformer(feat, hp)
    trans_pool = GlobalAveragePooling1D()(trans)
    trans_pool = Dropout(hp.dropout)(trans_pool)
    pred_transf = Dense(hp.H, name="pred_transf")(trans_pool)
    # === Fusao por gating aprendido ===
    fusao = Concatenate()([lstm_h, trans_pool])
    pesos = Dense(2, activation="softmax", name="pesos_fusao")(fusao)
    w_l = Lambda(lambda t: kops.expand_dims(t[:, 0], -1))(pesos)
    w_t = Lambda(lambda t: kops.expand_dims(t[:, 1], -1))(pesos)
    out = Add(name="saida_h")([Multiply()([w_l, pred_lstm]), Multiply()([w_t, pred_transf])])
    return _compilar(Model(inp, out, name="integrado_cnn_lstm_transformer"), hp)


CONSTRUTORES = {
    "lstm_solo": build_lstm_solo,
    "transformer_solo": build_transformer_solo,
    "integrado": build_integrado,
}
