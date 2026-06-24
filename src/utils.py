"""Shared helpers used by the training and prediction scripts."""

import os
import random

import numpy as np


def set_seeds(seed):
    """Make GRU training as reproducible as the libraries allow.

    Note: even with seeds fixed, GPU-based training can still introduce
    minor non-determinism depending on the backend. This gets you close
    to reproducible, not bit-for-bit identical, runs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def build_sequences(values, sequence_length):
    """Turn a 1-column array into (X, y) sliding-window sequences.

    X[i] = values[i : i+sequence_length]
    y[i] = values[i+sequence_length]
    """
    X, y = [], []
    for i in range(len(values) - sequence_length):
        X.append(values[i:(i + sequence_length)])
        y.append(values[i + sequence_length])
    return np.array(X), np.array(y)


def build_gru_model(sequence_length, n_units=30):
    """Same architecture as the original research notebook."""
    from keras.models import Sequential
    from keras.layers import GRU, Dense, Dropout, Input

    model = Sequential([
        Input(shape=(sequence_length, 1)),
        GRU(units=n_units),
        Dropout(0.5),
        Dense(64),
        Dropout(0.5),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model
