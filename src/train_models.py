"""
Run this manually (not on the daily schedule) to:
  1. Train the GRU model on the full available history.
  2. Walk forward through the test period to build a clean residual log
     (GRU's prediction error on each day, using only information that
     would genuinely have been available at the time).

This residual log is what bootstraps the Prophet correction model in
daily_predict.py. Re-run this whenever you want to retrain the GRU on
fresher data (e.g. monthly) — see the README for how to trigger it
manually via GitHub Actions.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import (
    DATA_PATH, GRU_MODEL_PATH, SCALER_PATH, RESIDUAL_LOG_PATH, MODEL_DIR,
    SEQUENCE_LENGTH, GRU_UNITS, EPOCHS, BATCH_SIZE, TRAIN_TEST_SPLIT,
    RANDOM_SEED,
)
from utils import set_seeds, build_sequences, build_gru_model


def train_gru_and_build_residual_log():
    set_seeds(RANDOM_SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    close = df[["Close"]].values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)

    train_size = int(len(scaled) * TRAIN_TEST_SPLIT)
    train_data = scaled[:train_size]

    X_train, y_train = build_sequences(train_data, SEQUENCE_LENGTH)

    print(f"Training GRU on {len(X_train)} sequences ({train_size} of {len(df)} rows)...")
    model = build_gru_model(SEQUENCE_LENGTH, n_units=GRU_UNITS)
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    model.save(GRU_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved model to {GRU_MODEL_PATH} and scaler to {SCALER_PATH}.")

    # --- Walk-forward residual log ---
    # For each day i after the training window, predict close[i] using only
    # the SEQUENCE_LENGTH days immediately before it (all of which occurred
    # before day i). This mirrors exactly what daily_predict.py will do live,
    # so the residual history Prophet learns from is methodologically
    # consistent with how it'll be used going forward — no peeking at the
    # value being predicted.
    print("Building walk-forward residual log...")
    residual_rows = []
    for i in range(max(train_size, SEQUENCE_LENGTH), len(scaled)):
        window = scaled[i - SEQUENCE_LENGTH:i].reshape(1, SEQUENCE_LENGTH, 1)
        pred_scaled = model.predict(window, verbose=0)
        pred = float(scaler.inverse_transform(pred_scaled)[0, 0])
        actual = float(close[i, 0])
        residual_rows.append({
            "Date": df.loc[i, "Date"],
            "actual": actual,
            "gru_pred": pred,
            "residual": actual - pred,
        })

    residual_df = pd.DataFrame(residual_rows)
    residual_df.to_csv(RESIDUAL_LOG_PATH, index=False)
    print(f"Residual log built with {len(residual_df)} rows -> {RESIDUAL_LOG_PATH}")


if __name__ == "__main__":
    train_gru_and_build_residual_log()
