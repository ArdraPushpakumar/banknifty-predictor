"""
Shared configuration for the Bank Nifty predictor.
All paths are relative to the repo root — every script should be run
from the repo root, e.g. `python src/daily_predict.py`.
"""

# --- Data source ---
TICKER = "^NSEBANK"
HISTORY_START_DATE = "2015-01-01"

# --- File paths (relative to repo root) ---
DATA_PATH = "data/banknifty_history.csv"
RESIDUAL_LOG_PATH = "data/residual_log.csv"
PREDICTIONS_LOG_PATH = "data/predictions_log.csv"

MODEL_DIR = "models"
GRU_MODEL_PATH = f"{MODEL_DIR}/gru_model.keras"
SCALER_PATH = f"{MODEL_DIR}/scaler.pkl"

# --- Model hyperparameters ---
SEQUENCE_LENGTH = 22          # trading days of history the GRU looks at
GRU_UNITS = 30
EPOCHS = 80
BATCH_SIZE = 32
TRAIN_TEST_SPLIT = 0.7         # used only for the one-time backtest/bootstrap
RANDOM_SEED = 42

# Minimum number of realized residuals needed before Prophet is trusted
# to produce a correction. Below this, the hybrid just falls back to the
# raw GRU prediction (correction = 0).
MIN_RESIDUAL_HISTORY = 30
