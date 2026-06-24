"""
The daily job. Triggered by .github/workflows/daily_predict.yml after
NSE market close on each trading day. Does three things, in order:

  1. Pulls the latest close price (data_pipeline.update_local_data).
  2. Reconciles any past prediction whose target date has now arrived —
     i.e. now that today's actual close is known, it computes how wrong
     yesterday's GRU prediction for today was, and logs that residual.
  3. Makes tomorrow's prediction: GRU's raw forecast from the last 22
     days, plus a correction forecast by Prophet trained on the residual
     log built up so far. Prophet only ever sees residuals for days
     that have already happened, so this stays leak-free.
"""

import os
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd

from config import (
    DATA_PATH, GRU_MODEL_PATH, SCALER_PATH, RESIDUAL_LOG_PATH,
    PREDICTIONS_LOG_PATH, SEQUENCE_LENGTH, MIN_RESIDUAL_HISTORY,
)
from data_pipeline import update_local_data


def load_artifacts():
    import keras
    model = keras.models.load_model(GRU_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def reconcile_predictions_log(latest_data):
    """Fill in 'actual' for predictions whose target date has occurred,
    and append the realized residual to the residual log."""
    if not os.path.exists(PREDICTIONS_LOG_PATH):
        return

    preds = pd.read_csv(PREDICTIONS_LOG_PATH, parse_dates=["made_on", "target_date"])
    if os.path.exists(RESIDUAL_LOG_PATH):
        residual_log = pd.read_csv(RESIDUAL_LOG_PATH, parse_dates=["Date"])
    else:
        residual_log = pd.DataFrame(columns=["Date", "actual", "gru_pred", "residual"])

    latest_indexed = latest_data.set_index("Date")
    new_residual_rows = []
    changed = False

    pending = preds[preds["actual"].isna()]
    for idx, row in pending.iterrows():
        target = row["target_date"]
        if target in latest_indexed.index:
            actual = float(latest_indexed.loc[target, "Close"])
            preds.loc[idx, "actual"] = actual
            changed = True
            if target not in residual_log["Date"].values:
                new_residual_rows.append({
                    "Date": target,
                    "actual": actual,
                    "gru_pred": row["gru_pred"],
                    "residual": actual - row["gru_pred"],
                })

    if new_residual_rows:
        residual_log = pd.concat([residual_log, pd.DataFrame(new_residual_rows)], ignore_index=True)
        residual_log = residual_log.drop_duplicates(subset="Date").sort_values("Date")
        residual_log.to_csv(RESIDUAL_LOG_PATH, index=False)
        print(f"Reconciled {len(new_residual_rows)} new realized residual(s).")

    if changed:
        preds.to_csv(PREDICTIONS_LOG_PATH, index=False)


def next_trading_date(from_date):
    target = from_date + timedelta(days=1)
    while target.weekday() >= 5:  # skip Sat/Sun; doesn't account for NSE holidays
        target += timedelta(days=1)
    return target


def predict_next_close():
    latest_data = update_local_data()
    latest_data = latest_data.sort_values("Date").reset_index(drop=True)

    reconcile_predictions_log(latest_data)

    model, scaler = load_artifacts()

    recent_close = latest_data["Close"].values[-SEQUENCE_LENGTH:].reshape(-1, 1)
    scaled_window = scaler.transform(recent_close).reshape(1, SEQUENCE_LENGTH, 1)
    gru_pred_scaled = model.predict(scaled_window, verbose=0)
    gru_pred = float(scaler.inverse_transform(gru_pred_scaled)[0, 0])

    residual_log = (
        pd.read_csv(RESIDUAL_LOG_PATH, parse_dates=["Date"])
        if os.path.exists(RESIDUAL_LOG_PATH) else pd.DataFrame()
    )

    if len(residual_log) >= MIN_RESIDUAL_HISTORY:
        from prophet import Prophet
        prophet_df = residual_log.rename(columns={"Date": "ds", "residual": "y"})[["ds", "y"]]
        m = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=False)
        m.fit(prophet_df)
        future = m.make_future_dataframe(periods=1)
        forecast = m.predict(future)
        residual_correction = float(forecast["yhat"].values[-1])
    else:
        print(f"Only {len(residual_log)} residuals logged (need {MIN_RESIDUAL_HISTORY}); "
              f"skipping Prophet correction for now.")
        residual_correction = 0.0

    final_pred = gru_pred + residual_correction

    last_date = latest_data["Date"].max()
    target_date = next_trading_date(last_date)

    new_row = pd.DataFrame([{
        "made_on": last_date,
        "target_date": target_date,
        "gru_pred": gru_pred,
        "residual_correction": residual_correction,
        "final_pred": final_pred,
        "actual": np.nan,
    }])

    if os.path.exists(PREDICTIONS_LOG_PATH):
        preds = pd.read_csv(PREDICTIONS_LOG_PATH, parse_dates=["made_on", "target_date"])
        # avoid duplicate prediction rows if this script gets re-run same day
        preds = preds[preds["target_date"] != target_date]
        preds = pd.concat([preds, new_row], ignore_index=True)
    else:
        preds = new_row

    preds.to_csv(PREDICTIONS_LOG_PATH, index=False)

    print(
        f"Prediction for {target_date.date()}: "
        f"GRU={gru_pred:.2f}, correction={residual_correction:+.2f}, final={final_pred:.2f}"
    )
    return final_pred


if __name__ == "__main__":
    predict_next_close()
