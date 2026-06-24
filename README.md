# Bank Nifty Next-Day Close Predictor

A live, self-updating forecast of the Bank Nifty (`^NSEBANK`) close price, built on
a GRU + Prophet residual-correction hybrid, with a Streamlit dashboard showing
the prediction alongside rolling accuracy and a comparison against the raw GRU.

This project extends the offline research notebook (GRU / ARIMA / Prophet /
hybrid backtests) into something that actually runs forward in time, every
trading day, without manual intervention.

## Why this is structured the way it is

The original notebook's hybrid models computed residuals using the *true* test
values before training the correction model on them — fine for a one-shot
backtest, but it can't work for live prediction, since on any given day you
don't yet know tomorrow's actual close.

This version fixes that: a running residual log only ever contains errors for
days that have *already happened*. Each day, after the actual close becomes
known, that day's GRU error gets appended to the log. Prophet is then refit on
that log and asked to forecast tomorrow's *expected* error, which gets added to
GRU's raw prediction. At no point does anything use information from the future.

## Architecture

```
data/
  banknifty_history.csv   <- raw OHLC history, updated daily
  residual_log.csv        <- GRU's realized prediction errors, day by day
  predictions_log.csv     <- every prediction made, reconciled against actuals
models/
  gru_model.keras          <- trained GRU
  scaler.pkl               <- the MinMaxScaler the GRU expects
src/
  config.py                <- paths & hyperparameters
  utils.py                 <- seeding, sequence building, GRU architecture
  data_pipeline.py         <- fetches/updates data from Yahoo Finance
  train_models.py          <- (manual) trains GRU, builds initial residual log
  daily_predict.py         <- (scheduled) reconciles yesterday, predicts tomorrow
dashboard/
  app.py                    <- Streamlit dashboard
.github/workflows/
  daily_predict.yml         <- runs daily_predict.py on trading days
  train_models.yml          <- manual trigger to (re)train the GRU
```

Two GitHub Actions workflows do the automation:
- **`daily_predict.yml`** runs automatically on weekdays after market close,
  pulls the latest close, reconciles the previous prediction, makes tomorrow's
  prediction, and commits the updated CSVs back to the repo.
- **`train_models.yml`** is manual-only (`workflow_dispatch`). Run it once to
  bootstrap the model, and again whenever you want to retrain the GRU on more
  recent data. It's deliberately not scheduled, since retraining changes the
  model the daily job depends on and you'll want to do that on purpose.

The dashboard reads straight from the CSVs in the repo — no separate database
or backend server needed.

## Local setup (do this first, before deploying anything)

```bash
git clone <your-repo-url>
cd banknifty-predictor
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# 1. Pull full history and train the GRU (takes a few minutes)
python src/data_pipeline.py
python src/train_models.py

# 2. Generate your first live prediction
python src/daily_predict.py

# 3. Check it locally
streamlit run dashboard/app.py
```

If `python src/train_models.py` fails to find a GPU, that's fine — the model
is small enough to train on CPU in a few minutes.

## Pushing to GitHub

Commit everything, **including `data/` and `models/`** — they're intentionally
not gitignored, since the GitHub Actions workflows rely on git itself as the
persistence layer (no database to set up).

```bash
git add .
git commit -m "Initial commit: trained model + first prediction"
git push
```

Then, in your repo on GitHub:
**Settings → Actions → General → Workflow permissions → set to
"Read and write permissions"**. This is required or the daily workflow won't
be able to commit its updates back to the repo.

## Deploying the dashboard (Streamlit Community Cloud, free)

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app**, select this repo, branch `main`, and set the main file
   path to `dashboard/app.py`.
3. Deploy. Streamlit Cloud reinstalls from `requirements.txt` automatically.

The dashboard will reflect whatever's currently in `data/` on the `main`
branch, so once the daily workflow starts committing updates, the dashboard
updates with it (Streamlit Cloud polls the repo for changes).

## What happens automatically from here

Every weekday evening (after NSE close), `daily_predict.yml`:
1. Fetches the day's close.
2. If a prediction was made yesterday for today, computes how far off it was
   and logs that residual.
3. Predicts tomorrow's close and appends it to `predictions_log.csv`.
4. Commits the updated `data/` files back to the repo.

You don't need to do anything day-to-day. Re-run `train_models.yml` manually
whenever you want the GRU itself to learn from more recent data — there's a
natural tradeoff here: retraining too often makes the residual log's history
somewhat inconsistent with the new model, so monthly or quarterly is a
reasonable cadence to start with.

## Known limitations / things to improve next

- `next_trading_date()` only skips weekends, not NSE holidays — on a holiday
  the prediction made the day before just won't get reconciled until the next
  real trading day, which is harmless but slightly delays accuracy tracking.
- The dashboard currently compares GRU-only vs. the GRU-Prophet hybrid. Adding
  the standalone ARIMA and Prophet models, or the GRU-ARIMA hybrid, as
  additional rolling-comparison lines is a natural next step if you want a
  fuller picture of which approach is actually winning live (not just in the
  original backtest).
- `prophet` has heavyweight build dependencies (`cmdstanpy`); if your
  GitHub Actions runner has trouble installing it, pinning a specific version
  in `requirements.txt` usually resolves it.
