import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_PATH = "data/banknifty_history.csv"
PREDICTIONS_LOG_PATH = "data/predictions_log.csv"

st.set_page_config(page_title="Bank Nifty Predictor", layout="wide")

st.title("Bank Nifty — Next-Day Close Predictor")
st.caption(
    "GRU forecast corrected by a Prophet model trained on the GRU's own "
    "historical errors, retrained daily on the latest market data."
)

if not os.path.exists(PREDICTIONS_LOG_PATH) or not os.path.exists(DATA_PATH):
    st.warning(
        "No data yet. Run `python src/train_models.py` once to bootstrap, "
        "then `python src/daily_predict.py` to generate the first prediction."
    )
    st.stop()

preds = pd.read_csv(PREDICTIONS_LOG_PATH, parse_dates=["made_on", "target_date"])
history = pd.read_csv(DATA_PATH, parse_dates=["Date"])

latest = preds.sort_values("target_date").iloc[-1]

col1, col2, col3 = st.columns(3)
col1.metric(
    "Predicted close",
    f"₹{latest['final_pred']:,.2f}",
    help=f"For {latest['target_date'].date()}, made on {latest['made_on'].date()}",
)
col2.metric("GRU raw prediction", f"₹{latest['gru_pred']:,.2f}")
col3.metric("Prophet residual correction", f"{latest['residual_correction']:+.2f}")

st.divider()
st.subheader("Actual vs predicted")

realized = preds.dropna(subset=["actual"]).copy()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=history["Date"], y=history["Close"],
    name="Actual close", line=dict(color="#1f77b4"),
))
fig.add_trace(go.Scatter(
    x=realized["target_date"], y=realized["final_pred"],
    name="GRU-Prophet prediction", line=dict(color="#ff7f0e", dash="dot"),
))
fig.add_trace(go.Scatter(
    x=realized["target_date"], y=realized["gru_pred"],
    name="GRU-only prediction", line=dict(color="#2ca02c", dash="dot"),
    visible="legendonly",
))
fig.update_layout(height=450, xaxis_title="Date", yaxis_title="Close price (₹)")
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Model comparison & accuracy over time")

if len(realized) > 0:
    realized["hybrid_abs_err"] = (realized["actual"] - realized["final_pred"]).abs()
    realized["gru_abs_err"] = (realized["actual"] - realized["gru_pred"]).abs()
    realized["hybrid_pct_err"] = realized["hybrid_abs_err"] / realized["actual"] * 100
    realized["gru_pct_err"] = realized["gru_abs_err"] / realized["actual"] * 100

    comparison = pd.DataFrame({
        "Model": ["GRU only", "GRU + Prophet hybrid"],
        "MAE": [realized["gru_abs_err"].mean(), realized["hybrid_abs_err"].mean()],
        "MAPE (%)": [realized["gru_pct_err"].mean(), realized["hybrid_pct_err"].mean()],
        "Predictions tracked": [len(realized), len(realized)],
    })
    st.dataframe(comparison.style.format({"MAE": "{:.2f}", "MAPE (%)": "{:.2f}"}), hide_index=True)

    st.caption("Rolling absolute error per day (lower is better)")
    err_chart = realized.set_index("target_date")[["hybrid_abs_err", "gru_abs_err"]]
    err_chart.columns = ["GRU + Prophet hybrid", "GRU only"]
    st.line_chart(err_chart, height=250)

    with st.expander("Raw prediction log"):
        st.dataframe(
            realized[[
                "target_date", "gru_pred", "residual_correction", "final_pred",
                "actual", "hybrid_abs_err", "hybrid_pct_err",
            ]].sort_values("target_date", ascending=False),
            hide_index=True,
        )
else:
    st.info(
        "No realized predictions yet. The dashboard fills in accuracy metrics "
        "as each prediction's target date passes and the actual close becomes known."
    )

st.divider()
pending = preds[preds["actual"].isna()]
if len(pending) > 0:
    st.caption(f"Pending prediction(s) awaiting tomorrow's actual close: {len(pending)}")
