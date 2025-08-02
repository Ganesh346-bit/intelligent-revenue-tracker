import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import math
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

# ── Page config ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Intelligent Revenue Tracker", layout="wide")
st.title("Intelligent Revenue Tracker")

# ── 1. Data upload & load ───────────────────────────────────────────────
uploaded = st.file_uploader("Upload your revenue CSV", type="csv")
if uploaded:
    df = pd.read_csv(uploaded, parse_dates=["date"])
    st.success("Custom data loaded & retraining models…")
else:
    df = pd.read_csv("data/sample_revenue.csv", parse_dates=["date"])
    st.info("Using sample data.")

df = df.sort_values("date").reset_index(drop=True)
df["prev_revenue"] = df["revenue"].shift(1)
df_eval = df.dropna().reset_index(drop=True)

# ── 2. Historical chart ─────────────────────────────────────────────────
st.subheader("Historical Revenue")
st.line_chart(df.set_index("date")["revenue"])

# ── 3. On-the-fly retrain baseline & ARIMA ──────────────────────────────
lin_model = LinearRegression().fit(
    df_eval[["prev_revenue"]], df_eval["revenue"]
)
arima_model = ARIMA(df["revenue"], order=(1, 0, 0)).fit()
last_rev = df["revenue"].iloc[-1]

# ── 4. Multi-step forecast UI ───────────────────────────────────────────
st.subheader("Multi-Step Forecast")
horizon = st.slider("Forecast horizon (months ahead)", 1, 6, 1)

# Linear multi-step
lin_preds = []
x = last_rev
for _ in range(horizon):
    p = lin_model.predict([[x]])[0]
    lin_preds.append(p)
    x = p

# ARIMA multi-step & confidence
arima_fc = arima_model.get_forecast(steps=horizon)
arima_preds = arima_fc.predicted_mean.values
ci = arima_fc.conf_int()

future_dates = [
    df["date"].iloc[-1] + pd.DateOffset(months=i+1)
    for i in range(horizon)
]
fc_df = pd.DataFrame({
    "date": future_dates,
    "Linear": lin_preds,
    "ARIMA": arima_preds,
    "ARIMA_lower": ci.iloc[:, 0].values,
    "ARIMA_upper": ci.iloc[:, 1].values,
}).set_index("date")

# Plot with Matplotlib to shade CI correctly
fig, ax = plt.subplots(figsize=(6, 3))
fc_df[["Linear", "ARIMA"]].plot(ax=ax)
ax.fill_between(
    fc_df.index,
    fc_df["ARIMA_lower"],
    fc_df["ARIMA_upper"],
    color="orange", alpha=0.2, label="ARIMA CI"
)
ax.legend(loc="upper left")
ax.set_ylabel("Revenue")
ax.set_xlabel("Date")
st.pyplot(fig)

# ── 5. Model Performance (Full History) ─────────────────────────────────
y_true = df_eval["revenue"].values
y_lin  = lin_model.predict(df_eval[["prev_revenue"]])
y_ari  = arima_model.predict(start=1, end=len(df_eval))

st.subheader("Model Performance (Full History)")
st.write("Here’s how each model did when fitting the entire dataset:")

col1, col2 = st.columns(2)
with col1:
    lin_mae  = mean_absolute_error(y_true, y_lin)
    lin_rmse = math.sqrt(mean_squared_error(y_true, y_lin))
    st.markdown(
        f"- **Linear Regression Mean Absolute Error (MAE):** ${lin_mae:,.2f}  \n"
        "  • Average absolute difference between actual and predicted values."
    )
    st.markdown(
        f"- **Linear Regression Root Mean Squared Error (RMSE):** ${lin_rmse:,.2f}  \n"
        "  • Square root of the average squared difference, penalizing large errors."
    )

with col2:
    ari_mae  = mean_absolute_error(y_true, y_ari)
    ari_rmse = math.sqrt(mean_squared_error(y_true, y_ari))
    st.markdown(
        f"- **ARIMA Mean Absolute Error (MAE):** ${ari_mae:,.2f}  \n"
        "  • Average absolute deviation of ARIMA forecasts from actuals."
    )
    st.markdown(
        f"- **ARIMA Root Mean Squared Error (RMSE):** ${ari_rmse:,.2f}  \n"
        "  • RMSE for ARIMA, highlighting larger deviations."
    )

# ── 6. Rolling performance ──────────────────────────────────────────────
st.subheader("Rolling MAE/RMSE Over Time")
if len(df_eval) >= 3:
    win = st.slider("Window size (months)", 2, len(df_eval) - 1, 3)
    dates, maes, rmses = [], [], []
    for i in range(win, len(df_eval)):
        sub = df_eval.iloc[: i + 1]
        X_sub, y_sub = sub[["prev_revenue"]], sub["revenue"]
        m = LinearRegression().fit(X_sub[:-1], y_sub[:-1])
        yhat = m.predict([X_sub.iloc[-1]])[0]
        dates.append(sub["date"].iloc[-1])
        maes.append(abs(y_sub.iloc[-1] - yhat))
        rmses.append(abs(y_sub.iloc[-1] - yhat))
    perf = pd.DataFrame({"MAE": maes, "RMSE": rmses}, index=dates)
    st.line_chart(perf)
else:
    st.info("Not enough data to compute rolling performance.")

# ── 7. Export Forecast ─────────────────────────────────────────────────
st.subheader("Export Forecast")
csv_buf = io.StringIO()
out = fc_df.reset_index()[["date", "Linear", "ARIMA"]]
out.to_csv(csv_buf, index=False)
st.download_button("📥 Download Forecast CSV", csv_buf.getvalue(), "forecast.csv", "text/csv")

# ── 8. Threshold alert ──────────────────────────────────────────────────
threshold = st.slider(
    "Alert if linear forecast below",
    min_value=0,
    max_value=int(df["revenue"].max()),
    value=int(last_rev),
    format="$%d"
)
if lin_preds[0] < threshold:
    st.error(f"⚠️ Predicted ${lin_preds[0]:,.2f} is below your threshold!")