import streamlit as st
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Intelligent Revenue Tracker", layout="wide")
st.title("Intelligent Revenue Tracker")

# ── 1. Data input (upload or sample) ──
uploaded = st.file_uploader("Upload your revenue CSV", type="csv")
if uploaded:
    df = pd.read_csv(uploaded, parse_dates=["date"])
    st.success("Custom data loaded.")
else:
    df = pd.read_csv("data/sample_revenue.csv", parse_dates=["date"])
    st.info("Using sample data.")

# Ensure sorted
df = df.sort_values("date").reset_index(drop=True)

# ── 2. Historical chart ──
st.subheader("Historical Revenue")
st.line_chart(df.set_index("date")["revenue"])

# Prepare previous‐month feature
df["prev_revenue"] = df["revenue"].shift(1)
df_eval = df.dropna()

# ── 3. Linear Baseline Forecast ──
baseline_model = joblib.load("baseline_model.pkl")
last_rev = df["revenue"].iloc[-1]
lin_pred = baseline_model.predict([[last_rev]])[0]

st.subheader("Linear Baseline Forecast")
st.write(f"Last month: ${last_rev:,.2f}   →   Predicted next month: ${lin_pred:,.2f}")

# Compute linear held‐out & full‐sample metrics
X = df_eval[["prev_revenue"]].values
y = df_eval["revenue"].values

y_pred_all = baseline_model.predict(X)
full_mae = mean_absolute_error(y, y_pred_all)
full_rmse = mean_squared_error(y, y_pred_all) ** 0.5

train, test = df_eval.iloc[:-1], df_eval.iloc[-1:]
lin_split = LinearRegression().fit(train[["prev_revenue"]], train["revenue"])
y_pred_test = lin_split.predict(test[["prev_revenue"]])
test_mae = mean_absolute_error(test["revenue"], y_pred_test)
test_rmse = mean_squared_error(test["revenue"], y_pred_test) ** 0.5

st.subheader("Linear Model Performance")
st.markdown(f"- **Held-Out MAE:** ${test_mae:,.2f}")
st.markdown(f"- **Held-Out RMSE:** ${test_rmse:,.2f}")
st.markdown(f"- **Full-Sample MAE:** ${full_mae:,.2f}")
st.markdown(f"- **Full-Sample RMSE:** ${full_rmse:,.2f}")

# ── AR(1) Forecast via statsmodels ──
# Load the pre-trained AR(1) model
arima_model = joblib.load("arima_statsmodel.pkl")

# Generate a one-step forecast
arima_fc    = arima_model.get_forecast(steps=1)
arima_pred  = arima_fc.predicted_mean.iloc[0]

st.subheader("AR(1) Forecast")
st.write(f"Last month: ${last_rev:,.2f}   →   Predicted next month: ${arima_pred:,.2f}")

# ── 5. Alert slider ──
threshold = st.slider(
    "Alert if linear forecast below:", 
    min_value=0, 
    max_value=int(df["revenue"].max()), 
    value=int(last_rev),
    format="$%d"
)
if lin_pred < threshold:
    st.error(f"Predicted ${lin_pred:,.2f} is below your threshold of ${threshold:,}")
