import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Intelligent Revenue Tracker", layout="wide")

# ── Load data & baseline model ──
df = pd.read_csv("data/sample_revenue.csv", parse_dates=["date"])
model = joblib.load("baseline_model.pkl")

# ── Historical Revenue ──
st.subheader("Historical Revenue")
st.line_chart(df.set_index("date")["revenue"])

# ── Forecast Section ──
last_rev = df["revenue"].iloc[-1]
pred     = model.predict([[last_rev]])[0]

st.subheader("Next Month Forecast")
st.write(f"Last month: ${last_rev:,.2f}    →    Predicted next month: ${pred:,.2f}")


# ── Held-Out Test Performance ──
df["prev_revenue"] = df["revenue"].shift(1)
df_eval = df.dropna()
train, test = df_eval.iloc[:-1], df_eval.iloc[-1:]

X_train, y_train = train[["prev_revenue"]], train["revenue"]
X_test,  y_test  = test[["prev_revenue"]],  test["revenue"]

model_split = LinearRegression().fit(X_train, y_train)
y_pred_test = model_split.predict(X_test)

test_mae  = mean_absolute_error(y_test, y_pred_test)
test_rmse = mean_squared_error(y_test, y_pred_test) ** 0.5

st.subheader("Held-Out Test Performance")
st.markdown(f"- **Test MAE:**  ${test_mae:,.2f}")
st.markdown(f"- **Test RMSE:** ${test_rmse:,.2f}")

# ── Full-Sample Performance ──
y_true     = df_eval["revenue"].values
y_pred_all = model.predict(df_eval[["prev_revenue"]].values)

full_mae  = mean_absolute_error(y_true, y_pred_all)
full_rmse = mean_squared_error(y_true, y_pred_all) ** 0.5

st.subheader("Full-Sample Performance")
st.markdown(f"- **MAE:**  ${full_mae:,.2f}")
st.markdown(f"- **RMSE:** ${full_rmse:,.2f}")

# ── Alert Slider ──
threshold = st.slider(
    "Alert if forecast below:", 
    min_value=0, 
    max_value=int(df["revenue"].max()), 
    value=int(last_rev),
    format="$%d"
)
if pred < threshold:
    st.error(f"Predicted ${pred:,.2f} is below your threshold of ${threshold:,}")
