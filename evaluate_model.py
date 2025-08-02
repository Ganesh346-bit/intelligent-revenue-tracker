
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm

# 1. Load & prepare data
df = pd.read_csv("data/sample_revenue.csv", parse_dates=["date"])
df["prev_revenue"] = df["revenue"].shift(1)
df = df.dropna()

# --- Baseline (LinearRegression) evaluation ---

X_full = df[["prev_revenue"]].values
y_full = df["revenue"].values
baseline = joblib.load("baseline_model.pkl")

# Full-sample
y_pred_full = baseline.predict(X_full)
mae_full = mean_absolute_error(y_full, y_pred_full)
rmse_full = mean_squared_error(y_full, y_pred_full, squared=False)

# Train/test split (last point as test)
train, test = df.iloc[:-1], df.iloc[-1:]
X_train, y_train = train[["prev_revenue"]].values, train["revenue"].values
X_test, y_test   = test[["prev_revenue"]].values, test["revenue"].values

lr = LinearRegression().fit(X_train, y_train)
y_pred_test = lr.predict(X_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)

# Print baseline results
print("=== Baseline (Linear Regression) ===")
print(f"Full MAE: {mae_full:.2f}, RMSE: {rmse_full:.2f}")
print(f"Test MAE: {mae_test:.2f}, RMSE: {rmse_test:.2f}\n")

# --- ARIMA (statsmodels) training & evaluation ---

# We'll model the series directly (no exog) with a simple ARIMA(1,1,0)
ts = df.set_index("date")["revenue"]
arima_mod = sm.tsa.ARIMA(ts, order=(1,1,0)).fit()

# Full-sample “in-sample” fit metrics
y_in = ts
y_hat_in = arima_mod.fittedvalues
mae_ar_in = mean_absolute_error(y_in, y_hat_in)
rmse_ar_in = mean_squared_error(y_in, y_hat_in, squared=False)

# One-step out-of-sample forecast for last point
y_hat_out = arima_mod.forecast(steps=1)
mae_ar_out = abs(y_hat_out.iloc[0] - ts.iloc[-1])
# (RMSE on 1 point is just abs difference)
rmse_ar_out = mae_ar_out

# Print ARIMA results
print("=== ARIMA(1,1,0) (statsmodels) ===")
print(f"In-sample MAE: {mae_ar_in:.2f}, RMSE: {rmse_ar_in:.2f}")
print(f"1-step forecast error (MAE/RMSE): {mae_ar_out:.2f}\n")

# 4. Persist the ARIMA model for your app
joblib.dump(arima_mod, "arima_statsmodel.pkl")
print("Saved arima_statsmodel.pkl")