import pandas as pd
import joblib
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1) Load & prepare
df = pd.read_csv("data/sample_revenue.csv", parse_dates=["date"])
series = df.set_index("date")["revenue"]

# 2) Split: use last month as test
train, test = series[:-1], series[-1:]

# 3) Fit a simple AR(1) model
model = ARIMA(train, order=(1, 0, 0)).fit()
joblib.dump(model, "arima_statsmodel.pkl")

# 4) Forecast next point
#    use get_forecast so we can pull .predicted_mean
fc = model.get_forecast(steps=1)
forecast = fc.predicted_mean.iloc[0]
actual   = test.iloc[0]

# 5) Compute errors
mae  = mean_absolute_error([actual], [forecast])
rmse = mean_squared_error([actual], [forecast]) ** 0.5

print(f"ARIMA(1,0,0) forecast for {test.index[0].strftime('%Y-%m')}: ${forecast:,.2f}")
print(f"ARIMA Test MAE:  ${mae:,.2f}")
print(f"ARIMA Test RMSE: ${rmse:,.2f}")
