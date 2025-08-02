import pandas as pd
import joblib
from statsmodels.tsa.arima.model import ARIMA

# Load data
df = pd.read_csv("data/sample_revenue.csv", parse_dates=["date"])
series = df.set_index("date")["revenue"]

# Train/test split
train = series[:-1]

# Fit AR(1)
model = ARIMA(train, order=(1, 0, 0)).fit()

# Save to disk
joblib.dump(model, "arima_statsmodel.pkl")
print("AR(1) model trained and saved.")

