import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# 1) Load your sample data
df = pd.read_csv("data/sample_revenue.csv", parse_dates=["date"])

# 2) Build the "previous month" feature
df["prev_revenue"] = df["revenue"].shift(1)
df_train = df.dropna()

# 3) Train the model
X = df_train[["prev_revenue"]]
y = df_train["revenue"]
model = LinearRegression().fit(X, y)

# 4) Save it out
joblib.dump(model, "baseline_model.pkl")
print("Saved baseline_model.pkl")
