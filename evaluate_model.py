import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Load & prepare data
df = pd.read_csv('data/sample_revenue.csv', parse_dates=['date'])
df['prev_revenue'] = df['revenue'].shift(1)
df = df.dropna()

# Full-sample evaluation
X_full = df[['prev_revenue']].values
y_full = df['revenue'].values
model_full = joblib.load('baseline_model.pkl')
y_pred_full = model_full.predict(X_full)
mae_full = mean_absolute_error(y_full, y_pred_full)
rmse_full = mean_squared_error(y_full, y_pred_full) ** 0.5

# Train/test split: last point as test
train, test = df.iloc[:-1], df.iloc[-1:]
X_train = train[['prev_revenue']].values
y_train = train['revenue'].values
X_test = test[['prev_revenue']].values
y_test = test['revenue'].values

# Retrain model on training set
model_split = LinearRegression()
model_split.fit(X_train, y_train)
y_pred_test = model_split.predict(X_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = mean_squared_error(y_test, y_pred_test) ** 0.5

# Print results
print("=== Full-Sample Performance ===")
print(f"MAE:  {mae_full:.2f}")
print(f"RMSE: {rmse_full:.2f}\n")
print("=== Train/Test Split Performance ===")
print(f"Test MAE:  {mae_test:.2f}")
print(f"Test RMSE: {rmse_test:.2f}")
