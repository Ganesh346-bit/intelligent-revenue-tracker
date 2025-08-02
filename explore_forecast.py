import pandas as pd
import joblib

# 1. Load data and trained model
df = pd.read_csv('data/sample_revenue.csv', parse_dates=['date'])
model = joblib.load('baseline_model.pkl')

# 2. Prepare last known revenue and predict next month
last_rev = df['revenue'].iloc[-1]
predicted = model.predict([[last_rev]])

# 3. Print results
print(f"Last actual revenue: {last_rev}")
print(f"Predicted next month revenue: {predicted[0]:.2f}")
