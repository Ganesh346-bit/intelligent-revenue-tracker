import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

def train_baseline(csv_path: str, model_path: str = "baseline_model.pkl"):
    # 1) Load and prepare data
    df = pd.read_csv(csv_path, parse_dates=['date'])
    # Use last month’s revenue to predict this month
    df['prev_revenue'] = df['revenue'].shift(1)
    df = df.dropna()

    X = df[['prev_revenue']].values  # feature
    y = df['revenue'].values         # target

    # 2) Train model
    model = LinearRegression()
    model.fit(X, y)

    # 3) Save model
    joblib.dump(model, model_path)
    print(f"Trained baseline model saved to {model_path}")

    return model

if __name__ == "__main__":
    train_baseline("data/sample_revenue.csv")
