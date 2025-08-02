import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the data
df = pd.read_csv('data/sample_revenue.csv', parse_dates=['date'])

# 2. Preview and stats
print(df.head())
print(df.describe())

# 3. Plot
plt.plot(df['date'], df['revenue'], marker='o')
plt.title("Monthly Revenue Over Time")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
