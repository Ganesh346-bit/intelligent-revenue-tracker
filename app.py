import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.title("Intelligent Revenue Tracker")

# Load data & model
df = pd.read_csv('data/sample_revenue.csv', parse_dates=['date'])
model = joblib.load('baseline_model.pkl')

# Historical chart
st.subheader("Historical Revenue")
st.line_chart(df.set_index('date')['revenue'])

# Forecast
last_rev = df['revenue'].iloc[-1]
pred = model.predict([[last_rev]])[0]
st.subheader("Next Month Forecast")
st.write(f"Last month: **{last_rev}**, Predicted next month: **{pred:.2f}**")

# Alert threshold slider
threshold = st.slider("Alert threshold:", 0, int(df['revenue'].max()), int(last_rev))
if pred < threshold:
    st.error(f"⚠️ Predicted ({pred:.2f}) below threshold ({threshold})")
