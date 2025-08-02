# 🚀 Intelligent Revenue Tracker System

[![Docker Image CI](https://github.com/Ganesh346-bit/intelligent-revenue-tracker/actions/workflows/docker-image.yml/badge.svg)](https://github.com/Ganesh346-bit/intelligent-revenue-tracker/actions/workflows/docker-image.yml)

**📊 What is this project?**  
The *Intelligent Revenue Tracker* is a dynamic, end-to-end system that helps businesses visualize historical revenue, forecast future earnings (1–6 months ahead), and monitor model performance—all in one interactive dashboard.

**💡 Why is it needed?**  
In real-world business scenarios, predicting future revenue is essential for budgeting, planning inventory, allocating resources, and making confident financial decisions. Manual forecasting is time-consuming and error-prone. This tool automates the entire pipeline—from ingesting raw data to generating accurate forecasts and exporting reports—saving time and improving clarity.

**🔍 What makes it powerful?**  
- Supports multi-step forecasting (1–6 months ahead)  
- Gives confidence intervals for predictions  
- Enables live model retraining on uploaded data  
- Provides clear visualizations and performance metrics  
- Exportable reports and alerts for key thresholds  
- Runs in-browser and is Docker-ready for deployment  

---

## 🧠 Overview

This app helps you:

- 📈 Visualize monthly revenue trends
- 🔮 Forecast next 1–6 months using Linear Regression & ARIMA
- 📊 Monitor model accuracy with intuitive metrics
- 🛠️ Retrain models instantly by uploading new data
- 📥 Export forecasts as CSV
- 🚨 Get alerts if future revenue drops below threshold

---

## 🛠️ Tech Stack

- **Python** 3.11
- **pandas**, **scikit-learn**, **statsmodels**
- **Streamlit** for interactive dashboard
- **Docker** for containerization
- **GitHub Actions** for CI/CD

---

## 💻 Local Setup

```bash
# Clone the repo
git clone https://github.com/Ganesh346-bit/intelligent-revenue-tracker.git
cd intelligent-revenue-tracker

# Create and activate a virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 🧪 Features

### 1. 📥 Upload Data
- Upload your own revenue CSV (`date`, `revenue`).
- Or use the default sample dataset.

### 2. 📈 Historical Visualization
- Interactive line chart of past revenue.

### 3. 🔁 On-the-Fly Retraining
- Instantly retrains Linear & ARIMA models when new data is uploaded.

### 4. ⏩ Multi-Step Forecasting
- Forecast 1 to 6 months ahead.
- Toggle forecast horizon using a slider.

### 5. 📉 Confidence Intervals
- ARIMA forecast includes upper/lower bounds shown as shaded bands.

### 6. 📐 Model Metrics
- Full-form metrics with explanations:
  - **Mean Absolute Error (MAE):** Average of absolute errors.
  - **Root Mean Squared Error (RMSE):** Penalizes larger errors more.

### 7. 📊 Rolling Performance
- Track MAE/RMSE over time using a rolling window.

### 8. 🚨 Threshold Alert
- Choose a revenue threshold. If forecasted revenue drops below it, an alert appears.

### 9. 📤 Export Forecasts
- Download forecast data as a CSV file.

---

## 📦 Docker Setup

To build and run with Docker:

```bash
# Build image
docker build -t revenue-tracker .

# Run container
docker run --rm -p 8501:8501 revenue-tracker
```

---

## 📁 Folder Structure

```
.
├── app.py                # Streamlit dashboard
├── train_baseline.py     # Linear Regression training
├── train_arima.py        # ARIMA training
├── explore.py            # Data exploration
├── explore_forecast.py   # Forecast visualizations
├── evaluate_model.py     # MAE/RMSE evaluation
├── Dockerfile            # Docker config
├── requirements.txt      # Python dependencies
├── README.md             # Project doc
└── data/
    └── sample_revenue.csv
```

---

## 📈 Sample Forecast

<img src="https://raw.githubusercontent.com/Ganesh346-bit/intelligent-revenue-tracker/main/assets/sample_output.png" width="600"/>

---

## 📣 Credits

Built by [Sai Ganesh Kalmali](https://www.linkedin.com/in/ganeshkalmali)

