# 🚀 Intelligent Revenue Tracker System

[![Docker Image CI](https://github.com/Ganesh346-bit/intelligent-revenue-tracker/actions/workflows/docker-image.yml/badge.svg)](https://github.com/Ganesh346-bit/intelligent-revenue-tracker/actions/workflows/docker-image.yml)

*An end-to-end tool to visualize monthly revenue, generate next-month forecasts, and monitor model performance.*

---

## 📝 Overview

This project demonstrates how to:

1. Ingest and clean historical revenue data
2. Build a simple baseline forecast model (Linear Regression)
3. Evaluate model performance with MAE & RMSE
4. Serve an interactive dashboard via Streamlit
5. Containerize the app with Docker for easy deployment

**Live demo**: [https://kuqizdymb3icpgsbe6zqeb.streamlit.app]

---

## 📦 Tech Stack

* **Python** 3.11
* **pandas** for data handling
* **scikit-learn** for modeling
* **statsmodels** for ARIMA forecasting
* **Streamlit** for dashboard UI
* **Docker** for containerization
* **GitHub Actions** for CI/CD

---

## ⚙️ Run Locally

1. **Clone the repo**

   ```bash
   git clone https://github.com/Ganesh346-bit/intelligent-revenue-tracker.git
   cd intelligent-revenue-tracker
   ```
2. **Create and activate a virtual environment**

   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**

   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```
4. **Run the app**

   ```bash
   streamlit run app.py
   ```
5. **View** at `http://localhost:8501`

---

## 🐳 Docker

Build and run with Docker:

```bash
# Build the image
docker build -t revenue-tracker:latest .

# Run the container
docker run --rm -p 8501:8501 revenue-tracker:latest
```

Open your browser at `http://localhost:8501`.

---

## 🔄 CI/CD

* **Docker Image CI**: automatically builds on every push. See [workflow details](https://github.com/Ganesh346-bit/intelligent-revenue-tracker/actions/workflows/docker-image.yml).

---

## 📄 License

Licensed under the MIT License.
