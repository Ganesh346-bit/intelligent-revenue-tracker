# 🚀 Intelligent Revenue Tracker System

[![Docker Image CI](https://github.com/Ganesh346-bit/intelligent-revenue-tracker/actions/workflows/docker-image.yml/badge.svg)](https://github.com/Ganesh346-bit/intelligent-revenue-tracker/actions/workflows/docker-image.yml)

_An end-to-end tool to visualize monthly revenue, generate next-month forecasts, and monitor model performance._

---

## 📝 Overview

This project demonstrates how to:  
1. Ingest and clean historical revenue data  
2. Build a simple baseline forecast model (Linear Regression)  
3. Evaluate model performance with MAE & RMSE  
4. Serve an interactive dashboard via Streamlit  
5. Containerize the app with Docker for easy deployment  

**Live demo**: https://your-chosen-name.streamlit.app

---

## 📦 Tech Stack

- **Python** 3.10  
- **pandas** for data handling  
- **scikit-learn** for modeling  
- **statsmodels** for ARIMA experiments  
- **Streamlit** for dashboard  
- **Docker** for containerization  

---

## ⚙️ Local Setup

1. **Clone the repo**  
   ```bash
   git clone git@github.com:Ganesh346-bit/intelligent-revenue-tracker.git
   cd intelligent-revenue-tracker