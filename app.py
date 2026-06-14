# app.py — Stock Market Anomaly Detection & Trend Forecasting
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# Page Config
st.set_page_config(
    page_title="Stock Anomaly Detector",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📈 Stock Market Anomaly Detection & Trend Forecasting")
st.markdown("Detect unusual market behavior and forecast future prices using ML & ARIMA")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Settings")

# Stock selector
ticker = st.sidebar.selectbox(
    "Select Stock",
    ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META"],
    index=0
)

# Date range selector
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-31"))

# Contamination slider
contamination = st.sidebar.slider(
    "Anomaly Sensitivity", 
    min_value=0.01, 
    max_value=0.10, 
    value=0.05, 
    step=0.01,
    help="Higher value = more anomalies detected"
)

st.sidebar.markdown("---")
st.sidebar.markdown("Built by **Shashi Kamal Mishra**")