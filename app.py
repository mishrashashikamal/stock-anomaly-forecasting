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