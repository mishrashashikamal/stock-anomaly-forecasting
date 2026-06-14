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

# Fetch Data
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df['Daily_Return'] = df['Close'].pct_change() * 100
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    return df

# Load data
with st.spinner(f"Fetching {ticker} data..."):
    df = load_data(ticker, start_date, end_date)

# Key Metrics Row
st.subheader(f"📊 {ticker} Key Metrics")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Start Price", f"${df['Close'].iloc[0]:.2f}")
col2.metric("End Price", f"${df['Close'].iloc[-1]:.2f}")
col3.metric("Total Return", 
    f"{((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100):.2f}%")
col4.metric("Total Trading Days", len(df))

st.markdown("---")

# Charts Row 1
st.subheader("📈 Price & Volume Analysis")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['Date'], df['Close'], color='steelblue', linewidth=1.5)
    ax.set_title(f'{ticker} Closing Price', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df['Date'], df['Volume'], color='orange', alpha=0.6)
    ax.set_title(f'{ticker} Trading Volume', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Volume')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# Anomaly Detection
st.subheader("🚨 Anomaly Detection")

# Apply Isolation Forest
features = df[['Close', 'Volume']].copy()
iso_forest = IsolationForest(contamination=contamination, random_state=42)
df['Anomaly'] = iso_forest.fit_predict(features)
anomalies = df[df['Anomaly'] == -1]

# Anomaly Chart
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df['Date'], df['Close'], 
        color='steelblue', linewidth=1.5, label='Normal')
ax.scatter(anomalies['Date'], anomalies['Close'], 
           color='red', s=50, zorder=5, label='Anomaly')
ax.set_title(f'{ticker} Anomaly Detection', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Anomaly Stats
col1, col2 = st.columns(2)
col1.metric("Total Anomalies Detected", len(anomalies))
col2.metric("Anomaly Percentage", f"{len(anomalies)/len(df)*100:.1f}%")

# Anomaly Table
with st.expander("🔍 View Anomaly Details"):
    st.dataframe(
        anomalies[['Date', 'Close', 'Volume', 'Daily_Return']]
        .reset_index(drop=True)
        .style.format({
            'Close': '${:.2f}',
            'Daily_Return': '{:.2f}%'
        })
    )

st.markdown("---")


# Moving Averages & Daily Returns
st.subheader("📉 Moving Averages & Daily Returns")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['Date'], df['Close'],
            color='steelblue', linewidth=1, label='Price', alpha=0.7)
    ax.plot(df['Date'], df['MA20'],
            color='orange', linewidth=2, label='20-Day MA')
    ax.plot(df['Date'], df['MA50'],
            color='red', linewidth=2, label='50-Day MA')
    ax.set_title(f'{ticker} Moving Averages', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df['Daily_Return'].dropna(), bins=50,
                 color='steelblue', kde=True, ax=ax)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax.set_title(f'{ticker} Daily Returns Distribution', fontsize=14)
    ax.set_xlabel('Daily Return (%)')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")