import pandas as pd
import json
import requests
from io import StringIO
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

def stock_data(ticker):
    period1='1d'
    period2='2d'
    data = yf.Ticker(ticker).history(period=period1, interval='5m')
    tmp = yf.Ticker(ticker).history(period=period2)

    yest_prc = tmp.iloc[0]['Close']
    now_prc = [price for price in data['Close']]
    live_ret = [round(((now-yest_prc)/now) * 100, 2) for now in now_prc]

    data = data.reset_index(names='Time')
    data['Return'] = live_ret
    return data

def plot_price(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data['Time'], data['Return'], label='Intraday Change (%)', color='blue')

    # 3. Add Numbers (Annotations) for specific intervals
    # Example: Annotate every 6th point (every 30 minutes)
    for i in range(0, len(data), 5):
        x_val = data['Time'].iloc[i]
        y_val = data['Return'].iloc[i]
        
        # plt.text(x, y, string, ...)
        plt.text(x_val, y_val, 
                f"{y_val:.2f}%",       # Format to 2 decimal places
                fontsize=9, 
                ha='right',            # Horizontal alignment
                va='bottom',           # Vertical alignment
                color='red')           # Text color

        # Optional: Add a dot at the annotated point
        plt.scatter(x_val, y_val, color='red', s=20, zorder=5)
    last = round(((tmp.iloc[-1]['Close']-tmp.iloc[0]['Close'])/tmp.iloc[0]['Close'])*100,2)
    plt.text(data['Time'].iloc[-1], last, 
            f"{last:.2f}%", 
            fontsize=9, 
            ha='right', 
            va='bottom', 
            color='red')
    plt.scatter(data['Time'].iloc[-1], last, color='blue', s=20, zorder=5)

    # 4. Formatting the Graph
    plt.title("Intraday Stock Price Change (vs Yesterday Close)")
    plt.xlabel("Time")
    plt.ylabel("Change (%)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axhline(0, color='black', linewidth=1) # Add a zero line
    plt.legend()
    plt.show()

