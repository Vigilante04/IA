import sys
import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

# Initialize Tkinter GUI
root = tk.Tk()
root.title("Stock Price Predictor")
root.geometry("600x600")

# Stock Symbols including risky stocks
indian_tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
                  "HDFC.NS", "BHARTIARTL.NS", "SBILIFE.NS", "SBI.NS", "KOTAKBANK.NS", 
                  "LT.NS", "AXISBANK.NS", "MARUTI.NS", "ITC.NS", "WIPRO.NS", "TATAMOTORS.NS", 
                  "ADANIENT.NS", "ADANIPORTS.NS", "BAJAJFINANCE.NS", "BAJAJFINSERV.NS", "DRREDDY.NS", 
                  "HINDUNILVR.NS", "JSWSTEEL.NS", "ONGC.NS", "POWERGRID.NS", "TITAN.NS", 
                  "ULTRACEMCO.NS", "VEDL.NS", "COALINDIA.NS", "BANDHANBNK.NS"]

analyzer = SentimentIntensityAnalyzer()

def fetch_stock_data():
    stock = stock_var.get()
    future_date = future_date_var.get()

    if not stock or not future_date:
        messagebox.showerror("Error", "Please enter stock and future date.")
        return

    try:
        future_date_obj = datetime.strptime(future_date, "%Y-%m-%d")
        today = datetime.now()
        days_ahead = (future_date_obj - today).days

        if days_ahead <= 0:
            messagebox.showerror("Error", "Please enter a future date.")
            return

        # Fix: If stock is Reliance, use BSE ticker
        if stock == "RELIANCE.NS":
            stock = "500325.BSE"

        # Fetch historical data for the last 5 years
        start_date = (today - timedelta(days=1825)).strftime("%Y-%m-%d")
        df = yf.download(stock, start=start_date, end=today.strftime("%Y-%m-%d"))

        print(f"Downloaded Data for {stock}:")
        print(df.head())  # Print first few rows

        if df.empty:
            messagebox.showerror("Error", f"No data found for {stock}. Try again later.")
            return

        df["Close"] = df["Close"].fillna(method='ffill')

        # Sentiment Analysis (Dummy News Headlines)
        news_headlines = ["Stock is performing well", "Market is volatile", "Investors are positive about future growth"]
        sentiment_score = np.mean([analyzer.polarity_scores(headline)['compound'] for headline in news_headlines])

        # Data Preprocessing
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

        seq_length = 60
        x_train, y_train = create_sequences(scaled_data[:int(len(scaled_data)*0.8)], seq_length)

        # Build & Train LSTM Model
        model = build_lstm_model(seq_length)
        model.fit(x_train, y_train, epochs=10, batch_size=16, verbose=0, callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])

        # Predict Future Stock Price
        predicted_price = predict_specific_date(model, scaled_data, scaler, seq_length, days_ahead)
        
        # Classify Stock Growth
        growth = (predicted_price - df["Close"].iloc[-1]) / df["Close"].iloc[-1]

        if (growth > 0.1).any() and (sentiment_score > 0.2).any():
            classification = "High Growth (High Risk)"
        elif (growth > 0.05).any():
            classification = "Moderate Growth"
        else:
            classification = "Low Growth (Stable Investment)"

        # Display Results
        display_results(stock, future_date, predicted_price, classification)
        plot_stock(df, stock)

    except Exception as e:
        messagebox.showerror("Error", str(e))

def create_sequences(dataset, seq_length):
    x, y = [], []
    for i in range(len(dataset) - seq_length):
        x.append(dataset[i:i+seq_length])
        y.append(dataset[i+seq_length])
    return np.array(x), np.array(y)

def build_lstm_model(seq_length):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_specific_date(model, scaled_data, scaler, seq_length, days_ahead):
    input_seq = scaled_data[-seq_length:].reshape(1, seq_length, 1)
    
    for _ in range(days_ahead):
        predicted_price = model.predict(input_seq)[0]
        input_seq = np.append(input_seq[:, 1:, :], [[predicted_price]], axis=1)
    
    return scaler.inverse_transform(np.array(predicted_price).reshape(-1, 1))[0][0]

def display_results(stock, future_date, predicted_price, classification):
    results_text.delete(1.0, tk.END)
    results_text.insert(tk.END, f"Stock: {stock}\n")
    results_text.insert(tk.END, f"Predicted Price on {future_date}: ₹{predicted_price:.2f}\n")
    results_text.insert(tk.END, f"Stock Classification: {classification}\n")

def plot_stock(df, stock):
    plt.figure(figsize=(8, 5))
    plt.plot(df.index, df["Close"], label="Stock Price", color="blue")
    plt.xlabel("Date")
    plt.ylabel("Price (₹)")
    plt.title(f"{stock} Stock Price Over Time")
    plt.legend()
    plt.show()

# GUI Elements
tk.Label(root, text="Select Company:").pack()
stock_var = tk.StringVar()
stock_dropdown = ttk.Combobox(root, textvariable=stock_var, values=indian_tickers)
stock_dropdown.pack()

tk.Label(root, text="Enter Future Date (YYYY-MM-DD):").pack()
future_date_var = tk.StringVar()
tk.Entry(root, textvariable=future_date_var).pack()

tk.Button(root, text="Predict", command=fetch_stock_data).pack(pady=10)

results_text = tk.Text(root, height=6, width=60)
results_text.pack()

root.mainloop()
