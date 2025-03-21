import mysql.connector
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from datetime import datetime, timedelta

# ✅ Database Connection
db = mysql.connector.connect(
    host="mysql-125e2e26-investmentadviser-01.b.aivencloud.com",
    port=25286,
    user="avnadmin",
    password="AVNS_vXNFl9Ki3ktcBsr0Iub",
    database="defaultdb"
)
cursor = db.cursor()

# ✅ Define Date Variables
today = datetime.now()
current_date = today.strftime("%Y-%m-%d")  # Today's date
start_date = (today - timedelta(days=1825)).strftime("%Y-%m-%d")  # Last 5 years
future_dates = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 31)]  # Next 30 days

# ✅ Ensure Table Exists
cursor.execute("""
    CREATE TABLE IF NOT EXISTS stock_predictions (
        id INT AUTO_INCREMENT PRIMARY KEY,
        stock_symbol VARCHAR(20),
        `date` DATE,
        predicted_price FLOAT,
        risk_category VARCHAR(10),
        UNIQUE (stock_symbol, `date`)  -- ✅ Prevent duplicate entries
    )
""")
db.commit()

# ✅ Delete existing predictions for today
cursor.execute("DELETE FROM stock_predictions WHERE `date` = %s", (current_date,))
db.commit()

# ✅ Risk Categorized Stocks
risk_categories = {
    "Low": ["HDFCBANK.NS", "INFY.NS", "TCS.NS", "HINDUNILVR.NS", "ITC.NS"],
    "Medium": ["RELIANCE.NS", "KOTAKBANK.NS", "SBIN.NS", "LT.NS", "TITAN.NS"],
    "High": ["ADANIENT.NS", "VEDL.NS", "TATAMOTORS.NS", "BANDHANBNK.NS", "COALINDIA.NS"]
}

def create_lstm_model(seq_length):
    """Builds an optimized LSTM model."""
    model = Sequential([
        Input(shape=(seq_length, 1)),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_stock_price(model, scaled_data, scaler, seq_length):
    """Predicts stock prices for the next 30 days using LSTM."""
    input_seq = scaled_data[-seq_length:].reshape(1, seq_length, 1)
    predicted_prices = []

    for _ in range(30):  # Predict for next 30 days
        pred_price = model.predict(input_seq, verbose=0)[0]
        predicted_prices.append(pred_price)
        input_seq = np.append(input_seq[:, 1:, :], [[pred_price]], axis=1)

    return scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1)).flatten()

def store_stock_data():
    """Fetches stock data and stores predictions for the next 30 days in MySQL."""
    for risk, stocks in risk_categories.items():
        for stock in stocks:
            print(f"📌 Fetching data for {stock} ({risk} risk)...")
            try:
                df = yf.download(stock, start=start_date, end=today.strftime("%Y-%m-%d"))
                if df.empty:
                    print(f"❌ No data available for {stock}")
                    continue

                df["Close"] = df["Close"].ffill()  # Forward fill missing values

                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

                seq_length = 60
                x_train, y_train = [], []
                for i in range(len(scaled_data) - seq_length):
                    x_train.append(scaled_data[i:i+seq_length])
                    y_train.append(scaled_data[i+seq_length])

                x_train, y_train = np.array(x_train), np.array(y_train)

                model = create_lstm_model(seq_length)
                model.fit(x_train, y_train, epochs=10, batch_size=16, verbose=0, validation_split=0.1)

                predicted_prices = predict_stock_price(model, scaled_data, scaler, seq_length)

                # ✅ Insert predictions for the next 30 days
                sql_query = """
                    INSERT INTO stock_predictions (stock_symbol, `date`, predicted_price, risk_category)
                    VALUES (%s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE predicted_price = VALUES(predicted_price), risk_category = VALUES(risk_category)
                """
                values = [(stock, future_date, float(pred_price), risk) for future_date, pred_price in zip(future_dates, predicted_prices)]

                cursor.executemany(sql_query, values)
                db.commit()
                print(f"✔ Stored predictions for {stock}")

            except Exception as e:
                print(f"❌ Error fetching data for {stock}: {e}")

store_stock_data()

# ✅ Close Connections
cursor.close()
db.close()
print("✅ Next 30 days' predictions stored successfully!")
