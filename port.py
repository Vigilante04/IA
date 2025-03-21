from flask import Flask, render_template, request, jsonify
import os
import mysql.connector
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from scipy.optimize import minimize

app = Flask(__name__)

# ✅ Connect to MySQL Database using Environment Variables
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

db = mysql.connector.connect(
    host=DB_HOST,
    port=DB_PORT,
    user=DB_USER,
    password=DB_PASSWORD,
    database=DB_NAME
)
cursor = db.cursor()

# ✅ Risk Categorized Stocks
risk_categories = {
    "Low": ["HDFCBANK.NS", "INFY.NS", "TCS.NS", "HINDUNILVR.NS", "ITC.NS"],
    "Medium": ["RELIANCE.NS", "KOTAKBANK.NS", "BAJAJFINANCE.NS", "LT.NS", "TITAN.NS"],
    "High": ["ADANIENT.NS", "VEDL.NS", "TATAMOTORS.NS", "BANDHANBNK.NS", "COALINDIA.NS"]
}

@app.route("/")
def index():
    """Render the main page with a dropdown for risk category and a date selector."""
    return render_template("index.html", categories=risk_categories.keys())

@app.route("/predict", methods=["POST"])
def predict():
    """Fetch predictions from MySQL based on user input."""
    risk_category = request.form.get("risk_category")
    future_date = request.form.get("future_date")

    if not risk_category or not future_date:
        return jsonify({"error": "Please provide both risk category and future date"}), 400

    try:
        future_date_obj = datetime.strptime(future_date, "%Y-%m-%d")
        today = datetime.now()

        if future_date_obj <= today:
            return jsonify({"error": "Please enter a valid future date"}), 400

        recommended_stocks = risk_categories.get(risk_category, [])
        cursor.execute(
            f"""
            SELECT stock_symbol, predicted_price FROM stock_predictions
            WHERE stock_symbol IN ({','.join(['%s'] * len(recommended_stocks))})
            AND date = %s AND risk_category = %s
            """,
            (*recommended_stocks, future_date, risk_category)
        )
        results = cursor.fetchall()

        predictions = [
            {"stock": row[0], "predicted_price": row[1] if row[1] else "No Prediction Available"}
            for row in results
        ]
        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/optimize_portfolio", methods=["POST"])
def optimize_portfolio():
    """Optimize portfolio based on the selected risk category using Markowitz's Efficient Frontier."""
    risk_category = request.form.get("risk_category")
    
    if not risk_category:
        return jsonify({"error": "Please select a risk category"}), 400

    try:
        stocks = risk_categories.get(risk_category, [])
        
        # Fetch historical stock data (last 1 year)
        data = yf.download(stocks, period="1y", interval="1d")["Adj Close"].dropna()
        
        if data.empty:
            return jsonify({"error": "No sufficient data available for selected stocks"}), 500

        # Calculate log returns
        returns = np.log(data / data.shift(1)).dropna() + 1e-8

        # Expected returns and covariance matrix
        mean_returns = returns.mean()
        cov_matrix = returns.cov() + np.eye(len(stocks)) * 1e-6  # Regularization

        # Number of assets
        num_assets = len(stocks)

        # Objective function: minimize portfolio volatility
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        # Constraints: Weights must sum to 1
        constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
        
        # Bounds: Each stock weight is between 0 and 1
        bounds = tuple((0, 1) for _ in range(num_assets))

        # Initial guess (equal allocation)
        initial_weights = np.ones(num_assets) / num_assets

        # Perform optimization
        result = minimize(portfolio_volatility, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints)
        
        if not result.success:
            return jsonify({"error": "Optimization failed. Try again."}), 500

        optimized_weights = result.x  # Optimized stock allocation
        
        # Prepare response
        optimized_portfolio = {stocks[i]: round(optimized_weights[i] * 100, 2) for i in range(num_assets)}
        return jsonify({"optimized_portfolio": optimized_portfolio})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
