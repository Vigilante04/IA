import yfinance as yf
import pandas as pd
from nsetools import Nse

# Initialize NSE object
nse = Nse()

def get_nifty_50_stocks():
    """Fetch NIFTY 50 stocks from NSE India"""
    nifty_50 = nse.get_index_constituents('NIFTY 50')
    return nifty_50[:5]  # Limiting to first 5 for quick analysis

def get_top_gainers():
    """Fetch top gainers from NSE India"""
    gainers = nse.get_top_gainers()
    return pd.DataFrame(gainers)[['symbol', 'ltp', 'netPrice']]

def assess_risk(beta, high_52, low_52, current):
    """Determine stock risk level"""
    if beta is not None:
        if beta < 0.8:
            return "Low Risk"
        elif 0.8 <= beta <= 1.2:
            return "Moderate Risk"
        else:
            return "High Risk"
    else:
        # If beta is unavailable, use price fluctuation as a proxy for risk
        volatility = (high_52 - low_52) / low_52
        if volatility < 0.3:
            return "Low Risk"
        elif 0.3 <= volatility <= 0.6:
            return "Moderate Risk"
        else:
            return "High Risk"

def get_stock_data(ticker):
    """Fetch stock data using Yahoo Finance and assess risk"""
    stock = yf.Ticker(ticker + ".NS")  # Adding ".NS" for NSE stocks in Yahoo Finance
    info = stock.info

    current_price = info.get("currentPrice", 0)
    high_52 = info.get("fiftyTwoWeekHigh", 0)
    low_52 = info.get("fiftyTwoWeekLow", 0)
    beta = info.get("beta", None)  # Beta represents stock volatility

    risk = assess_risk(beta, high_52, low_52, current_price)

    return {
        "Stock": ticker,
        "Market Cap": info.get("marketCap", "N/A"),
        "PE Ratio": info.get("trailingPE", "N/A"),
        "ROE": info.get("returnOnEquity", "N/A"),
        "52-Week High": high_52,
        "52-Week Low": low_52,
        "Current Price": current_price,
        "Beta": beta,
        "Risk Level": risk
    }

# Fetch stock data
nifty_50_stocks = get_nifty_50_stocks()
stock_details = [get_stock_data(stock) for stock in nifty_50_stocks]
df = pd.DataFrame(stock_details)

# Get top gainers
top_gainers = get_top_gainers()

# Display results
print("\nNIFTY 50 Stocks with Risk Analysis:")
print(df)

print("\nTop Gainers:")
print(top_gainers)
