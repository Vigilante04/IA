import mysql.connector
from datetime import datetime

# ✅ Risk Categorized Stocks
risk_categories = {
    "Low": ["HDFCBANK.NS", "INFY.NS", "TCS.NS", "HINDUNILVR.NS", "ITC.NS"],
    "Medium": ["RELIANCE.NS", "KOTAKBANK.NS", "SBIN.NS", "LT.NS", "TITAN.NS"],
    "High": ["ADANIENT.NS", "VEDL.NS", "TATAMOTORS.NS", "BANDHANBNK.NS", "COALINDIA.NS"]
}

# ✅ Function to get a fresh database connection
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host="mysql-125e2e26-investmentadviser-01.b.aivencloud.com",
            port=25286,
            user="avnadmin",
            password="AVNS_vXNFl9Ki3ktcBsr0Iub",
            database="defaultdb"
        )
        print("✅ Database connection successful.")
        return conn
    except mysql.connector.Error as err:
        print(f"❌ Database connection failed: {err}")
        return None

# ✅ Fetch Predictions from MySQL
def fetch_predictions(risk_category, future_date):
    try:
        future_date_obj = datetime.strptime(future_date, "%Y-%m-%d").date()
        today = datetime.now().date()

        if future_date_obj <= today:
            print("❗ Please enter a valid future date.")
            return

        recommended_stocks = risk_categories.get(risk_category, [])
        if not recommended_stocks:
            print("❗ Invalid risk category.")
            return
        
        predictions = []

        # ✅ Get Database Connection
        db = get_db_connection()
        if not db:
            return
        
        cursor = db.cursor()

        for stock in recommended_stocks:
            query = """
                SELECT predicted_price FROM stock_predictions 
                WHERE stock_symbol = %s AND `date` = %s AND risk_category = %s
            """
            cursor.execute(query, (stock, future_date_obj, risk_category))
            
            result = cursor.fetchone()
            print(f"📌 Fetching for: {stock} on {future_date_obj} → Result: {result}")

            predicted_price = float(result[0]) if result else "No Prediction Available"
            predictions.append({"stock": stock, "predicted_price": predicted_price})

        cursor.close()
        db.close()

        # ✅ Display Results
        if all(p["predicted_price"] == "No Prediction Available" for p in predictions):
            print("❗ No predictions available for this date and category.")
        else:
            print(f"\n📅 Predictions for {future_date_obj} ({risk_category} Risk Category)")
            for p in predictions:
                print(f" - {p['stock']}: {p['predicted_price']}")

    except Exception as e:
        print(f"❌ Error: {e}")

# ✅ Main Execution
if __name__ == "__main__":
    risk_category = input("Enter risk category (Low, Medium, High): ")
    future_date = input("Enter future date (YYYY-MM-DD): ")
    fetch_predictions(risk_category, future_date)
