from flask import Flask, render_template, request, jsonify
import mysql.connector
from datetime import datetime

app = Flask(__name__)  # ✅ Correct Flask app initialization

# ✅ Risk Categorized Stocks (Replacing "HDFC.NS" with "SBIN.NS")
risk_categories = {
    "Low": ["HDFCBANK.NS", "INFY.NS", "TCS.NS", "HINDUNILVR.NS", "ITC.NS"],
    "Medium": ["RELIANCE.NS", "KOTAKBANK.NS", "SBIN.NS", "LT.NS", "TITAN.NS"],  # ✅ Updated stock
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
        return conn
    except mysql.connector.Error as err:
        print(f"❌ Database connection failed: {err}")
        return None

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
        future_date_obj = datetime.strptime(future_date, "%Y-%m-%d").date()
        today = datetime.now().date()

        if future_date_obj <= today:
            return jsonify({"error": "Please enter a valid future date"}), 400

        recommended_stocks = risk_categories.get(risk_category, [])
        predictions = []

        # ✅ Get a fresh database connection
        db = get_db_connection()
        if not db:
            return jsonify({"error": "Database connection failed"}), 500

        cursor = db.cursor()

        for stock in recommended_stocks:
            query = """
                SELECT predicted_price FROM stock_predictions 
                WHERE stock_symbol = %s AND `date` = %s AND risk_category = %s
            """
            cursor.execute(query, (stock, future_date_obj, risk_category))
            
            result = cursor.fetchone()
            print(f"📌 Fetching for: {stock} on {future_date_obj} → Result: {result}")  # ✅ Debugging

            predicted_price = float(result[0]) if result else "No Prediction Available"
            predictions.append({"stock": stock, "predicted_price": predicted_price})

        cursor.close()
        db.close()

        if all(p["predicted_price"] == "No Prediction Available" for p in predictions):
            return jsonify({"error": "No predictions available for this date and category"}), 404

        return jsonify({
            "risk_category": risk_category,
            "date": future_date,
            "predictions": predictions
        })

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
