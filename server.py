import os
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import yfinance as yf
from Data_Model import train_and_predict_model_loop
from single_stock_price_fetcher import fetch_latest_stock_price

import logging
logging.basicConfig(level=logging.DEBUG)

# app instance
app = Flask(__name__)


CORS(app, resources={r"/api/*": {"origins": "*", "allow_headers": ["Content-Type", "Referer", "User-Agent"]}})

# /api/home
    
# Stock prediction Model API Loop:

@app.route("/api/train-model-loop", methods=['POST'])
def train_model_loop():
    try:
        data = request.get_json()
        print("Data received:", data)  # Add this line to debug
        stock_ticker = data.get("stock_ticker")

        # Add checks here
        if stock_ticker is None:
            print("The ticker is None:", stock_ticker)
            return jsonify({"error": "Ticker information is incomplete"}), 400


        combined_data, future_prediction, latest_closing_price, mape, correlation_coefficient, BuySignals, SellSignals, trade_profits, trade_data = train_and_predict_model_loop(stock_ticker)
 
        return jsonify({"stockTicker": stock_ticker, "latestClosingPrice": latest_closing_price, "modelPrediction": float(future_prediction[0]),'combinedData': combined_data,  "MAPE": mape, "correlationCoeff": correlation_coefficient, "Buys": BuySignals, "Sells": SellSignals, "Profits": trade_profits, 'tradeData': trade_data})

    except Exception as e:
        print("Exception:", str(e))  # <-- this is the new line
        # import traceback
        # traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/single-latest-price/<string:symbol>", methods=['GET'])
def independent_latest_price(symbol):
    price_info = fetch_latest_stock_price(symbol)
    if price_info:
        return jsonify(price_info)
    else:
        return jsonify({"error": "Price information not available for the specified symbol."}), 404


@app.errorhandler(500)
def internal_error(error):
    print("Server Error:", error)
    return jsonify({"error": "Internal server error"}), 500

import logging

logging.basicConfig(level=logging.DEBUG)


if __name__ == "__main__":
    # Use the 'WEBSITES_PORT' environment variable on Azure, or default to 8000 if running locally
    port = int(os.environ.get('WEBSITES_PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
