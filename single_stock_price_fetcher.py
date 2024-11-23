# independent_stock_price_fetcher.py

import yfinance as yf

def fetch_latest_stock_price(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    todays_data = stock.history(period='1d')
    if not todays_data.empty:
        latest_price_info = {
            'symbol': stock_symbol,
            'price': todays_data['Close'].iloc[-1],
            'date': str(todays_data.index[-1])
        }
        return latest_price_info
    else:
        return None
