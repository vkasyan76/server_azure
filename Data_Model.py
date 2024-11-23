import os
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Dense, Conv1D, MaxPooling1D
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta

def format_date(date_obj):
    """Convert datetime object to DD-MM-YYYY format."""
    return date_obj.strftime('%d-%m-%Y')

def mean_absolute_percentage_error(y_true, y_pred): 
    """Compute Mean Absolute Percentage Error (MAPE)"""
    y_true, y_pred = np.array(y_true), np.array(y_pred) 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Helper function to get Buy/Sell signals
def get_trade_signals(predicted_prices_data, actual_prices_data):
    Buy, Sell = [], []

    for i in range(2, len(predicted_prices_data)):
        if predicted_prices_data[i] > actual_prices_data[i] and predicted_prices_data[i-1] < actual_prices_data[i-1]:
            Buy.append(i)
        elif predicted_prices_data[i] < actual_prices_data[i] and predicted_prices_data[i-1] > actual_prices_data[i-1]:
            Sell.append(i)
    
    return Buy, Sell

# Helper function to calculate profits
def calculate_profits(Buy, Sell, actual_prices_data):
    BuyPrices = [actual_prices_data[i] for i in Buy]
    SellPrices = [actual_prices_data[i] for i in Sell]
    trade_profits = [(SellPrices[i] - BuyPrices[i]) for i in range(len(SellPrices))]
    
    return BuyPrices, SellPrices, trade_profits


# Helper function to calculate profits with dates
def calculate_profits_with_dates(Buy, Sell, actual_prices_data, dates):
    BuyPrices = [actual_prices_data[i] for i in Buy]
    SellPrices = [actual_prices_data[i] for i in Sell]
    
    # Ensure that we do not have a sell before the first buy and a buy after the last sell
    if Sell and Buy:
        if Sell[0] < Buy[0]:
            Sell.pop(0)
        if Buy[-1] > Sell[-1]:
            Buy.pop()

    BuyPrices = [actual_prices_data[i] for i in Buy]
    SellPrices = [actual_prices_data[i] for i in Sell]
    BuyDates = [dates[i] for i in Buy]
    SellDates = [dates[i] for i in Sell]
    # trade_profits = [(SellPrices[i] - BuyPrices[i]) for i in range(len(SellPrices))]
    # trade_profits = [{'Date': format_date(BuyDates[i]), 'Profit': trade_profits[i]} for i in range(len(trade_profits))]
        # Calculate trade profits
    trade_profits = [{'Date': format_date(SellDates[i]), 'Profit': SellPrices[i] - BuyPrices[i]} for i in range(len(SellPrices))]
    
    # Zip dates with prices for better output
    # BuySignals = list(zip(BuyDates, BuyPrices))
    # SellSignals = list(zip(SellDates, SellPrices))
    BuySignals = [{'Date': format_date(BuyDates[i]), 'BuyPrice': BuyPrices[i]} for i in range(len(BuyPrices))]
    SellSignals = [{'Date': format_date(SellDates[i]), 'SellPrice': SellPrices[i]} for i in range(len(SellPrices))]

    
    return BuySignals, SellSignals, trade_profits

def train_and_predict_model_loop(stock_ticker):

    best_mape = float('inf')
    best_model_results = {}

    end_date = datetime.now() - relativedelta(months=6)  
    # Fetch data for the past 2 years
    start_date = end_date - relativedelta(years=2)  
    
    # Fetch stock data
    data = yf.Ticker(stock_ticker).history(start=start_date, end=end_date)
    data.index = pd.to_datetime(data.index)

    # Calculate momentum indicators and rolling window features
    data['Momentum'] = data['Close'].diff()  # Simple difference in price to represent momentum
    data['ROC'] = data['Close'].pct_change() * 100  # Rate of Change
    data['MACD'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()  # MACD
    window_size = 5  # Example window size
    data['Rolling_Mean'] = data['Close'].rolling(window=window_size).mean()
    data['Rolling_Std'] = data['Close'].rolling(window=window_size).std()
    data.fillna(method='bfill', inplace=True)

    feature_columns = ['Close', 'Volume', 'ROC', 'MACD']

    # Preprocessing
    scaler = MinMaxScaler(feature_range=(0, 1))
    # scaled_data = scaler.fit_transform(data[['Close', 'Volume', 'Close_index', 'Close_industry']].fillna(0))
    data_scaled = scaler.fit_transform(data[feature_columns])
  
    # Creating training data
    x_train = []
    y_train = []
    prediction_days = 10

    for x in range(prediction_days, len(data_scaled)):
        x_train.append(data_scaled[x-prediction_days:x, :])  # Include all features
        y_train.append(data_scaled[x, 0])  # Still predicting 'Close'

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], data_scaled.shape[1]))   # # of features in data_scaled 

    # LSTM model architecture with CNN
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], data_scaled.shape[1])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')


    for run in range(5):
    # Fit the model
        model.fit(x_train, y_train, epochs=25, batch_size=32)

        # Preparing test data, using the same steps as training data to include the momentum and rolling features
        test_start = end_date
        test_end = dt.datetime.now()
        # Fetch and prepare test data for the stock
        test_data = yf.Ticker(stock_ticker).history(start=test_start, end=test_end)

        actual_prices = test_data['Close']
    
        # Calculate momentum indicators and rolling window features
        test_data['Momentum'] = test_data['Close'].diff()  # Simple difference in price to represent momentum
        test_data['ROC'] = test_data['Close'].pct_change() * 100  # Rate of Change
        test_data['MACD'] = test_data['Close'].ewm(span=12).mean() - test_data['Close'].ewm(span=26).mean()  # MACD
        window_size = 5  # Example window size
        test_data['Rolling_Mean'] = test_data['Close'].rolling(window=window_size).mean()
        test_data['Rolling_Std'] = test_data['Close'].rolling(window=window_size).std()
        test_data.fillna(method='bfill', inplace=True)

        test_data_scaled = scaler.transform(test_data[feature_columns].fillna(0))
    
        # Combine the last part of the training data with the scaled test data
        model_inputs = np.concatenate((data_scaled[-prediction_days:], test_data_scaled), axis=0)
    
        x_test = []
        for x in range(prediction_days, len(model_inputs)):
            x_test.append(model_inputs[x-prediction_days:x, :])
    
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], test_data_scaled.shape[1]))  # Reshaping to include 4 features 
        
        predicted_prices = model.predict(x_test)
    
        # Create a dummy array with 4 features for inverse transformation
        prediction_for_inverse = np.zeros((predicted_prices.shape[0], data_scaled.shape[1]))
        prediction_for_inverse[:, 0] = predicted_prices.flatten()  # Fill in the predictions in the first feature
        predicted_prices = scaler.inverse_transform(prediction_for_inverse)[:, 0]  # Inverse transform

        # Calculate the performance metrics for the current run
        actual_prices_arr = test_data['Close'].values
        current_mape = mean_absolute_percentage_error(actual_prices_arr, predicted_prices)

        print("Current Mape", current_mape)

        # If this run's MAPE is the best so far, store its results
        if current_mape < best_mape:
            best_mape = current_mape
            real_data = [model_inputs[len(model_inputs) - prediction_days:len(model_inputs), :]]
            real_data = np.array(real_data)
            real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], data_scaled.shape[1]))
            future_prediction = model.predict(real_data)

            best_model_results = {
                'predicted_prices': predicted_prices,
                'actual_prices': actual_prices_arr,
                'mape': current_mape,
                'model_inputs': model_inputs,  # Save this for future prediction
                'scaler': scaler,  # Save the scaler
                'test_data': test_data,  # Save the test_data
                'model': model,  # Store the current best model
                'real_data': real_data,
                'future_prediction': future_prediction,
            }

    test_data = best_model_results['test_data']
    predicted_prices = best_model_results['predicted_prices']
    model_inputs = best_model_results['model_inputs']  # Update model_inputs
    scaler = best_model_results['scaler']  # Update scaler
    real_data = best_model_results['real_data']
    future_prediction = best_model_results['future_prediction']


    predicted_df = pd.DataFrame({
        'Date': test_data.index,
        'Predicted_Price': predicted_prices
    })

    # future_prediction = model.predict(real_data)
    future_prediction_for_inverse = np.zeros((future_prediction.shape[0], data_scaled.shape[1]))
    future_prediction_for_inverse[:, 0] = future_prediction.flatten()
    future_prediction = scaler.inverse_transform(future_prediction_for_inverse)[:, 0]  # Inverse transform only the first feature (Close prices)

    latest_closing_price = test_data['Close'].iloc[-1]

    actual_prices_data = {
        'Date': test_data.index.tolist(),
        'Actual_Price': test_data['Close'].tolist()
    }

    predicted_prices_data = {
        'Date': predicted_df['Date'].tolist(),
        'Predicted_Price': predicted_prices.tolist()
    }

    combined_data = []
    for index in range(len(actual_prices_data["Date"])):
        combined_data.append({
            "Date": format_date(actual_prices_data["Date"][index]),
            "Actual_Price": actual_prices_data["Actual_Price"][index],
            "Predicted_Price": predicted_prices_data["Predicted_Price"][index]
        })
        
    actual_prices_arr = np.array(actual_prices_data["Actual_Price"])
    predicted_prices_arr = np.array(predicted_prices_data["Predicted_Price"])

    mape = mean_absolute_percentage_error(actual_prices_arr, predicted_prices_arr)
    correlation_coefficient = np.corrcoef(actual_prices_arr, predicted_prices_arr)[0, 1]

    Buy, Sell = get_trade_signals(predicted_prices_data['Predicted_Price'], actual_prices_data['Actual_Price'])
    # BuyPrices, SellPrices, trade_profits = calculate_profits(Buy, Sell, actual_prices_data['Actual_Price'])

    BuySignals, SellSignals, trade_profits = calculate_profits_with_dates(
    Buy, 
    Sell, 
    actual_prices_data['Actual_Price'], 
    actual_prices_data['Date']  # Assuming this is a list of dates corresponding to the actual prices
)

  # Calculate Buy/Sell signals and profits
    Buy, Sell = get_trade_signals(predicted_prices, actual_prices)
    BuySignals, SellSignals, trade_profits = calculate_profits_with_dates(
        Buy, Sell, actual_prices, test_data.index
    )

    # Combine all trade data
    trade_dates = sorted(set([d['Date'] for d in BuySignals + SellSignals + trade_profits]), 
                         key=lambda x: datetime.strptime(x, '%d-%m-%Y'))
    

    trade_data = []
    for date in trade_dates:
        buy_data = next((item for item in BuySignals if item['Date'] == date), {'BuyPrice': 0})
        sell_data = next((item for item in SellSignals if item['Date'] == date), {'SellPrice': 0})
        profit_data = next((item for item in trade_profits if item['Date'] == date), {'Profit': 0})
        trade_data.append({
            'Date': date,
            'BuyPrice': buy_data.get('BuyPrice', 0),
            'SellPrice': sell_data.get('SellPrice', 0),
            'Profit': profit_data.get('Profit', 0)
        })


    return combined_data, future_prediction, latest_closing_price, mape, correlation_coefficient, BuySignals, SellSignals, trade_profits, trade_data


# stock_ticker = "BABA"


# results = train_and_predict_model_loop(stock_ticker)

# Access the results by their index in the tuple
# combined_data = results[0]
# future_prediction = results[1]
# latest_closing_price = results[2]
# mape = results[3]
# correlation_coefficient = results[4]
# BuySignals = results[5]
# SellSignals = results[6]
# trade_profits = results[7]


# Now you can print them or use them as needed
# print("Buy Signals (Date, Price):", BuySignals)
# print("Sell Signals (Date, Price):", SellSignals)
# print("Trade Profits:", trade_profits)
# print("Future Prediction:", future_prediction)

# print("Combined Data", combined_data)


# print("MAPE", mape)