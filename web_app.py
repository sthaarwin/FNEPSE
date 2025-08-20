from flask import Flask, render_template, request, jsonify
import json
import pandas as pd
from trading_signal_predictor import TradingSignalPredictor
from scraper.nepse_scraper.Scraper import Nepse_scraper
import traceback

app = Flask(__name__)

# Initialize the predictor and scraper
predictor = TradingSignalPredictor()
scraper = Nepse_scraper()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/market_data')
def market_data():
    try:
        gainers = scraper.get_top_gainer()
        losers = scraper.get_top_loser()
        turnover = scraper.get_top_turnover()
        
        return jsonify({
            'gainers': gainers,
            'losers': losers,
            'turnover': turnover
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_for_ticker', methods=['POST'])
def predict_for_ticker():
    try:
        data = request.json
        ticker = data.get('ticker')
        
        if not ticker:
            return jsonify({'error': 'Ticker symbol is required'}), 400
            
        # Fetch historical data using the scraper
        # Note: The format from the scraper needs to be adapted for the predictor
        history_data = scraper.get_price_history(ticker)
        
        if not history_data or len(history_data) < 20:
            return jsonify({'error': 'Not enough historical data to make a prediction. At least 20 data points are needed.'}), 400

        # The scraper returns a list of lists, convert it to the required format
        ohlcv_data = []
        for item in history_data:
            # Assuming the order is [timestamp, open, high, low, close, volume]
            # We need to confirm the exact format from the scraper's API response
            ohlcv_data.append({
                "open": item[1],
                "high": item[2],
                "low": item[3],
                "close": item[4],
                "volume": item[5]
            })

        # Make prediction
        signal, action, confidence, explanation = predictor.predict_signal(ohlcv_data)
        
        # Prepare response
        response = {
            'ticker': ticker,
            'signal': int(signal),
            'action': action,
            'confidence': float(confidence),
            'explanation': explanation,
            'data_points': len(ohlcv_data),
            'latest_price': ohlcv_data[-1]['close'],
            'price_change': None
        }
        
        # Calculate price change
        if len(ohlcv_data) > 1:
            prev_close = ohlcv_data[-2]['close']
            current_close = ohlcv_data[-1]['close']
            price_change = ((current_close - prev_close) / prev_close) * 100
            response['price_change'] = round(price_change, 2)
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
