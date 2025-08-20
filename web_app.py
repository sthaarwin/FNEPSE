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
            # Handle the data format more robustly
            try:
                ohlcv_data.append({
                    "open": float(item[1]) if item[1] else 0.0,
                    "high": float(item[2]) if item[2] else 0.0,
                    "low": float(item[3]) if item[3] else 0.0,
                    "close": float(item[4]) if item[4] else 0.0,
                    "volume": float(item[5]) if len(item) > 5 and item[5] else 0.0
                })
            except (IndexError, ValueError, TypeError):
                # Skip malformed data points
                continue

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

@app.route('/recommendations')
def recommendations():
    try:
        # Get basic market data - reduced API calls
        top_turnover = scraper.get_top_turnover()[:15]  # Reduced from 50 to 15
        top_gainers = scraper.get_top_gainer()[:10]     # Reduced from 20 to 10
        top_losers = scraper.get_top_loser()[:10]       # Reduced from 20 to 10
        
        # Create a smaller list of stocks to analyze
        stock_list = []
        
        # Add top turnover stocks (most liquid)
        for stock in top_turnover:
            stock_list.append(stock['symbol'])
        
        # Add top gainers and losers for market sentiment
        for stock in top_gainers:
            if stock['symbol'] not in stock_list:
                stock_list.append(stock['symbol'])
        
        for stock in top_losers:
            if stock['symbol'] not in stock_list:
                stock_list.append(stock['symbol'])
        
        recommendations = []
        processed_count = 0
        max_stocks = 20  # Reduced from 100 to 20 for faster processing
        
        for ticker in stock_list:
            if processed_count >= max_stocks:
                break
                
            try:
                # Get price history (this is the most expensive call)
                history_data = scraper.get_price_history(ticker)
                
                if not history_data or len(history_data) < 10:  # Reduced minimum from 20 to 10
                    continue

                # Convert history data to OHLCV format - simplified
                ohlcv_data = []
                for item in history_data[-30:]:  # Only use last 30 data points for faster processing
                    try:
                        ohlcv_data.append({
                            "open": float(item[1]) if item[1] else 0.0,
                            "high": float(item[2]) if item[2] else 0.0,
                            "low": float(item[3]) if item[3] else 0.0,
                            "close": float(item[4]) if item[4] else 0.0,
                            "volume": float(item[5]) if len(item) > 5 and item[5] else 0.0
                        })
                    except (IndexError, ValueError, TypeError):
                        continue

                if len(ohlcv_data) < 10:
                    continue

                # Make prediction
                signal, action, confidence, explanation = predictor.predict_signal(ohlcv_data)
                
                # Calculate additional metrics
                current_price = ohlcv_data[-1]['close']
                prev_price = ohlcv_data[-2]['close'] if len(ohlcv_data) > 1 else current_price
                price_change = ((current_price - prev_price) / prev_price) * 100 if prev_price > 0 else 0
                
                recommendation = {
                    'ticker': ticker,
                    'action': action,
                    'confidence': round(float(confidence) * 100, 2),
                    'latest_price': current_price,
                    'price_change': round(price_change, 2),
                    'signal': int(signal),
                    'explanation': explanation,
                    'volume': ohlcv_data[-1]['volume'],
                    'data_points': len(ohlcv_data)
                }
                
                recommendations.append(recommendation)
                processed_count += 1
                
            except Exception as e:
                # Skip stocks that fail analysis
                print(f"Error processing {ticker}: {str(e)}")
                continue
        
        # Sort recommendations by confidence for better presentation
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        return jsonify(recommendations)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/quick_recommendations')
def quick_recommendations():
    """Quick recommendations based on current market data without historical analysis"""
    try:
        # Get current market data
        top_turnover = scraper.get_top_turnover()[:10]
        top_gainers = scraper.get_top_gainer()[:10]
        top_losers = scraper.get_top_loser()[:10]
        
        recommendations = []
        
        # Process top gainers as potential BUY candidates
        for stock in top_gainers:
            if stock.get('percentageChange', 0) > 5:  # Strong positive momentum
                recommendations.append({
                    'ticker': stock['symbol'],
                    'action': 'BUY',
                    'confidence': min(85, 60 + abs(stock.get('percentageChange', 0))),
                    'latest_price': stock.get('closingPrice', 0),
                    'price_change': stock.get('percentageChange', 0),
                    'signal': 2,
                    'explanation': f"Strong upward momentum with {stock.get('percentageChange', 0):.2f}% gain",
                    'volume': stock.get('totalTradeQuantity', 0),
                    'data_points': 'Live market data'
                })
        
        # Process moderate gainers as potential HOLD candidates
        for stock in top_turnover:
            change = stock.get('percentageChange', 0)
            if -2 <= change <= 5:  # Stable movement
                recommendations.append({
                    'ticker': stock['symbol'],
                    'action': 'HOLD',
                    'confidence': 70,
                    'latest_price': stock.get('closingPrice', 0),
                    'price_change': change,
                    'signal': 1,
                    'explanation': f"Stable price movement with good trading volume",
                    'volume': stock.get('totalTradeQuantity', 0),
                    'data_points': 'Live market data'
                })
        
        # Process top losers as potential SELL candidates (if declining severely)
        for stock in top_losers:
            if stock.get('percentageChange', 0) < -5:  # Strong negative momentum
                recommendations.append({
                    'ticker': stock['symbol'],
                    'action': 'SELL',
                    'confidence': min(80, 50 + abs(stock.get('percentageChange', 0))),
                    'latest_price': stock.get('closingPrice', 0),
                    'price_change': stock.get('percentageChange', 0),
                    'signal': 0,
                    'explanation': f"Strong downward momentum with {stock.get('percentageChange', 0):.2f}% loss",
                    'volume': stock.get('totalTradeQuantity', 0),
                    'data_points': 'Live market data'
                })
        
        # Remove duplicates and sort by confidence
        seen_tickers = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec['ticker'] not in seen_tickers:
                seen_tickers.add(rec['ticker'])
                unique_recommendations.append(rec)
        
        unique_recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        return jsonify(unique_recommendations[:20])  # Return top 20
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
