from flask import Flask, render_template, request, jsonify
import json
import os
from trading_signal_predictor import TradingSignalPredictor

app = Flask(__name__)

# Initialize the predictor globally
predictor = None

def init_predictor():
    """Initialize the trading signal predictor"""
    global predictor
    try:
        predictor = TradingSignalPredictor()
        return True
    except Exception as e:
        print(f"Failed to initialize predictor: {e}")
        return False

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'ohlcv_data' not in data:
            return jsonify({
                'error': 'Invalid input data. Please provide OHLCV data.'
            }), 400
        
        ohlcv_data = data['ohlcv_data']
        
        # Validate input data
        if not ohlcv_data or len(ohlcv_data) == 0:
            return jsonify({
                'error': 'No OHLCV data provided'
            }), 400
        
        # Validate each data point
        required_fields = ['open', 'high', 'low', 'close']
        for i, point in enumerate(ohlcv_data):
            for field in required_fields:
                if field not in point:
                    return jsonify({
                        'error': f'Missing {field} in data point {i+1}'
                    }), 400
                
                try:
                    float(point[field])
                except (ValueError, TypeError):
                    return jsonify({
                        'error': f'Invalid {field} value in data point {i+1}'
                    }), 400
        
        # Make prediction
        if predictor is None:
            return jsonify({
                'error': 'Predictor not initialized'
            }), 500
        
        signal, action, confidence, explanation = predictor.predict_signal(ohlcv_data)
        
        # Return prediction results
        return jsonify({
            'signal': int(signal),
            'action': action,
            'confidence': float(confidence),
            'explanation': explanation,
            'data_points_used': len(ohlcv_data),
            'success': True
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'predictor_loaded': predictor is not None
    })

if __name__ == '__main__':
    print("🚀 Starting Trading Signal Predictor Web App...")
    
    # Initialize predictor
    if init_predictor():
        print("✓ Predictor initialized successfully")
    else:
        print("⚠ Warning: Predictor failed to initialize - some features may not work")
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)