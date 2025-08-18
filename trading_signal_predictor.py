import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class TradingSignalPredictor:
    def __init__(self, model_path="models/candlestick_cnn_lstm.h5", 
                 trading_model_path="models/trading_signals_model.h5",
                 scaler_path="models/scaler.pkl"):
        """
        Initialize the trading signal predictor
        
        Args:
            model_path: Path to pattern recognition model
            trading_model_path: Path to trading signal model
            scaler_path: Path to saved scaler
        """
        self.pattern_model = None
        self.trading_model = None
        self.scaler = MinMaxScaler()
        self.lookback = 20
        self.price_history = []
        
        # Try to load models
        try:
            print("Loading pattern recognition model...")
            self.pattern_model = load_model(model_path)
            print("✓ Pattern model loaded successfully")
        except:
            print("⚠ Pattern model not found - will use basic features only")
            
        try:
            print("Loading trading signal model...")
            self.trading_model = load_model(trading_model_path)
            print("✓ Trading model loaded successfully")
        except:
            print("⚠ Trading model not found - will use rule-based signals")
            
        try:
            self.scaler = joblib.load(scaler_path)
            print("✓ Scaler loaded successfully")
        except:
            print("⚠ Scaler not found - will fit on new data")
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators from OHLCV data"""
        df = df.copy()
        
        # Basic indicators
        df['ma_5'] = df['close'].rolling(5, min_periods=1).mean()
        df['ma_20'] = df['close'].rolling(20, min_periods=1).mean()
        df['price_change'] = df['close'].pct_change().fillna(0)
        df['volatility'] = df['close'].rolling(10, min_periods=1).std().fillna(0)
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        df['oc_spread'] = (df['close'] - df['open']) / df['open']
        
        # EMAs
        df['ema_12'] = df['close'].ewm(span=12, min_periods=1).mean()
        df['ema_26'] = df['close'].ewm(span=26, min_periods=1).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, min_periods=1).mean()
        
        # Stochastic
        low_min = df['low'].rolling(14, min_periods=1).min()
        high_max = df['high'].rolling(14, min_periods=1).max()
        df['stoch_k'] = ((df['close'] - low_min) / (high_max - low_min + 1e-8)) * 100
        df['stoch_d'] = df['stoch_k'].rolling(3, min_periods=1).mean()
        
        return df.fillna(0)
    
    def rule_based_signal(self, latest_data):
        """Generate trading signal using rule-based approach"""
        if len(latest_data) < 5:
            return 1, "HOLD", "Insufficient data for analysis"
        
        latest = latest_data.iloc[-1]
        prev = latest_data.iloc[-2] if len(latest_data) > 1 else latest
        
        # Price momentum
        price_momentum = (latest['close'] - prev['close']) / prev['close']
        
        # Moving average signals
        ma_signal = 1 if latest['close'] > latest['ma_20'] else 0
        
        # RSI signals
        rsi_oversold = latest['rsi'] < 30
        rsi_overbought = latest['rsi'] > 70
        
        # MACD signals
        macd_bullish = latest['macd'] > latest['macd_signal']
        
        # Volume analysis (if available)
        volume_surge = False
        if 'volume' in latest_data.columns and len(latest_data) > 5:
            avg_volume = latest_data['volume'].rolling(5).mean().iloc[-2]
            volume_surge = latest['volume'] > 1.5 * avg_volume
        
        # Decision logic
        buy_signals = sum([
            price_momentum > 0.02,  # 2% price increase
            ma_signal,
            rsi_oversold,
            macd_bullish,
            volume_surge
        ])
        
        sell_signals = sum([
            price_momentum < -0.02,  # 2% price decrease
            not ma_signal,
            rsi_overbought,
            not macd_bullish
        ])
        
        if buy_signals >= 3:
            return 2, "BUY", f"Strong buy signals: {buy_signals}/5"
        elif sell_signals >= 3:
            return 0, "SELL", f"Strong sell signals: {sell_signals}/5"
        else:
            return 1, "HOLD", f"Mixed signals - Buy: {buy_signals}, Sell: {sell_signals}"
    
    def predict_signal(self, ohlcv_data):
        """
        Predict trading signal from OHLCV data
        
        Args:
            ohlcv_data: List of dictionaries with keys: open, high, low, close, volume (optional)
        
        Returns:
            signal (int): 0=Sell, 1=Hold, 2=Buy
            action (str): Human readable action
            confidence (float): Confidence score
            explanation (str): Explanation of the decision
        """
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv_data)
        
        # Add technical indicators
        df = self.calculate_technical_indicators(df)
        
        # Rule-based fallback if models not available
        if self.trading_model is None:
            signal, action, explanation = self.rule_based_signal(df)
            return signal, action, 0.7, explanation
        
        # Prepare features for model
        feature_cols = [
            "open", "high", "low", "close",
            "ma_5", "ma_20", "ema_12", "ema_26", 
            "price_change", "volatility", "hl_spread", "oc_spread",
            "rsi", "macd", "macd_signal", "stoch_k", "stoch_d"
        ]
        
        # Check if we have enough data
        if len(df) < self.lookback:
            signal, action, explanation = self.rule_based_signal(df)
            return signal, action, 0.6, f"Limited data - {explanation}"
        
        try:
            # Scale features
            features = df[feature_cols].values
            if hasattr(self.scaler, 'scale_'):
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = self.scaler.fit_transform(features)
            
            # Create sequence
            sequence = features_scaled[-self.lookback:].reshape(1, self.lookback, -1)
            
            # Get pattern probabilities if available
            if self.pattern_model is not None:
                pattern_probs = self.pattern_model.predict(sequence, verbose=0)
                
                # Combine with technical features
                n_features = sequence.shape[2]
                n_patterns = pattern_probs.shape[1]
                combined_sequence = np.zeros((1, self.lookback, n_features + n_patterns))
                combined_sequence[0, :, :n_features] = sequence[0]
                
                for i in range(self.lookback):
                    combined_sequence[0, i, n_features:] = pattern_probs[0]
                
                sequence = combined_sequence
            
            # Predict trading signal
            prediction = self.trading_model.predict(sequence, verbose=0)
            signal = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            actions = ["SELL", "HOLD", "BUY"]
            action = actions[signal]
            
            explanation = f"Model prediction with {confidence:.1%} confidence"
            
            return signal, action, confidence, explanation
            
        except Exception as e:
            print(f"Model prediction failed: {e}")
            signal, action, explanation = self.rule_based_signal(df)
            return signal, action, 0.5, f"Fallback rule-based: {explanation}"

def get_user_input():
    """Get OHLCV data from user input"""
    print("\n" + "="*50)
    print("📈 TRADING SIGNAL PREDICTOR")
    print("="*50)
    
    data_points = []
    
    print("\nEnter OHLCV data (minimum 1 point, recommended 20+ for better accuracy)")
    print("Enter 'done' when finished, or 'q' to quit")
    
    while True:
        try:
            print(f"\n--- Data Point {len(data_points) + 1} ---")
            
            if len(data_points) == 0:
                user_input = input("Continue? (Enter/y to continue, q to quit): ").strip().lower()
                if user_input in ['q', 'quit', 'exit']:
                    return None
            
            open_price = input("Open price: ").strip()
            if open_price.lower() in ['done', 'q', 'quit']:
                break
                
            high_price = input("High price: ").strip()
            if high_price.lower() in ['done', 'q', 'quit']:
                break
                
            low_price = input("Low price: ").strip()
            if low_price.lower() in ['done', 'q', 'quit']:
                break
                
            close_price = input("Close price: ").strip()
            if close_price.lower() in ['done', 'q', 'quit']:
                break
            
            volume = input("Volume (optional, press Enter to skip): ").strip()
            
            # Convert to float
            data_point = {
                'open': float(open_price),
                'high': float(high_price),
                'low': float(low_price),
                'close': float(close_price)
            }
            
            if volume and volume.lower() not in ['done', 'q', 'quit']:
                data_point['volume'] = float(volume)
            
            data_points.append(data_point)
            print(f"✓ Added data point {len(data_points)}")
            
        except ValueError:
            print("❌ Invalid input. Please enter numeric values.")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            return None
    
    return data_points if data_points else None

def main():
    """Main function to run the trading signal predictor"""
    print("🚀 Initializing Trading Signal Predictor...")
    predictor = TradingSignalPredictor()
    
    while True:
        try:
            # Get user input
            ohlcv_data = get_user_input()
            
            if ohlcv_data is None:
                print("\n👋 Goodbye!")
                break
            
            # Make prediction
            signal, action, confidence, explanation = predictor.predict_signal(ohlcv_data)
            
            # Display results
            print("\n" + "="*50)
            print("📊 TRADING SIGNAL RESULT")
            print("="*50)
            
            # Color coding for terminal
            colors = {
                "SELL": "\033[91m",    # Red
                "HOLD": "\033[93m",    # Yellow
                "BUY": "\033[92m",     # Green
                "END": "\033[0m"       # Reset
            }
            
            color = colors.get(action, "")
            end_color = colors["END"]
            
            print(f"🎯 Signal: {color}{action}{end_color}")
            print(f"📈 Action: {signal} (0=Sell, 1=Hold, 2=Buy)")
            print(f"🎯 Confidence: {confidence:.1%}")
            print(f"💡 Explanation: {explanation}")
            print(f"📊 Data points used: {len(ohlcv_data)}")
            
            # Risk warning
            print(f"\n⚠️  RISK WARNING:")
            print(f"   This is for educational purposes only.")
            print(f"   Always do your own research before trading.")
            print(f"   Past performance doesn't guarantee future results.")
            
            # Ask for another prediction
            print(f"\n" + "-"*50)
            continue_input = input("Make another prediction? (y/n): ").strip().lower()
            if continue_input in ['n', 'no', 'q', 'quit']:
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()