# FNEPSE - Stock Market Analyzer & Predictor

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive AI-powered stock market analysis and trading signal prediction system that combines CNN and LSTM neural networks with technical analysis. The system includes real-time NEPSE (Nepal Stock Exchange) data scraping, pattern recognition, and both command-line and web interfaces.

## 🚀 Features

### Core Functionality
- **AI-Powered Predictions**: CNN-LSTM hybrid model for trading signal generation
- **Pattern Recognition**: Candlestick pattern analysis using deep learning
- **Technical Analysis**: 15+ technical indicators (RSI, MACD, Moving Averages, etc.)
- **Real-time Data**: Live NEPSE market data scraping and analysis
- **Multi-timeframe Analysis**: Support for various timeframes and lookback periods

### Interfaces
- **Web Application**: Modern, responsive web interface with Bootstrap 5
- **CLI Tool**: Command-line interface for batch processing and automation
- **REST API**: RESTful endpoints for integration with other systems

### Data Sources
- **NEPSE Integration**: Real-time data from Nepal Stock Exchange
- **Kaggle Datasets**: Historical US stock and ETF data for training
- **Custom Data Input**: Support for manual OHLCV data entry

## 📊 Model Architecture

The system uses a sophisticated hybrid architecture:

1. **CNN Layer**: Extracts local patterns from candlestick formations
2. **LSTM Layer**: Captures temporal dependencies and market trends
3. **Technical Indicators**: Enhanced feature engineering with 10+ indicators
4. **Ensemble Methods**: Combines multiple prediction strategies

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM (recommended for model training)

### Quick Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sthaarwin/FNEPSE.git
   cd FNEPSE
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and preprocess data** (optional - for training):
   ```bash
   python src/data_loader.py
   python src/preprocess.py
   ```

## 🚀 Usage

### Web Application

Start the web interface:
```bash
python app.py
```
Then open your browser to `http://localhost:5000`

### Enhanced Web App with NEPSE Integration

For the full-featured web app with real-time NEPSE data:
```bash
python web_app.py
```
Access at `http://localhost:5001`

### Command Line Interface

For direct predictions:
```bash
python trading_signal_predictor.py
```

### API Usage

Make predictions via REST API:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ohlcv_data": [
      {"open": 100, "high": 105, "low": 98, "close": 103},
      {"open": 103, "high": 107, "low": 101, "close": 106}
    ]
  }'
```

## 📁 Project Structure

```
FNEPSE/
├── app.py                         # Main Flask web application
├── web_app.py                     # Enhanced web app with NEPSE integration
├── trading_signal_predictor.py    # Core prediction engine
├── config.yaml                    # Configuration settings
├── requirements.txt               # Python dependencies
├── data/                          # Data storage
│   ├── raw/                       # Raw market data
│   ├── processed/                 # Cleaned datasets
│   └── preprocessed/              # Model-ready data
├── models/                        # Trained models
│   ├── candlestick_cnn_lstm.h5    # Pattern recognition model
│   ├── trading_signals_model.h5   # Trading signal model
│   └── scaler.pkl                 # Feature scaler
├── scraper/                       # NEPSE data scraper
│   └── nepse_scraper/             # Web scraping utilities
├── src/                           # Source code
│   ├── data_loader.py             # Data downloading and loading
│   └── preprocess.py              # Data preprocessing pipeline
├── notebooks/                     # Jupyter notebooks
├── templates/                     # HTML templates
├── static/                        # Static web assets
└── utils/                         # Utility functions
```

## 🎯 Trading Signals

The system generates three types of signals:

- **BUY (2)**: Strong bullish indicators suggest potential upward movement
- **HOLD (1)**: Mixed or neutral signals, maintain current position
- **SELL (0)**: Bearish indicators suggest potential downward movement

Each prediction includes:
- Signal strength (0-2)
- Confidence score (0-1)
- Detailed explanation of factors
- Technical indicator analysis

## 📈 Technical Indicators

The system analyzes multiple technical indicators:

- **Trend**: Moving Averages (5, 20), EMA (12, 26)
- **Momentum**: RSI, MACD, Stochastic Oscillator
- **Volatility**: Standard Deviation, Bollinger Bands
- **Volume**: Volume surge analysis
- **Price Action**: Support/Resistance levels, Price spreads

## 🔧 Configuration

Customize the system behavior via `config.yaml`:

```yaml
model:
  lookback_period: 20
  confidence_threshold: 0.7
  
trading:
  risk_tolerance: "medium"
  max_positions: 10
  
data:
  update_frequency: "daily"
  data_sources: ["nepse", "kaggle"]
```

## 🚦 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface homepage |
| `/predict` | POST | Generate trading signal |
| `/market_data` | GET | Current NEPSE market data |
| `/predict_for_ticker` | POST | Predict for specific stock |
| `/recommendations` | GET | AI-generated stock recommendations |
| `/health` | GET | System health check |

## 🔒 Risk Warning

⚠️ **Important Disclaimer**: This tool is for educational and research purposes only. 

- Past performance does not guarantee future results
- Always conduct your own research before making investment decisions
- Consider consulting with financial advisors for investment advice
- The developers are not responsible for any financial losses

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Nepal Stock Exchange (NEPSE) for market data
- Kaggle community for stock datasets
- TensorFlow and Keras teams for ML frameworks
- Flask community for web framework

## 📧 Contact

- **Author**: Arwin Shrestha
- **GitHub**: [@sthaarwin](https://github.com/sthaarwin)
- **Project Link**: [https://github.com/sthaarwin/FNEPSE](https://github.com/sthaarwin/FNEPSE)

---

**Made with ❤️ for the trading community**
