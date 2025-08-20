# FNEPSE

A modern web interface for the AI-powered trading signal predictor. This web application provides an intuitive way to input OHLCV (Open, High, Low, Close, Volume) data and get trading signals.

## Features

- **Modern UI**: Clean, responsive design with Bootstrap 5
- **Real-time Predictions**: Get instant trading signals (BUY/SELL/HOLD)
- **Data Validation**: Comprehensive input validation and error handling
- **Sample Data**: Pre-loaded sample data for testing
- **Confidence Scores**: View prediction confidence levels
- **Mobile Friendly**: Responsive design works on all devices

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install flask
   ```

2. **Run the Web App**:
   ```bash
   python app.py
   ```

3. **Open Browser**:
   Navigate to `http://localhost:5000`

## Usage

1. **Add Data Points**: Click "Add Data Point" and enter OHLCV values
2. **Load Sample Data**: Use "Load Sample Data" for quick testing
3. **Get Prediction**: Click "Predict Signal" to analyze the data
4. **View Results**: See the trading signal with confidence score and explanation

## API Endpoints

- `GET /`: Main web interface
- `POST /predict`: Prediction API endpoint
- `GET /health`: Health check endpoint

## Input Format

Each data point requires:
- **Open**: Opening price
- **High**: Highest price
- **Low**: Lowest price  
- **Close**: Closing price
- **Volume**: Trading volume (optional)

## Output Format

The prediction returns:
- **Signal**: 0 (Sell), 1 (Hold), 2 (Buy)
- **Action**: Human-readable action
- **Confidence**: Prediction confidence (0-1)
- **Explanation**: Detailed analysis explanation

## Risk Warning

⚠️ **This tool is for educational purposes only. Always do your own research before making trading decisions. Past performance does not guarantee future results.**