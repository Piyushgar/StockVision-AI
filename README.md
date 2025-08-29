# StockVision-AI

A comprehensive stock market analysis and prediction platform powered by multiple machine learning models and interactive visualizations.

## Features

- **Multi-Model Stock Prediction**: Implements various advanced models including:
  - LSTM (Long Short-Term Memory)
  - BiLSTM (Bidirectional LSTM)
  - GRU (Gated Recurrent Unit)
  - Transformer
  - Informer
  - ARIMA
  - TFT (Temporal Fusion Transformer)

- **Interactive Web Interface**: Built with Streamlit featuring:
  - Real-time stock data visualization
  - Custom date range selection
  - Multiple technical indicators
  - Model comparison dashboard
  - Dark/Light mode support

- **Advanced Data Analysis**:
  - Technical indicator analysis
  - Historical price trends
  - Volume analysis
  - Performance metrics visualization

## Getting Started

### Prerequisites

- Python 3.9+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stockvision-AI.git
cd stockvision-AI
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Start the application:
```bash
streamlit run main.py
```

## Project Structure

- `main.py` - Main application file with Streamlit interface
- `train_all_model.py` - Training scripts for all ML models
- `models/` - Directory containing model implementations:
  - `arima_model.py` - ARIMA model implementation
  - `informer_model.py` - Informer model implementation
  - `lstm_model.py` - LSTM model implementation
  - `tft_model.py` - Temporal Fusion Transformer implementation
- `stocks/` - Directory containing stock data CSV files
- `utils/` - Utility functions and helper modules
- `images/` - Static images and resources

## Components

### 1. Data Processing
- Historical stock data loading
- Data preprocessing and normalization
- Feature engineering
- Technical indicator calculation

### 2. Model Training
- Multiple model architectures
- Hyperparameter optimization
- Cross-validation
- Model persistence

### 3. Prediction Interface
- Real-time predictions
- Multiple timeframe forecasting
- Confidence intervals
- Model performance metrics

### 4. Visualization
- Interactive price charts
- Technical indicator plots
- Prediction visualization
- Performance comparison graphs

## Models Available

1. **LSTM Model**: For capturing long-term dependencies in stock price movements
2. **BiLSTM Model**: Enhanced LSTM with bidirectional processing
3. **GRU Model**: Efficient alternative to LSTM
4. **Transformer Model**: Attention-based architecture for time series
5. **Informer Model**: Enhanced transformer for long sequence time-series forecasting
6. **ARIMA Model**: Statistical approach for time series forecasting
7. **TFT Model**: Advanced model combining recent developments in time series forecasting

## Usage

1. Launch the application
2. Select a stock symbol from the available options
3. Choose the desired date range
4. Select prediction models to compare
5. View predictions and analysis
6. Export results if needed

## License

This project is licensed under the MIT License - see the LICENSE file for details.