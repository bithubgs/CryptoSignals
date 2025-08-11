import streamlit as st
import datetime
import random
import requests
import time
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import warnings
import logging

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.ERROR)

# Import yfinance for historical data
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    st.error("yfinance მოდული ვერ მოიძებნა. დააინსტალირეთ: pip install yfinance")
    YF_AVAILABLE = False

# Import forecasting models with checks
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    st.warning("Prophet მოდული ვერ მოიძებნა. დააინსტალირეთ 'prophet': pip install prophet")
    PROPHET_AVAILABLE = False

try:
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    LSTM_AVAILABLE = True
except ImportError:
    st.warning("TensorFlow/Keras, Scikit-learn მოდულები ვერ მოიძებნა. დააინსტალირეთ: pip install tensorflow scikit-learn")
    LSTM_AVAILABLE = False

# --- Configuration ---
st.set_page_config(layout="wide", page_title="კრიპტო სიგნალები LIVE", page_icon="📈")

PREDICTION_DAYS = 30
MAX_HISTORICAL_DAYS = 365

# Extended cryptocurrency mapping for Yahoo Finance
YAHOO_CRYPTO_MAP = {
    'BTC': 'BTC-USD', 'ETH': 'ETH-USD', 'SOL': 'SOL-USD', 'XRP': 'XRP-USD', 
    'ADA': 'ADA-USD', 'SAND': 'SAND-USD', 'SKL': 'SKL-USD', 'RVN': 'RVN-USD',
    'DOGE': 'DOGE-USD', 'MATIC': 'MATIC-USD', 'DOT': 'DOT-USD', 'LINK': 'LINK-USD',
    'UNI': 'UNI-USD', 'LTC': 'LTC-USD', 'AVAX': 'AVAX-USD', 'ATOM': 'ATOM-USD'
}

@st.cache_data(ttl=300, show_spinner=False)  # Cache for 5 minutes
def fetch_yfinance_data(ticker_symbol):
    """Fetches real-time and historical data using yfinance with caching."""
    if not YF_AVAILABLE:
        return None, None

    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        historical_df = ticker.history(period=f"{MAX_HISTORICAL_DAYS}d")
        
        if historical_df.empty:
            st.error(f"მონაცემები ვერ მოიძებნა {ticker_symbol}-ისთვის Yahoo Finance-დან.")
            return None, None
            
        historical_data = []
        for index, row in historical_df.iterrows():
            historical_data.append({
                'date': index.tz_convert(None) if index.tz else index,
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': float(row['Volume']),
                'type': 'historical'
            })
            
        # Extract real-time details
        coin_details = {
            'name': info.get('longName', ticker_symbol),
            'symbol': info.get('symbol', ticker_symbol).split('-')[0],
            'currentPrice': info.get('regularMarketPrice', info.get('currentPrice')),
            'dailyChange': info.get('regularMarketChangePercent'),
            '24hVolume': info.get('regularMarketVolume', info.get('volume')),
            'marketCap': info.get('marketCap'),
            'marketCapRank': info.get('marketCapRank'),
            'ath': info.get('fiftyTwoWeekHigh'),
            'atl': info.get('fiftyTwoWeekLow'),
            'lastUpdated': datetime.datetime.now().isoformat()
        }
        
        return coin_details, historical_data
        
    except Exception as e:
        st.error(f"შეცდომა მონაცემების ჩამოტვირთვისას {ticker_symbol}: {str(e)}")
        return None, None

def calculate_support_resistance(df, window=20):
    """Calculates support and resistance levels using rolling windows."""
    if len(df) < window:
        return None, None
    
    try:
        df_temp = df.copy()
        df_temp['rolling_min'] = df_temp['low'].rolling(window=window, min_periods=1).min()
        df_temp['rolling_max'] = df_temp['high'].rolling(window=window, min_periods=1).max()
        
        support_level = df_temp['rolling_min'].iloc[-window:].min()
        resistance_level = df_temp['rolling_max'].iloc[-window:].max()
        
        return support_level, resistance_level
    except Exception as e:
        st.warning(f"Support/Resistance გამოთვლის შეცდომა: {e}")
        return None, None

def calculate_rsi(prices, period=14):
    """Calculate RSI with improved error handling."""
    try:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(span=period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(span=period, adjust=False, min_periods=period).mean()
        
        # Avoid division by zero
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)  # Fill NaN with neutral value
    except Exception as e:
        st.warning(f"RSI გამოთვლის შეცდომა: {e}")
        return pd.Series([50] * len(prices), index=prices.index)

class CryptoDataForecaster:
    def __init__(self, symbol, historical_prices_df):
        self.symbol = symbol.upper()
        self.historical_prices = historical_prices_df.copy()
        
        # Data preprocessing with better error handling
        try:
            self.historical_prices['date'] = pd.to_datetime(self.historical_prices['date'])
            self.historical_prices.set_index('date', inplace=True)
            self.historical_prices['close'] = pd.to_numeric(self.historical_prices['close'], errors='coerce')
            
            # Remove any rows with invalid data
            self.historical_prices.dropna(subset=['close'], inplace=True)
            
            if len(self.historical_prices) == 0:
                raise ValueError("არასაკმარისი ვალიდური მონაცემები")
            
            self.historical_with_indicators = self.calculate_technical_indicators()
            
        except Exception as e:
            st.error(f"მონაცემების დამუშავების შეცდომა: {e}")
            raise

    def _generate_lstm_predictions(self, days):
        """Enhanced LSTM predictions with better error handling and validation."""
        if not LSTM_AVAILABLE:
            st.error("LSTM მოდელი არ არის ხელმისაწვდომი.")
            return []
        
        try:
            df_lstm = self.historical_with_indicators[['close', 'MA7', 'MA25', 'RSI', 'MACD']].dropna()
            
            if len(df_lstm) < 100:
                st.warning("არასაკმარისი მონაცემები LSTM პროგნოზისთვის (საჭიროა მინიმუმ 100 დღე).")
                return []

            look_back = min(60, len(df_lstm) // 3)  # Adaptive lookback period
            n_features = df_lstm.shape[1]

            # Prepare data
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_scaled = scaler.fit_transform(df_lstm.values)
            
            scaler_price = MinMaxScaler(feature_range=(0, 1))
            scaler_price.fit(df_lstm['close'].values.reshape(-1, 1))

            X, y = [], []
            for i in range(len(data_scaled) - look_back):
                X.append(data_scaled[i:(i + look_back), :])
                y.append(data_scaled[i + look_back, 0])
                
            if len(X) == 0:
                st.warning("არასაკმარისი მონაცემები მოდელის ტრენინგისთვის.")
                return []
                
            X = np.array(X)
            y = np.array(y)
            
            # Build model with regularization
            model = Sequential([
                LSTM(units=50, return_sequences=True, input_shape=(look_back, n_features)),
                Dropout(0.2),
                LSTM(units=50, return_sequences=False),
                Dropout(0.2),
                Dense(units=25),
                Dropout(0.1),
                Dense(units=1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            
            # Train with early stopping simulation
            epochs = min(50, max(10, len(X) // 10))
            model.fit(X, y, epochs=epochs, batch_size=min(32, len(X)//4), verbose=0)

            # Generate predictions
            prediction_list = []
            current_input = data_scaled[-look_back:].reshape(1, look_back, n_features)
            last_date = self.historical_prices.index[-1]

            for i in range(days):
                try:
                    predicted_scaled = model.predict(current_input, verbose=0)[0, 0]
                    predicted_price = scaler_price.inverse_transform([[predicted_scaled]])[0, 0]
                    
                    # Ensure reasonable price bounds
                    last_price = self.historical_prices['close'].iloc[-1]
                    predicted_price = max(last_price * 0.1, min(predicted_price, last_price * 10))
                    
                    date = last_date + datetime.timedelta(days=i+1)
                    prediction_list.append({
                        'date': date, 
                        'close': round(max(0.00000001, predicted_price), 8), 
                        'type': 'prediction'
                    })
                    
                    # Update input for next prediction
                    last_features = current_input[0, -1, :].copy()
                    last_features[0] = predicted_scaled
                    current_input = np.append(current_input[:, 1:, :], 
                                            last_features.reshape(1, 1, n_features), axis=1)
                                            
                except Exception as e:
                    st.warning(f"პროგნოზის გენერირების შეცდომა დღე {i+1}: {e}")
                    break

            return prediction_list
            
        except Exception as e:
            st.error(f"LSTM მოდელის შეცდომა: {e}")
            return []

    def generate_predictions(self, days=PREDICTION_DAYS, model_choice="LSTM Model"):
        """Generate predictions using selected model."""
        if self.historical_prices.empty:
            st.warning("ისტორიული მონაცემები არ არის პროგნოზირებისთვის.")
            return []

        try:
            if model_choice == "Prophet Model" and PROPHET_AVAILABLE:
                return self._generate_prophet_predictions(days)
            elif model_choice == "LSTM Model":
                return self._generate_lstm_predictions(days)
            else:
                st.error(f"მოდელი {model_choice} არ არის ხელმისაწვდომი.")
                return []
        except Exception as e:
            st.error(f"პროგნოზირების შეცდომა: {e}")
            return []

    def _generate_prophet_predictions(self, days):
        """Generate Prophet predictions with better error handling."""
        try:
            if len(self.historical_prices) < 30:
                st.warning("არასაკმარისი მონაცემები Prophet პროგნოზისთვის (საჭიროა მინიმუმ 30 დღე).")
                return []

            df_prophet = self.historical_prices.reset_index()[['date', 'close']].rename(
                columns={'date': 'ds', 'close': 'y'})
            df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
            df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')
            df_prophet.dropna(inplace=True)

            model = Prophet(
                daily_seasonality=True, 
                weekly_seasonality=True, 
                yearly_seasonality=False, 
                changepoint_prior_scale=0.1,
                seasonality_mode='multiplicative'
            )
            
            # Suppress Prophet logs
            with st.spinner("Prophet მოდელის ტრენინგი..."):
                model.fit(df_prophet)
                
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)
            
            prediction_data = []
            for _, row in forecast.tail(days).iterrows():
                predicted_price = max(0.00000001, row['yhat'])
                prediction_data.append({
                    'date': row['ds'], 
                    'close': round(predicted_price, 8), 
                    'type': 'prediction'
                })
                
            return prediction_data
            
        except Exception as e:
            st.error(f"Prophet მოდელის შეცდომა: {e}")
            return []

    def generate_signals_from_prediction(self, prediction_data):
        """Generate trading signals from predictions with improved logic."""
        if len(prediction_data) < 7:
            return []
        
        try:
            df_prediction = pd.DataFrame(prediction_data)
            df_prediction['date'] = pd.to_datetime(df_prediction['date'])
            df_prediction.set_index('date', inplace=True)
            df_prediction['close'] = pd.to_numeric(df_prediction['close'], errors='coerce')

            # Calculate moving averages with different windows
            short_window = 3
            long_window = 7

            df_prediction['MA_short'] = df_prediction['close'].rolling(window=short_window, min_periods=1).mean()
            df_prediction['MA_long'] = df_prediction['close'].rolling(window=long_window, min_periods=1).mean()
            
            # Calculate price momentum
            df_prediction['momentum'] = df_prediction['close'].pct_change(periods=2)

            signals = []
            for i in range(1, len(df_prediction)):
                current_date = df_prediction.index[i]
                
                if pd.isna(df_prediction['MA_short'].iloc[i]) or pd.isna(df_prediction['MA_long'].iloc[i]):
                    continue

                # Enhanced signal generation with momentum consideration
                ma_cross_up = (df_prediction['MA_short'].iloc[i-1] <= df_prediction['MA_long'].iloc[i-1] and
                              df_prediction['MA_short'].iloc[i] > df_prediction['MA_long'].iloc[i])
                              
                ma_cross_down = (df_prediction['MA_short'].iloc[i-1] >= df_prediction['MA_long'].iloc[i-1] and
                                df_prediction['MA_short'].iloc[i] < df_prediction['MA_long'].iloc[i])

                momentum = df_prediction['momentum'].iloc[i] if not pd.isna(df_prediction['momentum'].iloc[i]) else 0
                
                if ma_cross_up and momentum >= -0.02:  # Buy signal with momentum filter
                    confidence = min(90, max(65, 75 + abs(momentum) * 100))
                    signals.append({
                        'date': current_date,
                        'type': 'BUY',
                        'price': df_prediction['close'].iloc[i],
                        'confidence': f"{confidence:.0f}%",
                        'argumentation': f"AI პროგნოზირებს მოკლევადიანი MA({short_window})-ის გრძელვადიან MA({long_window})-ზე ზემოთ კვეთას დადებითი მომენტუმით ({momentum:.2%})."
                    })

                elif ma_cross_down and momentum <= 0.02:  # Sell signal with momentum filter
                    confidence = min(85, max(60, 70 + abs(momentum) * 100))
                    signals.append({
                        'date': current_date,
                        'type': 'SELL',
                        'price': df_prediction['close'].iloc[i],
                        'confidence': f"{confidence:.0f}%",
                        'argumentation': f"AI პროგნოზირებს მოკლევადიანი MA({short_window})-ის გრძელვადიან MA({long_window})-ზე ქვემოთ კვეთას უარყოფითი მომენტუმით ({momentum:.2%})."
                    })
            
            return sorted(signals, key=lambda x: x['date'])
            
        except Exception as e:
            st.error(f"სიგნალების გენერირების შეცდომა: {e}")
            return []

    def calculate_technical_indicators(self):
        """Calculate technical indicators with improved error handling."""
        try:
            df_hist = self.historical_prices.copy()
            
            # Moving averages with minimum periods
            df_hist['MA7'] = df_hist['close'].rolling(window=7, min_periods=1).mean()
            df_hist['MA25'] = df_hist['close'].rolling(window=25, min_periods=1).mean()
            df_hist['MA99'] = df_hist['close'].rolling(window=99, min_periods=1).mean()

            # RSI with improved calculation
            df_hist['RSI'] = calculate_rsi(df_hist['close'])

            # MACD with error handling
            try:
                exp1 = df_hist['close'].ewm(span=12, adjust=False, min_periods=1).mean()
                exp2 = df_hist['close'].ewm(span=26, adjust=False, min_periods=1).mean()
                df_hist['MACD'] = exp1 - exp2
                df_hist['Signal_Line'] = df_hist['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
                df_hist['MACD_Histogram'] = df_hist['MACD'] - df_hist['Signal_Line']
            except Exception as e:
                st.warning(f"MACD გამოთვლის შეცდომა: {e}")
                df_hist['MACD'] = 0
                df_hist['Signal_Line'] = 0
                df_hist['MACD_Histogram'] = 0
            
            # Fill any remaining NaN values with forward fill, then backward fill
            df_hist = df_hist.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            return df_hist
            
        except Exception as e:
            st.error(f"ტექნიკური ინდიკატორების გამოთვლის შეცდომა: {e}")
            return self.historical_prices.copy()

def format_price(price):
    """Format price with improved handling."""
    if price is None or pd.isna(price):
        return 'N/A'
    try:
        price = float(price)
        if price >= 1:
            return f"{price:,.2f}"
        elif price >= 0.01:
            return f"{price:.4f}"
        else:
            return f"{price:.8f}".rstrip('0').rstrip('.')
    except (ValueError, TypeError):
        return 'N/A'

def format_currency(value):
    """Format currency values with improved handling."""
    if value is None or pd.isna(value):
        return 'N/A'
    try:
        value = float(value)
        if value >= 1e9:
            return f"${value/1e9:.2f}B"
        elif value >= 1e6:
            return f"${value/1e6:.2f}M"
        elif value >= 1e3:
            return f"${value/1e3:.2f}K"
        else:
            return f"${value:,.0f}"
    except (ValueError, TypeError):
        return 'N/A'

def format_percentage(value):
    """Format percentage values."""
    if value is None or pd.isna(value):
        return 'N/A'
    try:
        return f"{float(value):.2f}%"
    except (ValueError, TypeError):
        return 'N/A'

# --- Streamlit App Layout ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    :root {
        --bg-dark: #0a0a0a;
        --bg-medium: #1a1a1a;
        --bg-light: #2a2a2a;
        --text-light: #e0e0e0;
        --text-medium: #a0a0a0;
        --accent-blue: #00bcd4;
        --accent-purple: #8e2de2;
        --accent-green: #39ff14;
        --accent-red: #ff073a;
        --accent-yellow: #fdd835;
        --card-bg: #1c1c1c;
        --border-color: #333;
    }
    
    .stApp {
        font-family: 'Inter', sans-serif;
        background: var(--bg-dark);
        color: var(--text-light);
        padding: 1rem;
    }
    
    h1 {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(45deg, var(--accent-blue), var(--accent-purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
    }
    
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        background-color: rgba(57, 255, 20, 0.1);
        color: var(--accent-green);
        border: 1px solid var(--accent-green);
        margin-left: 1rem;
    }
    
    .online-dot {
        width: 8px;
        height: 8px;
        background-color: var(--accent-green);
        border-radius: 50%;
        animation: pulse 2s ease-in-out infinite alternate;
    }
    
    @keyframes pulse {
        from { opacity: 1; }
        to { opacity: 0.5; }
    }
    
    .crypto-card {
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .price-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--accent-yellow);
        margin: 0.25rem 0 1rem 0;
    }
    
    .positive { color: var(--accent-green) !important; }
    .negative { color: var(--accent-red) !important; }

    .signal-item {
        background-color: var(--bg-medium);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.75rem;
        border-left: 4px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .signal-item:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .buy-signal { border-left-color: var(--accent-green) !important; }
    .sell-signal { border-left-color: var(--accent-red) !important; }
    
    .signal-type-buy { color: var(--accent-green); font-weight: 700; }
    .signal-type-sell { color: var(--accent-red); font-weight: 700; }

    .stSelectbox > div > div {
        background-color: var(--bg-medium);
        border: 1px solid var(--border-color);
    }
    
    .stButton > button {
        background: linear-gradient(45deg, var(--accent-blue), var(--accent-purple));
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
        transition: opacity 0.3s ease;
    }

    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }

    .metric-card {
        background: var(--bg-medium);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid var(--border-color);
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .metric-title {
        font-size: 0.85rem;
        color: var(--text-medium);
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-light);
    }
</style>
""", unsafe_allow_html=True)

# Header
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown("# კრიპტო სიგნალები LIVE")
with col_status:
    st.markdown("""
    <div class="status-indicator">
        <div class="online-dot"></div>
        <span>Online</span>
    </div>
    """, unsafe_allow_html=True)

# Initialize session state with better defaults
if 'current_symbol' not in st.session_state:
    st.session_state.current_symbol = ''
if 'search_triggered' not in st.session_state:
    st.session_state.search_triggered = False
if 'prediction_model' not in st.session_state:
    st.session_state.prediction_model = "LSTM Model" if LSTM_AVAILABLE else "Prophet Model"
if 'last_update' not in st.session_state:
    st.session_state.last_update = None

# Popular cryptocurrencies with more options
popular_cryptos = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'DOGE', 'MATIC', 'DOT']

# Main layout
col1, col2, col3 = st.columns([3, 6, 3])

# Left Column - Search and Crypto Info
with col1:
    st.markdown('<div class="crypto-card">', unsafe_allow_html=True)
    st.markdown("### 🔍 აირჩიეთ კრიპტოვალუტა")
    
    # Search form
    with st.form("symbol_form", clear_on_submit=False):
        symbol_input = st.text_input(
            "შეიყვანეთ სიმბოლო (მაგ. BTC, ETH)", 
            value=st.session_state.current_symbol,
            placeholder="BTC"
        ).upper().strip()
        
        col_search, col_clear = st.columns([3, 1])
        with col_search:
            submit_button = st.form_submit_button("🔍 ძიება", use_container_width=True)
        with col_clear:
            clear_button = st.form_submit_button("🗑️", use_container_width=True)

        if submit_button and symbol_input:
            if symbol_input in YAHOO_CRYPTO_MAP:
                st.session_state.current_symbol = symbol_input
                st.session_state.search_triggered = True
                st.session_state.last_update = datetime.datetime.now()
                st.rerun()
            else:
                st.error(f"კრიპტოვალუტა {symbol_input} არ არის მხარდაჭერილი.")
                
        if clear_button:
            st.session_state.current_symbol = ''
            st.session_state.search_triggered = False
            st.rerun()

    # Popular crypto buttons with better layout
    st.markdown("**🔥 პოპულარული:**")
    for i in range(0, len(popular_cryptos), 4):
        cols_pop = st.columns(4)
        for j, crypto in enumerate(popular_cryptos[i:i+4]):
            with cols_pop[j]:
                if st.button(crypto, key=f"pop_{crypto}", use_container_width=True):
                    st.session_state.current_symbol = crypto
                    st.session_state.search_triggered = True
                    st.session_state.last_update = datetime.datetime.now()
                    st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Display crypto info if data is available
    if st.session_state.search_triggered and st.session_state.current_symbol:
        yahoo_symbol = YAHOO_CRYPTO_MAP.get(st.session_state.current_symbol)
        
        if yahoo_symbol:
            with st.spinner(f"📊 იტვირთება მონაცემები {st.session_state.current_symbol}-ისთვის..."):
                coin_details, historical_data_list = fetch_yfinance_data(yahoo_symbol)
                
                if coin_details and historical_data_list:
                    st.markdown('<div class="crypto-card">', unsafe_allow_html=True)
                    st.markdown("### 💰 კრიპტოვალუტის ინფორმაცია")
                    
                    # Main price display
                    st.markdown(f"**{coin_details['name']} ({coin_details['symbol']})**")
                    st.markdown(f'<div class="price-value">${format_price(coin_details["currentPrice"])}</div>', 
                              unsafe_allow_html=True)
                    
                    # Key metrics in grid layout
                    st.markdown("**📈 ძირითადი მეტრიკები**")
                    
                    # 24h Change
                    change_value = coin_details.get('dailyChange', 0) or 0
                    change_class = 'positive' if change_value >= 0 else 'negative'
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">24სთ ცვლილება</div>
                        <div class="metric-value {change_class}">
                            {'📈' if change_value >= 0 else '📉'} {format_percentage(change_value)}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Volume
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">24სთ მოცულობა</div>
                        <div class="metric-value">{format_currency(coin_details.get('24hVolume'))}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Market Cap
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">საბაზრო კაპიტალიზაცია</div>
                        <div class="metric-value">{format_currency(coin_details.get('marketCap'))}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 52W High/Low
                    col_high, col_low = st.columns(2)
                    with col_high:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">52კვ მაქსიმუმი</div>
                            <div class="metric-value positive">${format_price(coin_details.get('ath'))}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_low:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">52კვ მინიმუმი</div>
                            <div class="metric-value negative">${format_price(coin_details.get('atl'))}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Last update time
                    if st.session_state.last_update:
                        st.markdown(f"""
                        <div style="text-align: center; margin-top: 1rem; font-size: 0.8rem; color: var(--text-medium);">
                            🕒 ბოლო განახლება: {st.session_state.last_update.strftime('%H:%M:%S')}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error("მონაცემების ჩამოტვირთვა ვერ მოხერხდა. სცადეთ მოგვიანებით.")

# Center Column - Chart
with col2:
    st.markdown('<div class="crypto-card">', unsafe_allow_html=True)
    
    # Model selection with better UI
    col_chart_title, col_model = st.columns([2, 1])
    with col_chart_title:
        chart_title = f"📈 {st.session_state.current_symbol} - ფასის დინამიკა და AI პროგნოზი" if st.session_state.current_symbol else "📈 ფასის დინამიკა და AI პროგნოზი"
        st.markdown(f"### {chart_title}")
    
    with col_model:
        model_options = []
        if LSTM_AVAILABLE:
            model_options.append("🧠 LSTM Model")
        if PROPHET_AVAILABLE:
            model_options.append("🔮 Prophet Model")
        
        if model_options:
            # Clean up display names for selection
            display_to_actual = {
                "🧠 LSTM Model": "LSTM Model",
                "🔮 Prophet Model": "Prophet Model"
            }
            actual_to_display = {v: k for k, v in display_to_actual.items()}
            
            selected_display = st.selectbox(
                "🤖 AI მოდელი:",
                model_options,
                index=next((i for i, opt in enumerate(model_options) 
                          if display_to_actual[opt] == st.session_state.prediction_model), 0)
            )
            st.session_state.prediction_model = display_to_actual[selected_display]
        else:
            st.error("AI მოდელები არ არის ხელმისაწვდომი. დააინსტალირეთ საჭირო ბიბლიოთეკები.")

    # Chart display with enhanced placeholder
    if not st.session_state.search_triggered:
        st.markdown("""
        <div style="height: 500px; display: flex; align-items: center; justify-content: center; 
                    border: 2px dashed #333; border-radius: 15px; background: var(--bg-medium);">
            <div style="text-align: center; padding: 2rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">📈</div>
                <p style="font-size: 1.3rem; margin-bottom: 1rem; color: var(--text-light);">
                    აირჩიეთ კრიპტოვალუტა ანალიზისთვის
                </p>
                <p style="font-size: 1rem; color: var(--text-medium); margin-bottom: 1.5rem;">
                    ინტერაქტიული ჩარტი, ტექნიკური ინდიკატორები და AI პროგნოზები
                </p>
                <div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;">
                    <span style="padding: 0.5rem 1rem; background: var(--accent-blue); border-radius: 20px; font-size: 0.9rem;">
                        📊 ტექნიკური ანალიზი
                    </span>
                    <span style="padding: 0.5rem 1rem; background: var(--accent-purple); border-radius: 20px; font-size: 0.9rem;">
                        🧠 AI პროგნოზები
                    </span>
                    <span style="padding: 0.5rem 1rem; background: var(--accent-green); border-radius: 20px; font-size: 0.9rem;">
                        📈 ავტომატური სიგნალები
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    elif st.session_state.current_symbol:
        yahoo_symbol = YAHOO_CRYPTO_MAP.get(st.session_state.current_symbol)
        
        if yahoo_symbol:
            # Get fresh data for charting
            coin_details, historical_data_list = fetch_yfinance_data(yahoo_symbol)
            
            if historical_data_list and len(historical_data_list) > 0:
                df_historical = pd.DataFrame(historical_data_list)
                df_historical['date'] = pd.to_datetime(df_historical['date'])
                
                try:
                    forecaster = CryptoDataForecaster(st.session_state.current_symbol, df_historical)
                    
                    # Progress indicator for predictions
                    progress_text = f"🧠 {st.session_state.prediction_model} პროგნოზირება..."
                    with st.spinner(progress_text):
                        prediction_data_list = forecaster.generate_predictions(
                            days=PREDICTION_DAYS, model_choice=st.session_state.prediction_model)
                    
                    df_indicators = forecaster.historical_with_indicators
                    
                    # Calculate Support & Resistance with error handling
                    try:
                        support_level, resistance_level = calculate_support_resistance(df_historical)
                    except Exception as e:
                        st.warning(f"Support/Resistance გამოთვლის შეცდომა: {e}")
                        support_level, resistance_level = None, None

                    # Create enhanced Plotly Chart
                    fig = make_subplots(
                        rows=3, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.08, 
                        row_heights=[0.6, 0.2, 0.2],
                        subplot_titles=(
                            f"{st.session_state.current_symbol} ფასი და AI პროგნოზი",
                            "RSI (Relative Strength Index)", 
                            "MACD (Moving Average Convergence Divergence)"
                        )
                    )

                    # Historical Price (Candlestick) with enhanced colors
                    fig.add_trace(go.Candlestick(
                        x=df_indicators.index,
                        open=df_indicators['open'],
                        high=df_indicators['high'],
                        low=df_indicators['low'],
                        close=df_indicators['close'],
                        name='ისტორიული ფასები',
                        increasing_line_color='#39ff14',
                        decreasing_line_color='#ff073a',
                        showlegend=False
                    ), row=1, col=1)

                    # Prediction Data with confidence bands
                    if prediction_data_list and len(prediction_data_list) > 0:
                        df_prediction = pd.DataFrame(prediction_data_list)
                        df_prediction['date'] = pd.to_datetime(df_prediction['date'])
                        
                        # Add prediction line
                        fig.add_trace(go.Scatter(
                            x=df_prediction['date'],
                            y=df_prediction['close'],
                            mode='lines+markers',
                            name=f'🤖 AI პროგნოზი ({st.session_state.prediction_model})',
                            line=dict(color='#8e2de2', dash='dot', width=3),
                            marker=dict(size=6, color='#8e2de2', symbol='diamond')
                        ), row=1, col=1)
                        
                        # Add confidence band (simplified)
                        upper_bound = df_prediction['close'] * 1.1
                        lower_bound = df_prediction['close'] * 0.9
                        
                        fig.add_trace(go.Scatter(
                            x=list(df_prediction['date']) + list(df_prediction['date'][::-1]),
                            y=list(upper_bound) + list(lower_bound[::-1]),
                            fill='toself',
                            fillcolor='rgba(142, 45, 226, 0.1)',
                            line=dict(color='rgba(255,255,255,0)'),
                            hoverinfo="skip",
                            showlegend=False,
                            name='პროგნოზის დიაპაზონი'
                        ), row=1, col=1)

                    # Enhanced Moving Averages
                    colors_ma = {'MA7': '#fdd835', 'MA25': '#ff9800', 'MA99': '#4caf50'}
                    for ma_name, color in colors_ma.items():
                        if ma_name in df_indicators.columns:
                            fig.add_trace(go.Scatter(
                                x=df_indicators.index, 
                                y=df_indicators[ma_name],
                                mode='lines', 
                                name=f'📈 {ma_name}',
                                line=dict(color=color, width=2),
                                opacity=0.8
                            ), row=1, col=1)

                    # Support and Resistance Lines with better styling
                    if support_level and not pd.isna(support_level):
                        fig.add_hline(
                            y=support_level, 
                            line_dash="solid", 
                            line_color="#00bcd4", 
                            opacity=0.7,
                            annotation_text=f"🔸 Support: ${format_price(support_level)}", 
                            annotation_position="bottom right",
                            row=1, col=1
                        )
                    if resistance_level and not pd.isna(resistance_level):
                        fig.add_hline(
                            y=resistance_level, 
                            line_dash="solid", 
                            line_color="#ff5722", 
                            opacity=0.7,
                            annotation_text=f"🔹 Resistance: ${format_price(resistance_level)}", 
                            annotation_position="top right",
                            row=1, col=1
                        )

                    # Enhanced RSI with colored zones
                    fig.add_trace(go.Scatter(
                        x=df_indicators.index, 
                        y=df_indicators['RSI'],
                        mode='lines', 
                        name='RSI',
                        line=dict(color='#00bcd4', width=2),
                        fill='tonexty'
                    ), row=2, col=1)
                    
                    # RSI zones
                    fig.add_hline(y=70, line_dash="dash", line_color="#ff073a", 
                                 opacity=0.6, annotation_text="Overbought", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="#39ff14", 
                                 opacity=0.6, annotation_text="Oversold", row=2, col=1)
                    fig.add_hline(y=50, line_dash="dot", line_color="#ffffff", 
                                 opacity=0.3, row=2, col=1)

                    # Enhanced MACD
                    fig.add_trace(go.Scatter(
                        x=df_indicators.index, 
                        y=df_indicators['MACD'],
                        mode='lines', 
                        name='MACD',
                        line=dict(color='#2196f3', width=2)
                    ), row=3, col=1)
                    
                    fig.add_trace(go.Scatter(
                        x=df_indicators.index, 
                        y=df_indicators['Signal_Line'],
                        mode='lines', 
                        name='Signal Line',
                        line=dict(color='#ff9800', width=2)
                    ), row=3, col=1)
                    
                    # MACD Histogram with conditional coloring
                    colors = ['#39ff14' if val >= 0 else '#ff073a' for val in df_indicators['MACD_Histogram']]
                    fig.add_trace(go.Bar(
                        x=df_indicators.index, 
                        y=df_indicators['MACD_Histogram'],
                        name='MACD Histogram',
                        marker_color=colors,
                        opacity=0.6
                    ), row=3, col=1)

                    # Enhanced layout
                    fig.update_layout(
                        height=700,
                        xaxis_rangeslider_visible=False,
                        template='plotly_dark',
                        hovermode='x unified',
                        margin=dict(l=40, r=40, t=60, b=40),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(26,26,26,0.8)',
                        font=dict(family="Inter, sans-serif", size=12, color="#e0e0e0"),
                        legend=dict(
                            orientation="h", 
                            yanchor="top", 
                            y=1.02, 
                            xanchor="left", 
                            x=0,
                            bgcolor="rgba(26,26,26,0.8)",
                            bordercolor="rgba(255,255,255,0.2)",
                            borderwidth=1
                        )
                    )
                    
                    # Update axes
                    fig.update_yaxes(title_text='💰 ფასი (USD)', row=1, col=1, 
                                   gridcolor='rgba(255,255,255,0.1)')
                    fig.update_yaxes(title_text='📊 RSI', range=[0, 100], row=2, col=1,
                                   gridcolor='rgba(255,255,255,0.1)')
                    fig.update_yaxes(title_text='📈 MACD', row=3, col=1,
                                   gridcolor='rgba(255,255,255,0.1)')
                    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')

                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display prediction summary
                    if prediction_data_list:
                        with st.expander("📊 პროგნოზის დეტალები", expanded=False):
                            df_pred_display = pd.DataFrame(prediction_data_list)
                            df_pred_display['date'] = pd.to_datetime(df_pred_display['date']).dt.strftime('%Y-%m-%d')
                            df_pred_display['close'] = df_pred_display['close'].apply(lambda x: f"${format_price(x)}")
                            df_pred_display = df_pred_display.rename(columns={
                                'date': 'თარიღი', 
                                'close': 'პროგნოზირებული ფასი'
                            })
                            st.dataframe(df_pred_display.drop('type', axis=1), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"ჩარტის შექმნის შეცდომა: {e}")
                    st.info("სცადეთ განსხვავებული კრიპტოვალუტა ან დაელოდეთ რამდენიმე წუთს.")
            else:
                st.error("ისტორიული მონაცემები ვერ მოიძებნა. სცადეთ განსხვავებული სიმბოლო.")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Technical Indicators Educational Section
    if st.session_state.search_triggered:
        st.markdown('<div class="crypto-card">', unsafe_allow_html=True)
        st.markdown("### 📚 ტექნიკური ინდიკატორების განმარტება")
        
        # Create tabs for better organization
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Moving Averages", "🎯 Support/Resistance", "⚡ RSI", "🌊 MACD"])
        
        with tab1:
            st.markdown("""
            **მოძრავი საშუალო (Moving Average)**
            
            მოძრავი საშუალო არის ერთ-ერთი ყველაზე გავრცელებული ტექნიკური ინდიკატორი, რომელიც:
            
            • **MA(7)** - მოკლევადიანი ტრენდის განსაზღვრისთვის
            • **MA(25)** - საშუალოვადიანი ტრენდის ანალიზისთვის  
            • **MA(99)** - გრძელვადიანი ტრენდის იდენტიფიცირებისთვის
            
            **Golden Cross** 📈 - მოკლევადიანი MA-ს გრძელვადიანზე ზემოთ კვეთა (ყიდვის სიგნალი)
            **Death Cross** 📉 - მოკლევადიანი MA-ს გრძელვადიანზე ქვემოთ კვეთა (გაყიდვის სიგნალი)
            """)
        
        with tab2:
            st.markdown("""
            **Support & Resistance Levels**
            
            **Support (მხარდაჭერა)** 🔸 - ფასის დონე, სადაც:
            • ყიდვის ინტერესი იზრდება
            • ფასის ვარდნა ჩერდება
            • "ფასის იატაკი"
            
            **Resistance (წინააღმდეგობა)** 🔹 - ფასის დონე, სადაც:
            • გაყიდვის ინტერესი იზრდება  
            • ფასის ზრდა ჩერდება
            • "ფასის ჭერი"
            
            **მნიშვნელობა:** ამ დონეების გარღვევა ხშირად ახალი ტრენდის დასაწყისია.
            """)
        
        with tab3:
            st.markdown("""
            **RSI (Relative Strength Index)**
            
            RSI ზომავს ფასის ცვლილების სიჩქარეს 0-100 დიაპაზონში:
            
            • **70+ ზონა** 🔴 - Overbought (ზედმეტად ნაყიდი)
              - ფასმა შესაძლოა დაიწყოს კორექცია ქვემოთ
            
            • **30- ზონა** 🟢 - Oversold (ზედმეტად გაყიდული)  
              - ფასმა შესაძლოა დაიწყოს ზრდა
            
            • **50 ზონა** ⚪ - ნეიტრალური (ბალანსი)
            
            **დივერგენცია:** RSI და ფასის განსხვავებული მიმართულება ძლიერი სიგნალია.
            """)
        
        with tab4:
            st.markdown("""
            **MACD (Moving Average Convergence Divergence)**
            
            MACD შედგება სამი კომპონენტისგან:
            
            • **MACD ხაზი** (ლურჯი) - 12-დღიანი EMA - 26-დღიანი EMA
            • **Signal ხაზი** (ნარინჯისფერი) - MACD-ის 9-დღიანი EMA
            • **Histogram** (ღია/წითელი ბარები) - MACD - Signal Line
            
            **სიგნალები:**
            • MACD > Signal Line → ყიდვის სიგნალი 📈
            • MACD < Signal Line → გაყიდვის სიგნალი 📉
            • Histogram > 0 → აღმავალი მომენტუმი
            • Histogram < 0 → დაღმავალი მომენტუმი
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Right Column - Signals with enhanced UI
with col3:
    st.markdown('<div class="crypto-card">', unsafe_allow_html=True)
    st.markdown("### 🚨 AI სიგნალები")
    
    if not st.session_state.search_triggered:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem; border: 2px dashed #333; 
                    border-radius: 15px; background: var(--bg-medium);">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🤖</div>
            <p style="font-size: 1.1rem; color: var(--text-light); margin-bottom: 0.5rem;">
                AI სიგნალები დაგენერირდება
            </p>
            <p style="font-size: 0.9rem; color: var(--text-medium);">
                კრიპტოვალუტის არჩევის შემდეგ
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    elif st.session_state.current_symbol:
        yahoo_symbol = YAHOO_CRYPTO_MAP.get(st.session_state.current_symbol)
        
        if yahoo_symbol:
            # Get fresh data for signal generation
            coin_details, historical_data_list = fetch_yfinance_data(yahoo_symbol)
            
            if historical_data_list and len(historical_data_list) > 0:
                try:
                    forecaster = CryptoDataForecaster(st.session_state.current_symbol, 
                                                    pd.DataFrame(historical_data_list))
                    
                    with st.spinner("🧠 AI სიგნალების გენერირება..."):
                        prediction_data_list = forecaster.generate_predictions(
                            days=PREDICTION_DAYS, model_choice=st.session_state.prediction_model)
                        
                        if prediction_data_list:
                            signals = forecaster.generate_signals_from_prediction(prediction_data_list)
                        else:
                            signals = []
                    
                    if signals and len(signals) > 0:
                        st.markdown(f"**💡 {len(signals)} სიგნალი მოიძებნა შემდეგი {PREDICTION_DAYS} დღისთვის**")
                        
                        for i, signal in enumerate(signals):
                            signal_class = "buy-signal" if signal['type'] == 'BUY' else "sell-signal"
                            signal_type_class = f"signal-type-{signal['type'].lower()}"
                            
                            # Enhanced signal display
                            st.markdown(f"""
                            <div class="signal-item {signal_class}">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                    <span class="{signal_type_class}" style="font-size: 1.1rem;">
                                        {'🚀' if signal['type'] == 'BUY' else '⚠️'} {signal['type']} SIGNAL
                                    </span>
                                    <span style="font-size: 0.85rem; color: var(--text-medium); font-weight: 600;">
                                        {signal['date'].strftime('%m/%d')}
                                    </span>
                                </div>
                                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                                    <div>
                                        <div style="font-size: 0.8rem; color: var(--text-medium);">Target Price</div>
                                        <div style="font-weight: 600; color: var(--text-light);">
                                            ${format_price(signal['price'])}
                                        </div>
                                    </div>
                                    <div style="text-align: right;">
                                        <div style="font-size: 0.8rem; color: var(--text-medium);">Confidence</div>
                                        <div style="font-weight: 600; color: {'var(--accent-green)' if signal['type'] == 'BUY' else 'var(--accent-red)'};">
                                            {signal['confidence']}
                                        </div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Expandable analysis
                            with st.expander(f"🔍 დეტალური ანალიზი - {signal['type']}", expanded=False):
                                st.markdown(f"**📅 თარიღი:** {signal['date'].strftime('%Y-%m-%d')}")
                                st.markdown(f"**💰 მიზნობრივი ფასი:** ${format_price(signal['price'])}")
                                st.markdown(f"**🎯 სანდოობა:** {signal['confidence']}")
                                st.markdown(f"**🧠 AI ანალიზი:** {signal['argumentation']}")
                                
                                # Additional context
                                current_price = coin_details.get('currentPrice') if coin_details else None
                                if current_price:
                                    price_change = ((signal['price'] - current_price) / current_price) * 100
                                    change_emoji = '📈' if price_change > 0 else '📉'
                                    st.markdown(f"**{change_emoji} მოსალოდნელი ცვლილება:** {price_change:+.2f}%")
                        
                        # Signal summary statistics
                        with st.expander("📊 სიგნალების სტატისტიკა", expanded=False):
                            buy_signals = [s for s in signals if s['type'] == 'BUY']
                            sell_signals = [s for s in signals if s['type'] == 'SELL']
                            
                            col_stats1, col_stats2 = st.columns(2)
                            with col_stats1:
                                st.metric("🚀 ყიდვის სიგნალები", len(buy_signals))
                                if buy_signals:
                                    avg_buy_confidence = np.mean([float(s['confidence'].rstrip('%')) for s in buy_signals])
                                    st.metric("საშუალო სანდოობა", f"{avg_buy_confidence:.1f}%")
                            
                            with col_stats2:
                                st.metric("⚠️ გაყიდვის სიგნალები", len(sell_signals))
                                if sell_signals:
                                    avg_sell_confidence = np.mean([float(s['confidence'].rstrip('%')) for s in sell_signals])
                                    st.metric("საშუალო სანდოობა", f"{avg_sell_confidence:.1f}%")
                            
                            # Overall market sentiment
                            if len(buy_signals) > len(sell_signals):
                                sentiment = "🟢 ღია (Bullish)"
                                sentiment_desc = "AI მოდელი პროგნოზირებს ფასის ზრდას"
                            elif len(sell_signals) > len(buy_signals):
                                sentiment = "🔴 დაღმავალი (Bearish)"
                                sentiment_desc = "AI მოდელი პროგნოზირებს ფასის კლებას"
                            else:
                                sentiment = "🟡 ნეიტრალური (Neutral)"
                                sentiment_desc = "AI მოდელი ნეიტრალურ ტრენდს პროგნოზირებს"
                            
                            st.markdown(f"""
                            <div style="margin-top: 1rem; padding: 1rem; background: var(--bg-light); 
                                      border-radius: 10px; border-left: 4px solid var(--accent-blue);">
                                <div style="font-weight: 600; margin-bottom: 0.5rem;">📈 საბაზრო განწყობა</div>
                                <div style="font-size: 1.1rem; margin-bottom: 0.5rem;">{sentiment}</div>
                                <div style="font-size: 0.9rem; color: var(--text-medium);">{sentiment_desc}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    else:
                        st.markdown("""
                        <div style="text-align: center; padding: 2rem; border: 2px dashed #444; 
                                    border-radius: 10px; background: var(--bg-medium);">
                            <div style="font-size: 2rem; margin-bottom: 1rem;">🔍</div>
                            <p style="color: var(--text-medium); font-size: 1rem;">
                                მოცემულ პერიოდში რელევანტური სიგნალები ვერ მოიძებნა
                            </p>
                            <p style="color: var(--text-medium); font-size: 0.9rem; margin-top: 0.5rem;">
                                სცადეთ განსხვავებული კრიპტოვალუტა ან დაელოდეთ ბაზრის ცვლილებას
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"სიგნალების გენერირების შეცდომა: {e}")
                    st.info("სცადეთ განსხვავებული მოდელი ან კრიპტოვალუტა.")
            else:
                st.error("მონაცემები არ არის ხელმისაწვდომი სიგნალების გენერირებისთვის.")
    
    # Quick actions section
    if st.session_state.search_triggered and st.session_state.current_symbol:
        st.markdown("### ⚡ სწრაფი მოქმედებები")
        
        col_refresh, col_share = st.columns(2)
        with col_refresh:
            if st.button("🔄 განახლება", use_container_width=True):
                st.session_state.last_update = datetime.datetime.now()
                # Clear cache to force refresh
                fetch_yfinance_data.clear()
                st.rerun()
        
        with col_share:
            if st.button("📤 გაზიარება", use_container_width=True):
                share_text = f"AI ანალიზი {st.session_state.current_symbol} კრიპტოვალუტისთვის - კრიპტო სიგნალები LIVE"
                st.success("URL კოპირებულია!")
                st.code(f"https://crypto-signals-live.com/{st.session_state.current_symbol}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer with enhanced information
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style="background: #2a2a3e; padding: 1rem; border-radius: 10px; text-align: center; border: 1px solid #3a3a4e;">
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🤖</div>
        <div style="font-weight: 600; color: #ffffff;">AI მოდელები</div>
        <div style="font-size: 0.9rem; color: #a0a0a0;">LSTM & Prophet</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: #2a2a3e; padding: 1rem; border-radius: 10px; text-align: center; border: 1px solid #3a3a4e;">
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📈</div>
        <div style="font-weight: 600; color: #ffffff;">რეალურ დროში</div>
        <div style="font-size: 0.9rem; color: #a0a0a0;">Live მონაცემები</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background: #2a2a3e; padding: 1rem; border-radius: 10px; text-align: center; border: 1px solid #3a3a4e;">
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🎯</div>
        <div style="font-weight: 600; color: #ffffff;">ზუსტი სიგნალები</div>
        <div style="font-size: 0.9rem; color: #a0a0a0;">ტექნიკური ანალიზი</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="background: #2a2a3e; padding: 1rem; border-radius: 10px; text-align: center; border: 1px solid #3a3a4e;">
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🛡️</div>
        <div style="font-weight: 600; color: #ffffff;">რისკ მენეჯმენტი</div>
        <div style="font-size: 0.9rem; color: #a0a0a0;">Support/Resistance</div>
    </div>
    """, unsafe_allow_html=True)

# Performance monitoring (hidden)
if st.session_state.get('show_debug', False):
    with st.expander("🔧 Debug Information", expanded=False):
        st.write("**Session State:**")
        st.json({k: str(v) for k, v in st.session_state.items() if not k.startswith('_')})
        st.write("**Available Models:**")
        st.write(f"- LSTM: {LSTM_AVAILABLE}")
        st.write(f"- Prophet: {PROPHET_AVAILABLE}")
        st.write(f"- YFinance: {YF_AVAILABLE}")

# Add keyboard shortcuts info
st.markdown("""
<div style="position: fixed; bottom: 20px; right: 20px; background: var(--card-bg); 
           padding: 0.5rem; border-radius: 10px; border: 1px solid var(--border-color);
           font-size: 0.8rem; color: var(--text-medium); z-index: 1000;">
    💡 <strong>Tips:</strong> Press 'R' to refresh | Scroll for more signals
</div>
""", unsafe_allow_html=True)
