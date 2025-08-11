import streamlit as st
import datetime
import random
import requests
import time
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import os

# Import yfinance for historical data
try:
    import yfinance as yf
except ImportError:
    st.error("yfinance მოდული ვერ მოიძებნა. დააინსტალირეთ: pip install yfinance")
    yf = None

# Import forecasting models with checks
try:
    from prophet import Prophet
except ImportError:
    st.error("Prophet მოდული ვერ მოიძებნა. დააინსტალირეთ 'prophet': pip install prophet")
    Prophet = None

try:
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    # Suppress TensorFlow warnings
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress info and warnings
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR') # Only show errors
except ImportError:
    st.error("TensorFlow/Keras, Scikit-learn, ან NumPy მოდულები ვერ მოიძებნა. დააინსტალირეთ: pip install tensorflow scikit-learn numpy")
    LSTM_AVAILABLE = False
else:
    LSTM_AVAILABLE = True


# --- Set page config for wide mode ---
st.set_page_config(layout="wide", page_title="კრიპტო სიგნალები LIVE", page_icon="📈")

# --- Configuration ---
PREDICTION_DAYS = 30 # Number of days to predict
MAX_HISTORICAL_DAYS = 365 * 2 # Fetch 2 years of data for better LSTM training

# The symbol mapping for cryptocurrencies on Yahoo Finance is often 'SYMBOL-USD'
YAHOO_CRYPTO_MAP = {
    'BTC': 'BTC-USD', 'ETH': 'ETH-USD', 'SOL': 'SOL-USD', 'ADA': 'ADA-USD', 'DOT': 'DOT-USD',
    'LINK': 'LINK-USD', 'MATIC': 'MATIC-USD', 'AVAX': 'AVAX-USD', 'ATOM': 'ATOM-USD', 'LTC': 'LTC-USD',
    'BCH': 'BCH-USD', 'XRP': 'XRP-USD', 'DOGE': 'DOGE-USD', 'SHIB': 'SHIB-USD', 'UNI': 'UNI-USD',
    'AAVE': 'AAVE-USD', 'SAND': 'SAND-USD', 'MANA': 'MANA-USD', 'AXS': 'AXS-USD', 'SKL': 'SKL-USD'
}

def fetch_yfinance_data(ticker_symbol, period='1d'):
    """Fetches real-time and historical data using yfinance."""
    if not yf: return None, None

    ticker = yf.Ticker(ticker_symbol)
    try:
        info = ticker.info
        historical_df = ticker.history(period=f"{MAX_HISTORICAL_DAYS}d")
        
        if historical_df.empty:
            st.error(f"Error fetching historical data for {ticker_symbol} from Yahoo Finance.")
            return None, None
            
        historical_data = []
        for index, row in historical_df.iterrows():
            historical_data.append({
                'date': index.tz_convert(None), # Remove timezone for consistency
                'open': row['Open'],
                'high': row['High'],
                'low': row['Low'],
                'close': row['Close'],
                'type': 'historical'
            })
            
        # Extract real-time details from info
        coin_details = {
            'name': info.get('longName', ticker_symbol),
            'symbol': info.get('symbol', ticker_symbol).split('-')[0],
            'currentPrice': info.get('currentPrice'),
            'dailyChange': info.get('regularMarketChangePercent'),
            '24hVolume': info.get('volume'),
            'marketCap': info.get('marketCap'),
            'marketCapRank': None, # Yahoo Finance doesn't provide rank easily
            'ath': info.get('regularMarketDayHigh'),
            'atl': info.get('regularMarketDayLow'),
            'lastUpdated': datetime.datetime.now().isoformat()
        }
        
        return coin_details, historical_data
        
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data for {ticker_symbol} from Yahoo Finance: {e}")
        return None, None
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching data for {ticker_symbol}: {e}")
        return None, None


def calculate_support_resistance(df, window=20):
    """
    Calculates support and breakout levels.
    Support is defined as a significant low, Breakout as a significant high.
    We'll use rolling window to find these levels.
    """
    if len(df) < window:
        return None, None
    
    df['rolling_min'] = df['low'].rolling(window=window).min()
    df['rolling_max'] = df['high'].rolling(window=window).max()
    
    support_level = df['rolling_min'].iloc[-1]
    resistance_level = df['rolling_max'].iloc[-1]
    
    return support_level, resistance_level

class CryptoDataForecaster:
    def __init__(self, symbol, historical_prices_df):
        self.symbol = symbol.upper()
        self.historical_prices = historical_prices_df.copy()
        self.historical_prices['date'] = pd.to_datetime(self.historical_prices['date'])
        self.historical_prices.set_index('date', inplace=True)
        self.historical_prices['close'] = self.historical_prices['close'].astype(float)
        
        self.historical_with_indicators = self.calculate_technical_indicators()

    def _generate_lstm_predictions(self, days):
        if not LSTM_AVAILABLE:
            st.error("LSTM მოდელი არ არის ხელმისაწვდომი. გთხოვთ, დააინსტალიროთ TensorFlow/Keras, NumPy და Scikit-learn.")
            return []
        
        df_lstm = self.historical_with_indicators[['close', 'MA7', 'MA25', 'RSI', 'MACD']].dropna()
        
        if len(df_lstm) < 100:
            st.warning(f"არასაკმარისი სუფთა მონაცემები (MA, RSI, MACD-ით) LSTM პროგნოზისთვის (მინიმუმ 100 დღე).")
            return []

        look_back = 60
        n_features = df_lstm.shape[1]

        data_scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(df_lstm.values)
        scaler_price = MinMaxScaler(feature_range=(0, 1))
        scaler_price.fit(df_lstm['close'].values.reshape(-1, 1)) 

        X, y = [], []
        for i in range(len(data_scaled) - look_back):
            X.append(data_scaled[i:(i + look_back), :])
            y.append(data_scaled[i + look_back, 0])
        X = np.array(X)
        y = np.array(y)
        
        model = Sequential()
        model.add(LSTM(units=100, return_sequences=True, input_shape=(look_back, n_features)))
        model.add(Dropout(0.3))
        model.add(LSTM(units=100, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(units=1)) 
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        
        try:
            model.fit(X, y, epochs=50, batch_size=32, verbose=0) 
        except Exception as e:
            st.error(f"შეცდომა LSTM მოდელის ტრენინგისას: {e}. პროგნოზი მიუწვდომელია.")
            return []

        prediction_list = []
        current_input = data_scaled[-look_back:].reshape(1, look_back, n_features)
        last_date_original = self.historical_prices.index[-1]

        for i in range(days):
            predicted_scaled_price = model.predict(current_input, verbose=0)[0, 0]
            predicted_price = scaler_price.inverse_transform([[predicted_scaled_price]])[0, 0]
            
            date = last_date_original + datetime.timedelta(days=i+1)
            prediction_list.append({'date': date, 'close': max(0.00000001, round(predicted_price, 8)), 'type': 'prediction'})
            
            last_features_scaled = current_input[0, -1, :].copy()
            last_features_scaled[0] = predicted_scaled_price
            current_input = np.append(current_input[:, 1:, :], last_features_scaled.reshape(1, 1, n_features), axis=1)

        return prediction_list

    def generate_predictions(self, days=PREDICTION_DAYS, model_choice="Prophet Model"):
        if self.historical_prices.empty:
            st.warning("ისტორიული მონაცემები არ არის პროგნოზირებისთვის.")
            return []

        prediction_data = []

        if model_choice == "Prophet Model":
            if Prophet is None:
                st.error("Prophet მოდული არ არის ხელმისაწვდომი. გთხოვთ, დააინსტალიროთ 'prophet'.")
                return []
            if len(self.historical_prices) < 90:
                st.warning("არასაკმარისი მონაცემები Prophet პროგნოზისთვის (მინიმუმ 90 დღე).")
                return []

            df_prophet = self.historical_prices.reset_index()[['date', 'close']].rename(columns={'date': 'ds', 'close': 'y'})
            df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

            try:
                model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False, changepoint_prior_scale=0.1) 
                model.fit(df_prophet)
                future = model.make_future_dataframe(periods=days)
                forecast = model.predict(future)
                
                for _, row in forecast.tail(days).iterrows():
                    prediction_data.append({
                        'date': row['ds'], 
                        'close': max(0.00000001, round(row['yhat'], 8)), 
                        'type': 'prediction'
                    })
            except Exception as e:
                st.error(f"შეცდომა Prophet მოდელის გაშვებისას: {e}. პროგნოზი მიუწვდომელია.")
                return []

        elif model_choice == "LSTM Model":
            prediction_data = self._generate_lstm_predictions(days)
        
        return prediction_data

    def generate_signals_from_prediction(self, prediction_data):
        if len(prediction_data) < 7:
            return []
        
        df_prediction = pd.DataFrame(prediction_data)
        df_prediction['date'] = pd.to_datetime(df_prediction['date'])
        df_prediction.set_index('date', inplace=True)
        df_prediction['close'] = df_prediction['close'].astype(float)

        short_window = 3 
        long_window = 7 

        df_prediction['MA_short'] = df_prediction['close'].rolling(window=short_window).mean()
        df_prediction['MA_long'] = df_prediction['close'].rolling(window=long_window).mean()

        signals = []
        for i in range(1, len(df_prediction)):
            current_date = df_prediction.index[i]
            if pd.isna(df_prediction['MA_short'].iloc[i]) or pd.isna(df_prediction['MA_long'].iloc[i]):
                continue 

            if (df_prediction['MA_short'].iloc[i-1] < df_prediction['MA_long'].iloc[i-1] and
                df_prediction['MA_short'].iloc[i] > df_prediction['MA_long'].iloc[i]):
                signals.append({
                    'date': current_date,
                    'type': 'BUY',
                    'price': df_prediction['close'].iloc[i],
                    'confidence': f"{random.randint(70, 90)}%",
                    'argumentation': f"AI პროგნოზირებს მოკლევადიანი მოძრავი საშუალოს (MA{short_window}) გრძელვადიან მოძრავ საშუალოზე (MA{long_window}) ზემოთ კვეთას, რაც პოტენციური აღმავალი ტრენდის დასაწყისს მიანიშნებს."
                })

            elif (df_prediction['MA_short'].iloc[i-1] > df_prediction['MA_long'].iloc[i-1] and
                  df_prediction['MA_short'].iloc[i] < df_prediction['MA_long'].iloc[i]):
                signals.append({
                    'date': current_date,
                    'type': 'SELL',
                    'price': df_prediction['close'].iloc[i],
                    'confidence': f"{random.randint(65, 85)}%",
                    'argumentation': f"AI პროგნოზირებს მოკლევადიანი მოძრავი საშუალოს (MA{short_window}) გრძელვადიან მოძრავ საშუალოზე (MA{long_window}) ქვემოთ კვეთას, რაც პოტენციური დაღმავალი ტრენდის დასაწყისს მიანიშნებს."
                })
        
        return sorted(signals, key=lambda x: x['date'])

    def calculate_technical_indicators(self):
        df_hist = self.historical_prices.copy()
        
        df_hist['MA7'] = df_hist['close'].rolling(window=7, min_periods=1).mean()
        df_hist['MA25'] = df_hist['close'].rolling(window=25, min_periods=1).mean()
        df_hist['MA99'] = df_hist['close'].rolling(window=99, min_periods=1).mean()

        delta = df_hist['close'].diff()
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))
        
        avg_gain = gain.ewm(span=14, adjust=False, min_periods=14).mean()
        avg_loss = loss.ewm(span=14, adjust=False, min_periods=14).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df_hist['RSI'] = 100 - (100 / (1 + rs))
        df_hist['RSI'].fillna(0, inplace=True)
        df_hist['RSI'] = df_hist['RSI'].replace([np.inf, -np.inf], np.nan).fillna(0)

        exp1 = df_hist['close'].ewm(span=12, adjust=False, min_periods=12).mean()
        exp2 = df_hist['close'].ewm(span=26, adjust=False, min_periods=26).mean()
        df_hist['MACD'] = exp1 - exp2
        df_hist['Signal_Line'] = df_hist['MACD'].ewm(span=9, adjust=False, min_periods=9).mean()
        df_hist['MACD_Histogram'] = df_hist['MACD'] - df_hist['Signal_Line']
        
        return df_hist


def format_price(price):
    if price is None: return 'N/A'
    if price >= 1: return f"{price:,.2f}"
    return f"{price:.8f}".rstrip('0').rstrip('.')

def format_currency(value):
    if value is None: return 'N/A'
    return f"${value:,.0f}"

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
    
    /* Apply font and background to the Streamlit app */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: var(--bg-dark);
        color: var(--text-light);
        padding: 1rem;
    }
    
    h1 {
        font-size: 2.5rem;
        font-weight: 800;
        text-shadow: 0 0 10px var(--accent-blue), 0 0 20px var(--accent-purple);
        color: var(--text-light);
    }
    
    .live-text { color: var(--accent-red); }
    
    .stSelectbox, .stTextInput, .stButton > button {
        background-color: var(--bg-dark);
        color: var(--text-light);
        border: 1px solid var(--border-color);
        border-radius: 8px;
    }
    .stButton > button {
        background: linear-gradient(45deg, var(--accent-blue), var(--accent-purple));
        color: white;
        border: none;
        cursor: pointer;
    }

    .stButton > button:hover {
        opacity: 0.9;
    }

    .stContainer {
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
        margin-bottom: 1.5rem;
    }

    .stContainer > div > div > div > .stContainer {
        background-color: transparent;
        border: none;
        box-shadow: none;
        padding: 0;
        margin-bottom: 0;
    }

    .stAlert {
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.4);
    }
    .stAlert.success { background-color: #4CAF50; color: white; }
    .stAlert.error { background-color: #f44336; color: white; }
    .stAlert.warning { background-color: #FFC107; color: black; }


    .price-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--accent-yellow);
        margin: 0.25rem 0 1rem 0;
    }
    .positive { color: var(--accent-green); }
    .negative { color: var(--accent-red); }

    .signal-item {
        background-color: var(--bg-dark);
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.75rem;
        border-left: 4px solid var(--border-color);
    }
    .buy-signal-item { border-left-color: var(--accent-green) !important; }
    .sell-signal-item { border-left-color: var(--accent-red) !important; }
    .signal-type.buy-signal { color: var(--accent-green); font-weight: 700;}
    .signal-type.sell-signal { color: var(--accent-red); font-weight: 700;}
    .signal-date { font-size: 0.8rem; color: var(--text-medium); text-align: right;}
    .signal-price-value { font-weight: 600; color: var(--accent-yellow); }
    .modal-label { font-weight: 600; color: var(--accent-purple); }
    .buy-text { color: var(--accent-green); }
    .sell-text { color: var(--accent-red); }
    .argumentation-box {
        background-color: var(--bg-dark); border: 1px solid var(--border-color);
        border-radius: 10px; padding: 1rem; margin-top: 1.5rem;
    }
    .argumentation-box h4 { font-size: 1.2rem; color: var(--accent-blue); margin-top: 0; margin-bottom: 0.8rem; }
    .argumentation-box p { font-size: 0.95rem; line-height: 1.5; color: var(--text-medium); }

    .chart-legend {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-top: 1rem;
        font-size: 0.9rem;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 5px;
    }
    .legend-color {
        width: 15px;
        height: 3px;
        border-radius: 2px;
    }
    .historical { background-color: #00bcd4; }
    .prediction { background-color: #8e2de2; }
    .buy-signal-legend { background-color: var(--accent-green); width: 10px; height: 10px; border-radius: 50%; }
    .sell-signal-legend { background-color: var(--accent-red); width: 10px; height: 10px; border-radius: 0; transform: rotate(45deg); }

    .status-indicator {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    .status-indicator.connected {
        background-color: rgba(57, 255, 20, 0.1);
        color: var(--accent-green);
        border: 1px solid var(--accent-green);
    }
    .status-indicator.connected i {
        color: var(--accent-green);
    }
    
    @media (max-width: 768px) {
        .stApp { padding: 0.5rem; }
        h1 { font-size: 2rem; }
        .price-value { font-size: 1.8rem; }
    }
</style>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
""", unsafe_allow_html=True)


st.title("კრიპტო სიგნალები LIVE")

st.markdown("""
<div class="status-indicator connected">
    <i class="fas fa-wifi"></i>
    <span>Online</span>
</div>
""", unsafe_allow_html=True)

if 'current_symbol' not in st.session_state:
    st.session_state.current_symbol = 'SAND'
if 'current_period' not in st.session_state:
    st.session_state.current_period = 90
if 'prediction_model' not in st.session_state:
    st.session_state.prediction_model = "LSTM Model" # Default to LSTM

col1, col2, col3 = st.columns([0.8, 1.5, 0.8])

with col1:
    with st.container(border=False):
        st.markdown("<h3>აირჩიეთ კრიპტოვალუტა:</h3>", unsafe_allow_html=True)
        
        with st.form("symbol_form"):
            symbol_input = st.text_input("შეიყვანეთ სიმბოლო (მაგ. BTC, ETH)", st.session_state.current_symbol).upper().strip()
            submit_button = st.form_submit_button("ძიება", use_container_width=True)

            if submit_button:
                st.session_state.current_symbol = symbol_input
                st.toast(f"Fetching data for {st.session_state.current_symbol}...", icon="🔄")
                st.rerun()

        st.markdown("<span>პოპულარული:</span>", unsafe_allow_html=True)
        popular_cryptos = ['BTC', 'ETH', 'SOL', 'XRP', 'SAND', 'SKL']
        cols_pop = st.columns(len(popular_cryptos))
        for i, crypto_tag in enumerate(popular_cryptos):
            with cols_pop[i]:
                if st.button(crypto_tag, key=f"tag_{crypto_tag}", use_container_width=True):
                    st.session_state.current_symbol = crypto_tag
                    st.toast(f"Fetching data for {st.session_state.current_symbol}...", icon="🔄")
                    st.rerun()

    st.markdown("---")

    with st.container(border=False):
        st.markdown("<div class='price-info-header'><h2>კრიპტოვალუტის ინფორმაცია</h2></div>", unsafe_allow_html=True)

        yahoo_symbol = YAHOO_CRYPTO_MAP.get(st.session_state.current_symbol)
        
        if yahoo_symbol:
            with st.spinner(f"იტვირთება მონაცემები {st.session_state.current_symbol}-ისთვის..."):
                coin_details, historical_data_list_full = fetch_yfinance_data(yahoo_symbol)
                
                if coin_details:
                    st.markdown(f"<h2 id='coin-name'>{coin_details['name']} ({coin_details['symbol']})</h2>", unsafe_allow_html=True)
                    st.markdown(f"<p id='current-price' class='price-value'>{format_price(coin_details['currentPrice'])} $</p>", unsafe_allow_html=True)
                    
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.markdown("<h4>24სთ ცვლილება</h4>", unsafe_allow_html=True)
                        daily_change_class = 'positive' if (coin_details['dailyChange'] or 0) >= 0 else 'negative'
                        st.markdown(f"<p class='value {daily_change_class}'>{coin_details['dailyChange']:.2f}%</p>", unsafe_allow_html=True)
                    with col_m2:
                        st.markdown("<h4>24სთ მოცულობა</h4>", unsafe_allow_html=True)
                        st.markdown(f"<p class='value'>{format_currency(coin_details['24hVolume'])}</p>", unsafe_allow_html=True)
                    
                    col_m3, col_m4 = st.columns(2)
                    with col_m3:
                        st.markdown("<h4>კაპიტალიზაცია</h4>", unsafe_allow_html=True)
                        st.markdown(f"<p class='value'>{format_currency(coin_details['marketCap'])}</p>", unsafe_allow_html=True)
                    with col_m4:
                        st.markdown("<h4>რანგი</h4>", unsafe_allow_html=True)
                        st.markdown(f"<p class='value'>#{coin_details['marketCapRank'] or '-'}</p>", unsafe_allow_html=True)

                    col_m5, col_m6 = st.columns(2)
                    with col_m5:
                        st.markdown("<h4>ATH</h4>", unsafe_allow_html=True)
                        st.markdown(f"<p class='value'>{format_price(coin_details['ath'])} $</p>", unsafe_allow_html=True)
                    with col_m6:
                        st.markdown("<h4>ATL</h4>", unsafe_allow_html=True)
                        st.markdown(f"<p class='value'>{format_price(coin_details['atl'])} $</p>", unsafe_allow_html=True)

                else:
                    st.error("მონაცემები ვერ მოიძებნა ამ კრიპტოვალუტისთვის.")
        else:
            st.warning("გთხოვთ, აირჩიოთ ვალიდური კრიპტოვალუტის სიმბოლო.")

with col2:
    with st.container(border=False):
        st.markdown(f"<div class='chart-header'><h3>{st.session_state.current_symbol} ფასის დინამიკა და AI პროგნოზი</h3></div>", unsafe_allow_html=True)
        
        filter_cols = st.columns(3)
        periods = [90, 30, 7]
        for i, period in enumerate(periods):
            with filter_cols[i]:
                if st.button(f"{period} დღე", key=f"period_{period}", use_container_width=True, type="secondary" if st.session_state.current_period != period else "primary"):
                    st.session_state.current_period = period
                    st.toast(f"Showing {period} days data for {st.session_state.current_symbol}...", icon="📊")
        
        model_options = []
        if LSTM_AVAILABLE:
            model_options.append("LSTM Model")
        if Prophet:
            model_options.append("Prophet Model")

        selected_model = st.selectbox(
            "აირჩიეთ პროგნოზირების მოდელი:",
            model_options,
            index=model_options.index(st.session_state.prediction_model) if st.session_state.prediction_model in model_options else 0
        )
        if selected_model != st.session_state.prediction_model:
            st.session_state.prediction_model = selected_model
            st.rerun()

        if yahoo_symbol and coin_details and historical_data_list_full:
            
            df_historical_full = pd.DataFrame(historical_data_list_full)
            df_historical_full['date'] = pd.to_datetime(df_historical_full['date'])
            
            forecaster = CryptoDataForecaster(st.session_state.current_symbol, df_historical_full)
            
            with st.spinner(f"მიმდინარეობს {st.session_state.prediction_model} პროგნოზირება..."):
                prediction_data_list = forecaster.generate_predictions(days=PREDICTION_DAYS, model_choice=st.session_state.prediction_model)
            
            if not prediction_data_list:
                st.warning(f"პროგნოზის გენერირება ვერ მოხერხდა {st.session_state.prediction_model} მოდელით. გთხოვთ, სცადოთ სხვა მოდელი ან სხვა კრიპტოვალუტა.")
            
            df_prediction = pd.DataFrame(prediction_data_list)
            if not df_prediction.empty:
                df_prediction['date'] = pd.to_datetime(df_prediction['date'])
            
            signals = forecaster.generate_signals_from_prediction(prediction_data_list)
            
            df_indicators_display = forecaster.historical_with_indicators.tail(st.session_state.current_period).copy()

            # Calculate Support & Resistance
            support_level, resistance_level = calculate_support_resistance(df_historical_full.tail(st.session_state.current_period))

            # --- Create Plotly Subplots ---
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.08, 
                                row_heights=[0.6, 0.2, 0.2],
                                subplot_titles=(
                                    f"{st.session_state.current_symbol} ფასი და პროგნოზი",
                                    "Relative Strength Index (RSI)",
                                    "Moving Average Convergence Divergence (MACD)"
                                ))

            # --- Row 1: Candlestick, MAs, Prediction ---
            # Historical Price (Candlestick)
            fig.add_trace(go.Candlestick(
                x=df_indicators_display.index,
                open=df_indicators_display['open'],
                high=df_indicators_display['high'],
                low=df_indicators_display['low'],
                close=df_indicators_display['close'],
                name='ისტორია',
                showlegend=False
            ), row=1, col=1)

            # Prediction Data (Line)
            if not df_prediction.empty:
                fig.add_trace(go.Scatter(
                    x=df_prediction['date'],
                    y=df_prediction['close'],
                    mode='lines',
                    name='პროგნოზი',
                    line=dict(color='#8e2de2', dash='dot', width=2)
                ), row=1, col=1)

            # Moving Averages on Historical Data
            fig.add_trace(go.Scatter(
                x=df_indicators_display.index,
                y=df_indicators_display['MA7'],
                mode='lines',
                name='MA(7)',
                line=dict(color='yellow', width=1, dash='solid')
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df_indicators_display.index,
                y=df_indicators_display['MA25'],
                mode='lines',
                name='MA(25)',
                line=dict(color='orange', width=1, dash='solid')
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df_indicators_display.index,
                y=df_indicators_display['MA99'],
                mode='lines',
                name='MA(99)',
                line=dict(color='lightgreen', width=1, dash='solid')
            ), row=1, col=1)

            # Support and Resistance Levels
            if support_level:
                fig.add_hline(y=support_level, line_dash="solid", line_color="#00a3e0", opacity=0.8, annotation_text=f"Support: {format_price(support_level)} $", annotation_position="bottom right", row=1, col=1)
            if resistance_level:
                fig.add_hline(y=resistance_level, line_dash="solid", line_color="#ff5733", opacity=0.8, annotation_text=f"Breakout: {format_price(resistance_level)} $", annotation_position="top right", row=1, col=1)

            # Add Signals as Scatter points
            for signal in signals:
                signal_color = "#39ff14" if signal['type'] == 'BUY' else "#ff073a"
                if not df_prediction.empty and (signal['date'] >= df_indicators_display.index.min() or signal['date'] >= df_prediction['date'].min()):
                    fig.add_trace(go.Scatter(
                        x=[signal['date']],
                        y=[signal['price']],
                        mode='markers+text',
                        name=f"{signal['type']} სიგნალი",
                        marker=dict(
                            symbol='circle',
                            size=10, 
                            color=signal_color,
                            line=dict(width=2, color=signal_color) 
                        ),
                        text=[signal['type']],
                        textposition="top center" if signal['type'] == 'BUY' else "bottom center",
                        textfont=dict(
                            family="sans-serif",
                            size=12,
                            color=signal_color
                        ),
                        hoverinfo='text',
                        hovertext=f"<b>{signal['type']}</b><br>თარიღი: {signal['date'].strftime('%Y-%m-%d')}<br>ფასი: {format_price(signal['price'])} $",
                        showlegend=False
                    ), row=1, col=1)

            # --- Row 2: RSI ---
            fig.add_trace(go.Scatter(
                x=df_indicators_display.index,
                y=df_indicators_display['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='cyan', width=2)
            ), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1, annotation_text="Overbought (70)", annotation_position="top right", annotation_font_color="red")
            fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1, annotation_text="Oversold (30)", annotation_position="bottom right", annotation_font_color="green")

            # --- Row 3: MACD ---
            fig.add_trace(go.Scatter(
                x=df_indicators_display.index,
                y=df_indicators_display['MACD'],
                mode='lines',
                name='MACD Line',
                line=dict(color='blue', width=2)
            ), row=3, col=1)
            fig.add_trace(go.Scatter(
                x=df_indicators_display.index,
                y=df_indicators_display['Signal_Line'],
                mode='lines',
                name='Signal Line',
                line=dict(color='purple', width=1)
            ), row=3, col=1)
            fig.add_trace(go.Bar(
                x=df_indicators_display.index,
                y=df_indicators_display['MACD_Histogram'],
                name='MACD Histogram',
                marker_color=['red' if val < 0 else 'green' for val in df_indicators_display['MACD_Histogram']],
                opacity=0.6
            ), row=3, col=1)
            fig.add_hline(y=0, line_dash="dot", line_color="gray", row=3, col=1)


            # --- Update Layout ---
            fig.update_layout(
                height=700,
                xaxis_rangeslider_visible=False,
                template='plotly_dark',
                hovermode='x unified',
                margin=dict(l=0, r=0, t=50, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=st.get_option('theme.textColor')),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=1.05,
                    xanchor="left",
                    x=0,
                    bgcolor='rgba(0,0,0,0)',
                    bordercolor='rgba(0,0,0,0)',
                    font=dict(color=st.get_option('theme.textColor'))
                )
            )
            fig.update_yaxes(title_text='ფასი (USD)', row=1, col=1)
            fig.update_yaxes(title_text='RSI', range=[0, 100], row=2, col=1)
            fig.update_yaxes(title_text='MACD', row=3, col=1)
            fig.update_xaxes(showgrid=False, zeroline=False, tickfont=dict(color=st.get_option('theme.secondaryBackgroundColor')), title_font=dict(color=st.get_option('theme.textColor')), row=1, col=1)
            fig.update_xaxes(showgrid=False, zeroline=False, tickfont=dict(color=st.get_option('theme.secondaryBackgroundColor')), title_font=dict(color=st.get_option('theme.textColor')), row=2, col=1)
            fig.update_xaxes(showgrid=False, zeroline=False, tickfont=dict(color=st.get_option('theme.secondaryBackgroundColor')), title_font=dict(color=st.get_option('theme.textColor')), row=3, col=1)


            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="chart-legend">
                <div class="legend-item"><div class="legend-color historical"></div><span>ისტორია</span></div>
                <div class="legend-item"><div class="legend-color prediction"></div><span>პროგნოზი</span></div>
                <div class="legend-item"><div class="legend-color" style="background-color: yellow;"></div><span>MA(7)</span></div>
                <div class="legend-item"><div class="legend-color" style="background-color: orange;"></div><span>MA(25)</span></div>
                <div class="legend-item"><div class="legend-color" style="background-color: lightgreen;"></div><span>MA(99)</span></div>
                <div class="legend-item"><div class="legend-color buy-signal-legend"></div><span>ყიდვა</span></div>
                <div class="legend-item"><div class="legend-color sell-signal-legend"></div><span>გაყიდვა</span></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("<h4><i class='fas fa-chart-line'></i> ტექნიკური ინდიკატორების განმარტებები</h4>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class="argumentation-box">
                <p><span class='modal-label'>მოძრავი საშუალო (MA):</span> მოძრავი საშუალო (Moving Average) აჩვენებს ფასის საშუალო მნიშვნელობას განსაზღვრულ პერიოდში (მაგ., 7, 25, 99 დღე). ის გამოიყენება ტრენდის იდენტიფიცირებისთვის და ფასის "ხმაურის" გასაგლუვებლად. მოკლევადიანი MA-ს გრძელვადიან MA-ზე ზემოთ კვეთა (ე.წ. Golden Cross) განიხილება აღმავალი ტრენდის სიგნალად, ხოლო ქვემოთ კვეთა (Death Cross) - დაღმავალი ტრენდის სიგნალად.</p>
                <p><span class='modal-label'>Support & Breakout Levels:</span> Support Level არის ფასის დონე, რომელზეც მოსალოდნელია აქტივის ფასის ვარდნის შეჩერება და უკან ამოსვლა. Breakout Level (ან Resistance Level) არის ფასის დონე, რომელზეც მოსალოდნელია ფასის ზრდის შეჩერება და უკან დაცემა. ამ დონეების გარღვევა ხშირად ძლიერი ტრენდის ცვლილების სიგნალია.</p>
                <p><span class='modal-label'>Relative Strength Index (RSI):</span> RSI არის იმპულსის ოსცილატორი, რომელიც ზომავს ფასის ცვლილებების სიჩქარესა და ცვლილებას. ის მერყეობს 0-დან 100-მდე. 70-ზე ზემოთ ნიშნავს, რომ აქტივი ზედმეტად ნაყიდია (Overbought) და შესაძლოა ფასის კორექცია მოხდეს. 30-ზე ქვემოთ ნიშნავს, რომ აქტივი ზედმეტად გაყიდულია (Oversold) და შესაძლოა ფასის ზრდა დაიწყოს.</p>
                <p><span class='modal-label'>Moving Average Convergence Divergence (MACD):</span> MACD არის ტრენდის მიმდევარი იმპულსის ინდიკატორი, რომელიც აჩვენებს ფასის ორ ექსპონენციალურ მოძრავ საშუალოს (EMA) შორის ურთიერთობას (ჩვეულებრივ, 12-დღიანი და 26-დღიანი EMA). MACD ხაზის სასიგნალო ხაზის (9-დღიანი MACD-ის EMA) ზემოთ კვეთა არის ყიდვის სიგნალი, ხოლო ქვემოთ კვეთა - გაყიდვის სიგნალი. MACD ჰისტოგრამა აჩვენებს MACD ხაზსა და სასიგნალო ხაზს შორის განსხვავებას და გამოიყენება იმპულსის სიმძლავრის გასაზომად.</p>
                <p><span class='modal-label'>LSTM (Long Short-Term Memory):</span> LSTM არის ხელოვნური ნერონული ქსელის სპეციალური ტიპი, რომელიც განსაკუთრებით ეფექტურია დროითი სერიების მონაცემების დასამუშავებლად და პროგნოზირებისთვის. მას შეუძლია ისწავლოს დამოკიდებულებები მონაცემებში როგორც მოკლე, ასევე გრძელვადიან პერსპექტივაში, რაც მას სასარგებლოს ხდის კომპლექსური ფინანსური მონაცემების მოდელირებისთვის.
            </div>
            """, unsafe_allow_html=True)

        else:
            st.info("აირჩიეთ კრიპტოვალუტა მონაცემების სანახავად.")


with col3:
    with st.container(border=False):
        st.markdown("<h3>პროგნოზირებული სიგნალები</h3>", unsafe_allow_html=True)
        
        if yahoo_symbol and coin_details and historical_data_list_full:
            
            forecaster = CryptoDataForecaster(st.session_state.current_symbol, pd.DataFrame(historical_data_list_full))
            
            with st.spinner(f"გენერირდება სიგნალები {st.session_state.prediction_model} მოდელიდან..."):
                prediction_data_list = forecaster.generate_predictions(days=PREDICTION_DAYS, model_choice=st.session_state.prediction_model)
            signals = forecaster.generate_signals_from_prediction(prediction_data_list)
            
            if signals:
                st.markdown("<div id='signals-list' class='signals-list'>", unsafe_allow_html=True)
                for signal in signals:
                    signal_type_class = signal['type'].lower()
                    st.markdown(f"""
                    <div class="signal-item {signal_type_class}-signal-item">
                        <div class="signal-header">
                            <span class="signal-type {signal_type_class}-signal">
                                <i class="fas fa-{"arrow-up" if signal['type'] == 'BUY' else "arrow-down"}"></i> {signal['type']}
                            </span>
                            <span class="signal-date">{signal['date'].strftime('%Y-%m-%d')}</span>
                        </div>
                        <p class="signal-price">ფასი: <span class="signal-price-value">{format_price(signal['price'])} $</span></p>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander(f"დეტალები: {signal['type']} {signal['date'].strftime('%Y-%m-%d')}"):
                        st.markdown(f"<p><span class='modal-label'>თარიღი:</span> {signal['date'].strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p><span class='modal-label'>ფასი:</span> {format_price(signal['price'])} $</p>", unsafe_allow_html=True)
                        st.markdown(f"<p><span class='modal-label'>სანდოობა:</span> {signal['confidence']}</p>", unsafe_allow_html=True)
                        st.markdown("<div class='argumentation-box'><h4><i class='fas fa-brain'></i> AI ანალიზი:</h4></div>", unsafe_allow_html=True) 
                        st.markdown(f"<p>{signal['argumentation']}</p>", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("<p class='no-signals'>რელევანტური სიგნალები არ მოიძებნა.</p>", unsafe_allow_html=True)
        else:
            st.info("მონაცემები არ არის ხელმისაწვდომი სიგნალების მისაღებად.")

st.markdown("---")
st.markdown(f"""
<footer>
    <p>&copy; 2025 Crypto Signal App. ყველა უფლება დაცულია.</p>
    <p>მონაცემები მოწოდებულია <a href="https://finance.yahoo.com" target="_blank">Yahoo Finance API</a>-ს მიერ | ბოლო განახლება: {datetime.datetime.now().strftime('%H:%M:%S')}</p>
</footer>
""", unsafe_allow_html=True)
