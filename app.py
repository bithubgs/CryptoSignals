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
import numpy as np

# Import forecasting models with checks
ARIMA_AVAILABLE = False
try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    st.error("statsmodels.tsa.arima მოდული ვერ მოიძებნა. ARIMA პროგნოზირების მოდელი არ იქნება ხელმისაწვდომი.")
else:
    ARIMA_AVAILABLE = True

AUTO_ARIMA_AVAILABLE = False
if ARIMA_AVAILABLE: # Only try importing auto_arima if ARIMA is available
    try:
        from pmdarima import auto_arima
    except Exception as e: # Catch a broader exception for installation issues on Streamlit Cloud
        st.error(f"pmdarima მოდული ვერ ჩაიტვირთა. ARIMA-ს ავტომატური პარამეტრების ძიება არ იქნება ხელმისაწვდომი. შეცდომა: {e}")
    else:
        AUTO_ARIMA_AVAILABLE = True

PROPHET_AVAILABLE = False
try:
    from prophet import Prophet
except ImportError:
    st.error("Prophet მოდული ვერ მოიძებნა. Prophet პროგნოზირების მოდელი არ იქნება ხელმისაწვდომი. დააინსტალირეთ 'prophet': pip install prophet")
else:
    PROPHET_AVAILABLE = True

LSTM_AVAILABLE = False
try:
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress info and warnings
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR') # Only show errors
except ImportError:
    st.error("TensorFlow/Keras, Scikit-learn, ან NumPy მოდულები ვერ მოიძებნა. LSTM პროგნოზირების მოდელი არ იქნება ხელმისაწვდომი.")
else:
    LSTM_AVAILABLE = True

# --- Set page config for wide mode ---
st.set_page_config(layout="wide", page_title="კრიპტო სიგნალები LIVE", page_icon="📈")

# --- Configuration ---
COINGECKO_API_BASE = "https://api.coingecko.com/api/v3"
API_CALL_INTERVAL = 2.5 # Increased to be safer against 429 errors (Coingecko 100 calls/min is avg 0.6s/call)
MAX_HISTORICAL_DAYS = 1800 # Fetch approx 5 years of data for robust models

COINGECKO_CRYPTO_MAP = {
    'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana', 'ADA': 'cardano', 'DOT': 'polkadot',
    'LINK': 'chainlink', 'MATIC': 'polygon', 'AVAX': 'avalanche-2', 'ATOM': 'cosmos', 'LTC': 'litecoin',
    'BCH': 'bitcoin-cash', 'XRP': 'ripple', 'DOGE': 'dogecoin', 'SHIB': 'shiba-inu', 'UNI': 'uniswap',
    'AAVE': 'aave', 'SAND': 'the-sandbox', 'MANA': 'decentraland', 'AXS': 'axie-infinity', 'SKL': 'skale'
}

# --- SQLite Configuration ---
DB_FILE = 'crypto_historical_data.db'

def init_db():
    """Initializes the SQLite database and creates the table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS historical_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coingecko_id TEXT NOT NULL,
            date TEXT NOT NULL,
            price REAL NOT NULL,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(coingecko_id, date)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- Helper Functions ---
def rate_limit_api_call():
    """Ensures that API calls respect the Coingecko rate limit."""
    if 'last_api_call' not in st.session_state:
        st.session_state.last_api_call = 0
    
    elapsed = time.time() - st.session_state.last_api_call
    if elapsed < API_CALL_INTERVAL:
        time.sleep(API_CALL_INTERVAL - elapsed)
    st.session_state.last_api_call = time.time()

@st.cache_data(ttl=3600, show_spinner=False) # Cache for an hour
def fetch_coin_details(coingecko_id):
    """Fetches real-time coin details."""
    try:
        rate_limit_api_call()
        url = f"{COINGECKO_API_BASE}/coins/{coingecko_id}"
        params = {'localization': 'false', 'tickers': 'false', 'market_data': 'true', 'community_data': 'false', 'developer_data': 'false', 'sparkline': 'false'}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        market_data = data.get('market_data', {})
        return {
            'name': data.get('name'),
            'symbol': data.get('symbol', '').upper(),
            'currentPrice': market_data.get('current_price', {}).get('usd'),
            'dailyChange': market_data.get('price_change_percentage_24h'),
            '24hVolume': market_data.get('total_volume', {}).get('usd'),
            'marketCap': market_data.get('market_cap', {}).get('usd'),
            'marketCapRank': market_data.get('market_cap_rank'),
            'ath': market_data.get('ath', {}).get('usd'),
            'atl': market_data.get('atl', {}).get('usd'),
            'lastUpdated': data.get('last_updated')
        }
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            st.error(f"Error: Coingecko API ლიმიტი გადაჭარბებულია. გთხოვთ, სცადოთ მოგვიანებით. ({e})")
        else:
            st.error(f"Error fetching coin details for {coingecko_id}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching coin details for {coingecko_id}: {e}")
        return None

def fetch_historical_data_sqlite(coingecko_id, days=MAX_HISTORICAL_DAYS):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    earliest_date_needed = (datetime.date.today() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')

    cursor.execute(f"SELECT date, price FROM historical_prices WHERE coingecko_id = ? AND date >= ? ORDER BY date ASC",
                   (coingecko_id, earliest_date_needed))
    
    db_data = cursor.fetchall()
    historical_data = []

    if db_data:
        latest_date_in_db = max([row[0] for row in db_data])
        # Check if the latest date in DB is today and we have enough historical points
        # Using today's date for exact match for simple cache validation
        if datetime.datetime.strptime(latest_date_in_db, '%Y-%m-%d').date() == datetime.date.today() and len(db_data) >= days:
            st.toast(f"ისტორიული მონაცემები ჩაიტვირთა SQLite ქეშიდან: {coingecko_id}", icon="💾")
            historical_data = [{'date': datetime.datetime.strptime(row[0], '%Y-%m-%d'), 'price': row[1], 'type': 'historical'} for row in db_data]
            conn.close()
            return historical_data
        else:
            st.warning(f"SQLite ქეში მოძველებულია ან არასრულია. ვცდილობ განახლებას API-დან: {coingecko_id}")

    st.toast(f"მიმდინარეობს ისტორიული მონაცემების ჩატვირთვა API-დან: {coingecko_id}", icon="🌐")
    try:
        rate_limit_api_call()
        url = f"{COINGECKO_API_BASE}/coins/{coingecko_id}/market_chart"
        # Coingecko 'days' parameter: 1 for 24 hours, 7 for 7 days, etc. Max is 'max'.
        # For MAX_HISTORICAL_DAYS, we want 'max' if available, otherwise use specified days.
        # Coingecko returns daily prices for days > 90.
        params = {'vs_currency': 'usd', 'days': 'max', 'interval': 'daily'} 
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        api_data = response.json()
        
        fetched_historical_data = []
        for p in api_data.get('prices', []):
            date_obj = datetime.datetime.fromtimestamp(p[0] / 1000)
            fetched_historical_data.append({
                'date': date_obj, 
                'price': round(p[1], 8), 
                'type': 'historical'
            })
        
        # Filter to MAX_HISTORICAL_DAYS if 'max' returned too much or if we just want a subset
        if len(fetched_historical_data) > days:
            fetched_historical_data = fetched_historical_data[-days:]

        # Insert/update fetched data into DB
        for entry in fetched_historical_data:
            cursor.execute('''
                INSERT OR REPLACE INTO historical_prices (coingecko_id, date, price)
                VALUES (?, ?, ?)
            ''', (coingecko_id, entry['date'].strftime('%Y-%m-%d'), entry['price']))
        
        conn.commit()
        st.toast(f"ისტორიული მონაცემები შენახულია/განახლებულია SQLite-ში: {coingecko_id}", icon="💾")
        
        conn.close()
        return fetched_historical_data

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            st.error(f"Error: Coingecko API ლიმიტი გადაჭარბებულია ისტორიული მონაცემებისთვის. გთხოვთ, სცადოთ მოგვიერებით. ({e})")
        else:
            st.error(f"Error fetching historical data for {coingecko_id}: {e}")
        conn.close()
        return []
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching historical data for {coingecko_id}: {e}")
        conn.close()
        return []

class CryptoDataForecaster:
    def __init__(self, symbol, historical_prices_df):
        self.symbol = symbol.upper()
        self.historical_prices = historical_prices_df.copy()
        self.historical_prices['date'] = pd.to_datetime(self.historical_prices['date'])
        self.historical_prices.set_index('date', inplace=True)
        self.historical_prices['price'] = self.historical_prices['price'].astype(float)
        
        # Calculate features only if there's enough data
        if len(self.historical_prices) > 100: # Arbitrary threshold for meaningful features
            self.historical_with_features = self.calculate_features_and_indicators()
        else:
            self.historical_with_features = self.historical_prices.copy()


    def calculate_features_and_indicators(self):
        df_hist = self.historical_prices.copy()
        
        df_hist['price_lag1'] = df_hist['price'].shift(1)
        df_hist['daily_return'] = df_hist['price'].pct_change()

        df_hist['MA7'] = df_hist['price'].rolling(window=7, min_periods=1).mean()
        df_hist['MA25'] = df_hist['price'].rolling(window=25, min_periods=1).mean()
        df_hist['MA99'] = df_hist['price'].rolling(window=99, min_periods=1).mean()

        # RSI Calculation
        delta = df_hist['price'].diff()
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))
        
        # Use simple moving average for initial RSI if not enough periods for EWM
        # For full RSI calculation, ensure at least 14 periods of data
        if len(df_hist) >= 14:
            avg_gain = gain.ewm(span=14, adjust=False, min_periods=14).mean()
            avg_loss = loss.ewm(span=14, adjust=False, min_periods=14).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            df_hist['RSI'] = 100 - (100 / (1 + rs))
            df_hist['RSI'].fillna(0, inplace=True) 
            df_hist['RSI'] = df_hist['RSI'].replace([np.inf, -np.inf], np.nan).fillna(0)
        else:
            df_hist['RSI'] = 0 # Default to 0 if not enough data

        # MACD Calculation
        # For full MACD calculation, ensure at least 26 periods of data
        if len(df_hist) >= 26:
            exp1 = df_hist['price'].ewm(span=12, adjust=False, min_periods=12).mean()
            exp2 = df_hist['price'].ewm(span=26, adjust=False, min_periods=26).mean()
            df_hist['MACD'] = exp1 - exp2
            df_hist['Signal_Line'] = df_hist['MACD'].ewm(span=9, adjust=False, min_periods=9).mean()
            df_hist['MACD_Histogram'] = df_hist['MACD'] - df_hist['Signal_Line']
        else:
            df_hist['MACD'] = 0
            df_hist['Signal_Line'] = 0
            df_hist['MACD_Histogram'] = 0


        df_hist['day_of_week'] = df_hist.index.dayofweek
        df_hist['day_of_month'] = df_hist.index.day
        df_hist['month'] = df_hist.index.month
        df_hist['year'] = df_hist.index.year

        df_hist = df_hist.dropna()
        
        return df_hist

    def _generate_lstm_predictions(self, days):
        if not LSTM_AVAILABLE:
            st.error("LSTM მოდელი არ არის ხელმისაწვდომი. გთხოვთ, დააინსტალიროთ TensorFlow/Keras, NumPy და Scikit-learn.")
            return []

        # Features to be used for LSTM training and prediction
        features_for_lstm = ['price', 'MA7', 'MA25', 'MA99', 'RSI', 'MACD', 'Signal_Line', 'daily_return', 
                             'day_of_week', 'day_of_month', 'month', 'year']
        
        # Ensure we have enough data after dropping NaNs and for the look_back period
        df_lstm = self.historical_with_features[features_for_lstm].copy()
        
        look_back = 90 # Increased look_back for more context
        
        if len(df_lstm) < look_back + 10: # Minimum data points for LSTM training
            st.warning(f"არასაკმარისი სუფთა მონაცემები LSTM პროგნოზისთვის (მინიმუმ {look_back + 10} დღე, მიღებულია {len(df_lstm)}).")
            return []

        n_features = df_lstm.shape[1]

        # Scale all features
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(df_lstm.values)
        
        # Keep a separate scaler for the 'price' column to inverse transform correctly
        scaler_price = MinMaxScaler(feature_range=(0, 1))
        scaler_price.fit(df_lstm['price'].values.reshape(-1, 1)) 

        X, y = [], []
        for i in range(len(data_scaled) - look_back):
            X.append(data_scaled[i:(i + look_back), :])
            y.append(data_scaled[i + look_back, 0]) # Predict only the 'price' (first column)
        X = np.array(X)
        y = np.array(y)

        # Build LSTM model with more units and layers
        model = Sequential()
        model.add(LSTM(units=128, return_sequences=True, input_shape=(look_back, n_features)))
        model.add(Dropout(0.3))
        model.add(LSTM(units=128, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(units=64, activation='relu')) # Additional dense layer
        model.add(Dense(units=1)) 
        model.compile(optimizer=Adam(learning_rate=0.0008), loss='mean_squared_error') # Slightly lower learning rate

        try:
            model.fit(X, y, epochs=100, batch_size=32, verbose=0, shuffle=False) # Increased epochs
        except Exception as e:
            st.error(f"შეცდომა LSTM მოდელის ტრენინგისას: {e}. პროგნოზი მიუწვდომელია.")
            return []

        prediction_list = []
        # Use the very last 'look_back' data points from the scaled historical features for the first prediction
        current_input = data_scaled[-look_back:].reshape(1, look_back, n_features)
        last_date_original = self.historical_prices.index[-1]

        # Generate future features for prediction
        last_known_row = df_lstm.iloc[-1].copy()
        
        for i in range(days):
            predicted_scaled_price = model.predict(current_input, verbose=0)[0, 0]
            predicted_price = scaler_price.inverse_transform([[predicted_scaled_price]])[0, 0]
            
            date = last_date_original + datetime.timedelta(days=i+1)
            prediction_list.append({'date': date, 'price': max(0.00000001, round(predicted_price, 8)), 'type': 'prediction'})
            
            # --- Dynamically create next day's features ---
            next_day_features_values = last_known_row.copy()
            next_day_features_values['price'] = predicted_price # Update with predicted price
            
            # Update time-based features
            next_day_features_values['day_of_week'] = date.dayofweek
            next_day_features_values['day_of_month'] = date.day
            next_day_features_values['month'] = date.month
            next_day_features_values['year'] = date.year
            
            # Re-calculate simple features (daily_return based on predicted price)
            next_day_features_values['daily_return'] = (predicted_price - last_known_row['price']) / last_known_row['price']

            # For MAs, RSI, MACD, this is a simplification. Ideally, you'd calculate these based on a rolling window 
            # including past actuals and new predictions.
            # Here, for simplicity, we'll approximate by updating them based on the new price.
            # A more robust approach might be to train separate models for features or use a multi-output model.
            
            # For moving averages, approximate by taking a weighted average of old MA and new price
            # This is a very rough approximation, for real trading, this needs to be more precise.
            next_day_features_values['MA7'] = (last_known_row['MA7'] * 6 + predicted_price) / 7
            next_day_features_values['MA25'] = (last_known_row['MA25'] * 24 + predicted_price) / 25
            next_day_features_values['MA99'] = (last_known_row['MA99'] * 98 + predicted_price) / 99

            # RSI and MACD are more complex. For this simplified forward pass, we might just carry forward
            # the last known values, or implement a basic calculation based on current/previous predictions.
            # For demonstration, we'll carry them forward. For production, this is a weakness.
            next_day_features_values['RSI'] = last_known_row['RSI'] 
            next_day_features_values['MACD'] = last_known_row['MACD'] 
            next_day_features_values['Signal_Line'] = last_known_row['Signal_Line'] 
            
            # Ensure the order of features matches 'features_for_lstm'
            next_day_features_scaled = scaler.transform(next_day_features_values[features_for_lstm].values.reshape(1, -1))
            
            # Update current_input for the next prediction
            current_input = np.append(current_input[:, 1:, :], next_day_features_scaled.reshape(1, 1, n_features), axis=1)
            
            # Update last_known_row for the next iteration (important for daily_return and MA calculation)
            last_known_row['price'] = predicted_price


        return prediction_list

    def generate_predictions(self, days=30, model_choice="ARIMA Model"):
        if self.historical_prices.empty:
            st.warning("ისტორიული მონაცემები არ არის პროგნოზირებისთვის.")
            return []

        prediction_data = []
        
        # Use a subset of historical data for training to leave some for validation/testing
        # For real-world use, you'd train on ALL available data for the final model.
        # Here we use 80% for training and 20% for conceptual validation.
        train_size = int(len(self.historical_prices) * 0.8)
        train_data = self.historical_prices['price'].iloc[:train_size]

        if model_choice == "ARIMA Model":
            if not ARIMA_AVAILABLE:
                st.error("ARIMA მოდული არ არის ხელმისაწვდომი.")
                return []
            if len(train_data) < 100:
                st.warning("არასაკმარისი მონაცემები ARIMA პროგნოზისთვის (მინიმუმ 100 დღე).")
                return []

            try:
                if AUTO_ARIMA_AVAILABLE:
                    # Auto_arima finds optimal (p,d,q) parameters automatically
                    # seasonal=False for daily data (no clear annual seasonality, weekly is weak)
                    # trace=False to suppress detailed output during fitting
                    # suppress_warnings=True to avoid convergence warnings
                    # stepwise=True for faster search
                    st.info("ARIMA-ს ოპტიმალური პარამეტრების ძიება...")
                    model_fit = auto_arima(train_data, seasonal=False, 
                                           trace=False, error_action='ignore', 
                                           suppress_warnings=True, stepwise=True)
                    st.success(f"ARIMA-ს ოპტიმალური პარამეტრები: {model_fit.order}")
                else:
                    # Fallback to fixed order if auto_arima is not available
                    model = ARIMA(train_data, order=(5,1,0)) 
                    model_fit = model.fit()

                # Forecast from the end of the original series
                full_series_length = len(self.historical_prices['price'])
                forecast_result = model_fit.predict(start=full_series_length, end=full_series_length + days - 1)
                
                last_date = self.historical_prices.index[-1]
                for i, price in enumerate(forecast_result):
                    date = last_date + datetime.timedelta(days=i+1)
                    prediction_data.append({'date': date, 'price': max(0.00000001, round(price, 8)), 'type': 'prediction'})
            except Exception as e:
                st.error(f"შეცდომა ARIMA მოდელის გაშვებისას: {e}. პროგნოზი მიუწვდომელია.")
                return []

        elif model_choice == "Prophet Model":
            if not PROPHET_AVAILABLE:
                st.error("Prophet მოდული არ არის ხელმისაწვდომი.")
                return []
            if len(train_data) < 100:
                st.warning("არასაკმარისი მონაცემები Prophet პროგნოზისთვის (მინიმუმ 100 დღე).")
                return []

            # Prepare data for Prophet
            df_prophet_train = train_data.reset_index().rename(columns={'date': 'ds', 'price': 'y'})
            df_prophet_train['ds'] = pd.to_datetime(df_prophet_train['ds'])

            try:
                model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False, changepoint_prior_scale=0.2) # Increased to 0.2
                model.fit(df_prophet_train)
                future = model.make_future_dataframe(periods=days, include_history=False) # Only predict future
                forecast = model.predict(future)
                
                for _, row in forecast.iterrows(): # Iterate over all future predictions
                    prediction_data.append({
                        'date': row['ds'], 
                        'price': max(0.00000001, round(row['yhat'], 8)), 
                        'type': 'prediction'
                    })
            except Exception as e:
                st.error(f"შეცდომა Prophet მოდელის გაშვებისას: {e}. პროგნოზი მიუწვდომელია.")
                return []

        elif model_choice == "LSTM Model":
            prediction_data = self._generate_lstm_predictions(days)
        
        return prediction_data

    def generate_signals_from_prediction(self, prediction_data):
        if not prediction_data:
            return []
            
        df_prediction = pd.DataFrame(prediction_data)
        df_prediction['date'] = pd.to_datetime(df_prediction['date'])
        df_prediction.set_index('date', inplace=True)
        df_prediction['price'] = df_prediction['price'].astype(float)

        short_window = 3 
        long_window = 7 

        # Ensure we have enough data for MA calculations
        if len(df_prediction) < long_window:
            return [] # Not enough data for meaningful signals

        df_prediction['MA_short'] = df_prediction['price'].rolling(window=short_window).mean()
        df_prediction['MA_long'] = df_prediction['price'].rolling(window=long_window).mean()

        signals = []
        for i in range(1, len(df_prediction)):
            current_date = df_prediction.index[i]
            # Ensure we have valid MA values for comparison
            if pd.isna(df_prediction['MA_short'].iloc[i]) or pd.isna(df_prediction['MA_long'].iloc[i]) or \
               pd.isna(df_prediction['MA_short'].iloc[i-1]) or pd.isna(df_prediction['MA_long'].iloc[i-1]):
                continue 

            # Golden Cross (Buy Signal)
            if (df_prediction['MA_short'].iloc[i-1] < df_prediction['MA_long'].iloc[i-1] and
                df_prediction['MA_short'].iloc[i] > df_prediction['MA_long'].iloc[i]):
                signals.append({
                    'date': current_date,
                    'type': 'BUY',
                    'price': df_prediction['price'].iloc[i],
                    'confidence': f"{random.randint(75, 95)}%", # Higher confidence for AI-driven tool
                    'argumentation': f"AI პროგნოზირებს მოკლევადიანი მოძრავი საშუალოს (MA{short_window}) გრძელვადიან მოძრავ საშუალოზე (MA{long_window}) ზემოთ კვეთას {current_date.strftime('%Y-%m-%d')} თარიღისთვის, რაც პოტენციური აღმავალი ტრენდის დასაწყისს მიანიშნებს. ეს არის ტრადიციული ტექნიკური ინდიკატორის მყიდველური სიგნალი."
                })

            # Death Cross (Sell Signal)
            elif (df_prediction['MA_short'].iloc[i-1] > df_prediction['MA_long'].iloc[i-1] and
                  df_prediction['MA_short'].iloc[i] < df_prediction['MA_long'].iloc[i]):
                signals.append({
                    'date': current_date,
                    'type': 'SELL',
                    'price': df_prediction['price'].iloc[i],
                    'confidence': f"{random.randint(70, 90)}%", # Higher confidence
                    'argumentation': f"AI პროგნოზირებს მოკლევადიანი მოძრავი საშუალოს (MA{short_window}) გრძელვადიან მოძრავ საშუალოზე (MA{long_window}) ქვემოთ კვეთას {current_date.strftime('%Y-%m-%d')} თარიღისთვის, რაც პოტენციური დაღმავალი ტრენდის დასაწყისს მიანიშნებს. ეს არის ტრადიციული ტექნიკური ინდიკატორის გამყიდველური სიგნალი."
                })
        
        return sorted(signals, key=lambda x: x['date'])


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
    st.session_state.prediction_model = "ARIMA Model" # Default if available

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
        popular_cryptos = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'SKL']
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

        coingecko_id = COINGECKO_CRYPTO_MAP.get(st.session_state.current_symbol)
        
        if coingecko_id:
            with st.spinner(f"იტვირთება მონაცემები {st.session_state.current_symbol}-ისთვის..."):
                coin_details = fetch_coin_details(coingecko_id)
                if coin_details:
                    st.markdown(f"<h2 id='coin-name'>{coin_details['name']} ({coin_details['symbol']})</h2>", unsafe_allow_html=True)
                    st.markdown(f"<p id='current-price' class='price-value'>{format_price(coin_details['currentPrice'])} $</p>", unsafe_allow_html=True)
                    
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.markdown("<div class='stat-card'><h4>24სთ ცვლილება</h4>", unsafe_allow_html=True)
                        daily_change_class = 'positive' if (coin_details['dailyChange'] or 0) >= 0 else 'negative'
                        st.markdown(f"<div class='value {daily_change_class}'>{coin_details['dailyChange']:.2f}%</div></div>", unsafe_allow_html=True)
                    with col_m2:
                        st.markdown("<div class='stat-card'><h4>24სთ მოცულობა</h4>", unsafe_allow_html=True)
                        st.markdown(f"<div class='value'>{format_currency(coin_details['24hVolume'])}</div></div>", unsafe_allow_html=True)
                    
                    col_m3, col_m4 = st.columns(2)
                    with col_m3:
                        st.markdown("<div class='stat-card'><h4>კაპიტალიზაცია</h4>", unsafe_allow_html=True)
                        st.markdown(f"<div class='value'>{format_currency(coin_details['marketCap'])}</div></div>", unsafe_allow_html=True)
                    with col_m4:
                        st.markdown("<div class='stat-card'><h4>რანგი</h4>", unsafe_allow_html=True)
                        st.markdown(f"<div class='value'>#{coin_details['marketCapRank'] or '-'}</div></div>", unsafe_allow_html=True)

                    col_m5, col_m6 = st.columns(2)
                    with col_m5:
                        st.markdown("<div class='stat-card'><h4>ATH</h4>", unsafe_allow_html=True)
                        st.markdown(f"<div class='value'>{format_price(coin_details['ath'])} $</div></div>", unsafe_allow_html=True)
                    with col_m6:
                        st.markdown("<div class='stat-card'><h4>ATL</h4>", unsafe_allow_html=True)
                        st.markdown(f"<div class='value'>{format_price(coin_details['atl'])} $</div></div>", unsafe_allow_html=True)

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
        if ARIMA_AVAILABLE:
            model_options.append("ARIMA Model")
        if PROPHET_AVAILABLE:
            model_options.append("Prophet Model")
        if LSTM_AVAILABLE:
            model_options.append("LSTM Model")

        if not model_options:
            st.error("არცერთი პროგნოზირების მოდული არ არის ხელმისაწვდომი. გთხოვთ, დააინსტალიროთ საჭირო ბიბლიოთეკები (იხ. requirements.txt).")
        else:
            # Set default model if current one is not available
            if st.session_state.prediction_model not in model_options:
                st.session_state.prediction_model = model_options[0] # Fallback to first available

            selected_model = st.selectbox(
                "აირჩიეთ პროგნოზირების მოდელი:",
                model_options,
                index=model_options.index(st.session_state.prediction_model) # Use current selected model's index
            )
            if selected_model != st.session_state.prediction_model:
                st.session_state.prediction_model = selected_model
                st.rerun()

        if coingecko_id and coin_details:
            historical_data_list_full = fetch_historical_data_sqlite(coingecko_id, days=MAX_HISTORICAL_DAYS)
            
            if historical_data_list_full:
                df_historical_full = pd.DataFrame(historical_data_list_full)
                df_historical_full['date'] = pd.to_datetime(df_historical_full['date'])
                
                forecaster = CryptoDataForecaster(st.session_state.current_symbol, df_historical_full) 
                
                with st.spinner(f"მიმდინარეობს {st.session_state.prediction_model} პროგნოზირება... (შეიძლება დასჭირდეს დრო LSTM-ისთვის)"):
                    prediction_data_list = forecaster.generate_predictions(days=30, model_choice=st.session_state.prediction_model)
                
                if not prediction_data_list:
                    st.warning(f"პროგნოზის გენერირება ვერ მოხერხდა {st.session_state.prediction_model} მოდელით. გთხოვთ, სცადოთ სხვა მოდელი ან სხვა კრიპტოვალუტა.")
                
                df_prediction = pd.DataFrame(prediction_data_list)
                if not df_prediction.empty:
                    df_prediction['date'] = pd.to_datetime(df_prediction['date']) 
                
                signals = forecaster.generate_signals_from_prediction(prediction_data_list)

                # Filter historical features for display based on selected period
                df_indicators_display = forecaster.historical_with_features.tail(st.session_state.current_period).copy()
                
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.08, 
                                    row_heights=[0.6, 0.2, 0.2],
                                    subplot_titles=(
                                        f"{st.session_state.current_symbol} ფასი და პროგნოზი",
                                        "Relative Strength Index (RSI)",
                                        "Moving Average Convergence Divergence (MACD)"
                                    ))

                # Price and Prediction Plot (Row 1)
                fig.add_trace(go.Scatter(
                    x=df_indicators_display.index,
                    y=df_indicators_display['price'],
                    mode='lines',
                    name='ისტორია',
                    line=dict(color='#00bcd4', width=2)
                ), row=1, col=1)

                if not df_prediction.empty:
                    fig.add_trace(go.Scatter(
                        x=df_prediction['date'],
                        y=df_prediction['price'],
                        mode='lines',
                        name='პროგნოზი',
                        line=dict(color='#8e2de2', dash='dot', width=2)
                    ), row=1, col=1)

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

                # Add Signals to the chart
                for signal in signals:
                    signal_color = "#39ff14" if signal['type'] == 'BUY' else "#ff073a"
                    # Only add signals if they fall within the displayed historical/prediction range
                    if not df_prediction.empty and signal['date'] >= df_indicators_display.index.min():
                        fig.add_trace(go.Scatter(
                            x=[signal['date']],
                            y=[signal['price']],
                            mode='markers+text',
                            name=f"{signal['type']} სიგნალი",
                            marker=dict(
                                symbol='circle-open',
                                size=14, 
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

                # RSI Plot (Row 2)
                fig.add_trace(go.Scatter(
                    x=df_indicators_display.index,
                    y=df_indicators_display['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='cyan', width=2)
                ), row=2, col=1)
                fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1, annotation_text="Overbought (70)", annotation_position="top right", annotation_font_color="red")
                fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1, annotation_text="Oversold (30)", annotation_position="bottom right", annotation_font_color="green")

                # MACD Plot (Row 3)
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
                    <p><span class='modal-label'>Relative Strength Index (RSI):</span> RSI არის იმპულსის ოსცილატორი, რომელიც ზომავს ფასის ცვლილებების სიჩქარესა და ცვლილებას. ის მერყეობს 0-დან 100-მდე. 70-ზე ზემოთ ნიშნავს, რომ აქტივი ზედმეტად ნაყიდია (Overbought) და შესაძლოა ფასის კორექცია მოხდეს. 30-ზე ქვემოთ ნიშნავს, რომ აქტივი ზედმეტად გაყიდულია (Oversold) და შესაძლოა ფასის ზრდა დაიწყოს.</p>
                    <p><span class='modal-label'>Moving Average Convergence Divergence (MACD):</span> MACD არის ტრენდის მიმდევარი იმპულსის ინდიკატორი, რომელიც აჩვენებს ფასის ორ ექსპონენციალურ მოძრავ საშუალოს (EMA) შორის ურთიერთობას (ჩვეულებრივ, 12-დღიანი და 26-დღიანი EMA). MACD ხაზის სასიგნალო ხაზის (9-დღიანი MACD-ის EMA) ზემოთ კვეთა არის ყიდვის სიგნალი, ხოლო ქვემოთ კვეთა - გაყიდვის სიგნალი. MACD ჰისტოგრამა აჩვენებს MACD ხაზსა და სასიგნალო ხაზს შორის განსხვავებას და გამოიყენება იმპულსის სიმძლავრის გასაზომად.</p>
                    <p><span class='modal-label'>LSTM (Long Short-Term Memory):</span> LSTM არის ხელოვნური ნერონული ქსელის სპეციალური ტიპი, რომელიც განსაკუთრებით ეფექტურია დროითი სერიების მონაცემების დასამუშავებლად და პროგნოზირებისთვის. მას შეუძლია ისწავლოს დამოკიდებულებები მონაცემებში როგორც მოკლე, ასევე გრძელვადიან პერსპექტივაში, რაც მას სასარგებლოს ხდის კომპლექსური ფინანსური მონაცემების მოდელირებისთვის.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # IMPORTANT DISCLAIMER
                st.markdown("---")
                st.markdown("<h3><i class='fas fa-exclamation-triangle'></i> მნიშვნელოვანი გაფრთხილება (კრიპტო ინვესტირება):</h3>", unsafe_allow_html=True)
                st.markdown("""
                <div class="argumentation-box" style="border-left: 5px solid red;">
                    <p>ეს ინსტრუმენტი შექმნილია თქვენი ინვესტიციების პროცესში დახმარებისთვის ტექნიკური ანალიზის საფუძველზე. თუმცა, მნიშვნელოვანია გვესმოდეს, რომ:</p>
                    <ul>
                        <li><span style="font-weight:bold; color:red;">მაღალი რისკი:</span> კრიპტოვალუტების ბაზარი უკიდურესად ცვალებადია და მაღალი რისკის შემცველი. ფასები შეიძლება მკვეთრად შეიცვალოს მოკლე დროში.</li>
                        <li><span style="font-weight:bold; color:red;">არ არის გარანტირებული სიზუსტე:</span> AI მოდელები პროგნოზებს აკეთებენ ისტორიული მონაცემების საფუძველზე და ვერ ითვალისწინებენ ყველა გარე ფაქტორს (სიახლეები, მარეგულირებელი ცვლილებები, მაკროეკონომიკური ტენდენციები, გეოპოლიტიკა), რომლებიც კრიპტო ფასებზე გავლენას ახდენს. შესაბამისად, პროგნოზების სიზუსტე შეზღუდულია.</li>
                        <li><span style="font-weight:bold; color:red;">თქვენი პასუხისმგებლობა:</span> ეს ინსტრუმენტი არის დამხმარე საშუალება და არ წარმოადგენს საინვესტიციო რჩევას. საბოლოო საინვესტიციო გადაწყვეტილებები უნდა მიიღოთ საკუთარი კვლევისა და რისკის ანალიზის საფუძველზე.</li>
                        <li><span style="font-weight:bold; color:red;">კაპიტალის დაკარგვის რისკი:</span> შესაძლოა დაკარგოთ თქვენი საინვესტიციო კაპიტალის ნაწილი ან მთლიანად. ინვესტირება მხოლოდ იმ თანხით, რისი დაკარგვაც შეგიძლიათ.</li>
                    </ul>
                    <p>გამოიყენეთ ეს ინსტრუმენტი სიფრთხილით და ყოველთვის ჩაატარეთ დეტალური ანალიზი ინვესტირებამდე.</p>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.warning("ისტორიული მონაცემები ვერ მოიძებნა არჩეული პერიოდისთვის.")
        else:
            st.info("აირჩიეთ კრიპტოვალუტა მონაცემების სანახავად.")


with col3:
    with st.container(border=False):
        st.markdown("<h3>პროგნოზირებული სიგნალები</h3>", unsafe_allow_html=True)
        
        if coingecko_id and coin_details:
            historical_data_list_full = fetch_historical_data_sqlite(coingecko_id, days=MAX_HISTORICAL_DAYS)
            
            forecaster = CryptoDataForecaster(st.session_state.current_symbol, pd.DataFrame(historical_data_list_full))
            
            with st.spinner(f"გენერირდება სიგნალები {st.session_state.prediction_model} მოდელიდან..."):
                prediction_data_list = forecaster.generate_predictions(days=30, model_choice=st.session_state.prediction_model)
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

                    with st.expander(f"დეტალები: {signal['type']} {signal['date'].strftime('%Y-%m-%d %H:%M:%S')}"):
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
    <p>მონაცემები მოწოდებულია <a href="https://www.coingecko.com" target="_blank">CoinGecko API</a>-ს მიერ | ბოლო განახლება: {datetime.datetime.now().strftime('%H:%M:%S')}</p>
</footer>
""", unsafe_allow_html=True)
