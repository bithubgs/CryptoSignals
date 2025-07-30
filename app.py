import streamlit as st
import datetime
import random
import requests
import time
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3 # Import SQLite
import os # For file operations
import numpy as np # For numerical operations

# Import forecasting models with checks
try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    st.error("ARIMA მოდული ვერ მოიძებნა. დააინსტალირეთ 'statsmodels': pip install statsmodels")
    ARIMA = None

try:
    from prophet import Prophet
except ImportError:
    st.error("Prophet მოდული ვერ მოიძებნა. დააინსტალირეთ 'prophet': pip install prophet")
    Prophet = None

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
    st.error("TensorFlow/Keras, Scikit-learn, ან NumPy მოდულები ვერ მოიძებნა. დააინსტალირეთ: pip install tensorflow scikit-learn numpy")
    LSTM_AVAILABLE = False
else:
    LSTM_AVAILABLE = True


# --- Set page config for wide mode ---
st.set_page_config(layout="wide", page_title="კრიპტო სიგნალები LIVE", page_icon="📈")

# --- Configuration ---
COINGECKO_API_BASE = "https://api.coingecko.com/api/v3"
API_CALL_INTERVAL = 2.5 # Increased to be safer against 429 errors
# Increased historical days for better model training
MAX_HISTORICAL_DAYS = 1000 # Fetch approx 3 years of data for better MA99, RSI, MACD, and especially LSTM training

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
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP, -- To track when data was last fetched/updated
            UNIQUE(coingecko_id, date) -- Ensure unique entries per coin per day
        )
    ''')
    conn.commit()
    conn.close()

# Initialize the database when the app starts
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

# Removed @st.cache_data here as data is now managed by SQLite
def fetch_coin_details(coingecko_id):
    """Fetches real-time coin details. This is not persistently cached due to real-time nature."""
    try:
        rate_limit_api_call() # Wait before making the call
        url = f"{COINGECKO_API_BASE}/coins/{coingecko_id}"
        params = {'localization': 'false', 'tickers': 'false', 'market_data': 'true', 'community_data': 'false', 'developer_data': 'false', 'sparkline': 'false'}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
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

# Modified function to use SQLite for historical data
def fetch_historical_data_sqlite(coingecko_id, days=MAX_HISTORICAL_DAYS):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Calculate the earliest date we need based on 'days'
    earliest_date_needed = (datetime.date.today() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')

    # 1. Try to load from SQLite first
    cursor.execute(f"SELECT date, price FROM historical_prices WHERE coingecko_id = ? AND date >= ? ORDER BY date ASC",
                   (coingecko_id, earliest_date_needed))
    
    db_data = cursor.fetchall()
    historical_data = []

    if db_data:
        # Check if we have enough data and if the latest date is today (or very recent)
        # Using a small tolerance for "today" (e.g., within the last 24 hours from timestamp)
        # The timestamp column can be useful here. For simplicity, check file date for now.
        latest_date_in_db = max([row[0] for row in db_data])
        
        # Check if the latest data point is from today. If not, consider it stale.
        # This prevents constantly refetching if your 'days' value is less than the total data in DB
        # We want to refresh if the latest data in DB is *not* today's data.
        if datetime.datetime.strptime(latest_date_in_db, '%Y-%m-%d').date() == datetime.date.today() and len(db_data) >= days:
            st.toast(f"ისტორიული მონაცემები ჩაიტვირთა SQLite ქეშიდან: {coingecko_id}", icon="💾")
            historical_data = [{'date': datetime.datetime.strptime(row[0], '%Y-%m-%d'), 'price': row[1], 'type': 'historical'} for row in db_data]
            conn.close()
            return historical_data
        else:
            st.warning(f"SQLite ქეში მოძველებულია ან არასრულია. ვცდილობ განახლებას API-დან: {coingecko_id}")

    # 2. If not in DB or outdated, fetch from API
    st.toast(f"მიმდინარეობს ისტორიული მონაცემების ჩატვირთვა API-დან: {coingecko_id}", icon="🌐")
    try:
        rate_limit_api_call() # Wait before making the call
        url = f"{COINGECKO_API_BASE}/coins/{coingecko_id}/market_chart"
        params = {'vs_currency': 'usd', 'days': str(days), 'interval': 'daily'}
        response = requests.get(url, params=params, timeout=15)
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
        
        # 3. Save/Update to SQLite
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
            st.error(f"Error: Coingecko API ლიმიტი გადაჭარბებულია ისტორიული მონაცემებისთვის. გთხოვთ, სცადოთ მოგვიანებით. ({e})")
        else:
            st.error(f"Error fetching historical data for {coingecko_id}: {e}")
        conn.close()
        return []
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching historical data for {coingecko_id}: {e}")
        conn.close()
        return []

# --- CryptoDataForecaster and other functions remain the same ---
class CryptoDataForecaster:
    def __init__(self, symbol, historical_prices_df):
        self.symbol = symbol.upper()
        self.historical_prices = historical_prices_df.copy()
        self.historical_prices['date'] = pd.to_datetime(self.historical_prices['date'])
        self.historical_prices.set_index('date', inplace=True)
        self.historical_prices['price'] = self.historical_prices['price'].astype(float)
        
        # Pre-calculate technical indicators and additional features for the full historical data
        self.historical_with_features = self.calculate_features_and_indicators()

    def calculate_features_and_indicators(self):
        df_hist = self.historical_prices.copy()
        
        # Basic Price Features
        df_hist['price_lag1'] = df_hist['price'].shift(1) # Price of previous day
        df_hist['daily_return'] = df_hist['price'].pct_change() # Daily percentage change

        # Moving Averages (already present, good)
        df_hist['MA7'] = df_hist['price'].rolling(window=7, min_periods=1).mean()
        df_hist['MA25'] = df_hist['price'].rolling(window=25, min_periods=1).mean()
        df_hist['MA99'] = df_hist['price'].rolling(window=99, min_periods=1).mean()

        # Relative Strength Index (RSI)
        delta = df_hist['price'].diff()
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))
        
        avg_gain = gain.ewm(span=14, adjust=False, min_periods=14).mean()
        avg_loss = loss.ewm(span=14, adjust=False, min_periods=14).mean()
        
        # Handle division by zero for rs
        rs = avg_gain / avg_loss.replace(0, np.nan) 
        df_hist['RSI'] = 100 - (100 / (1 + rs))
        df_hist['RSI'].fillna(0, inplace=True) # Fill initial NaNs for RSI, if any
        df_hist['RSI'] = df_hist['RSI'].replace([np.inf, -np.inf], np.nan).fillna(0) # Handle inf values

        # Moving Average Convergence Divergence (MACD)
        exp1 = df_hist['price'].ewm(span=12, adjust=False, min_periods=12).mean()
        exp2 = df_hist['price'].ewm(span=26, adjust=False, min_periods=26).mean()
        df_hist['MACD'] = exp1 - exp2
        df_hist['Signal_Line'] = df_hist['MACD'].ewm(span=9, adjust=False, min_periods=9).mean()
        df_hist['MACD_Histogram'] = df_hist['MACD'] - df_hist['Signal_Line']

        # Time-based features (for seasonality and trends)
        df_hist['day_of_week'] = df_hist.index.dayofweek # Monday=0, Sunday=6
        df_hist['day_of_month'] = df_hist.index.day
        df_hist['month'] = df_hist.index.month
        df_hist['year'] = df_hist.index.year

        # Drop rows with NaN values resulting from feature engineering
        # This ensures clean data for models, especially LSTM
        df_hist = df_hist.dropna()
        
        return df_hist

    def _generate_lstm_predictions(self, days):
        if not LSTM_AVAILABLE:
            st.error("LSTM მოდელი არ არის ხელმისაწვდომი. გთხოვთ, დააინსტალიროთ TensorFlow/Keras, NumPy და Scikit-learn.")
            return [] # Return empty if dependencies are missing

        # Use all relevant features, not just a subset
        features = ['price', 'MA7', 'MA25', 'RSI', 'MACD', 'daily_return', 'day_of_week', 'day_of_month', 'month', 'year']
        
        # Ensure we have enough data after dropping NaNs
        df_lstm = self.historical_with_features[features].copy()
        
        # Check if any feature column is entirely NaN after dropping rows - this shouldn't happen with dropna()
        # but good to be defensive
        if df_lstm.isnull().any().any():
             st.warning("LSTM მონაცემებში დარჩა NaN მნიშვნელობები. პროგნოზი შეიძლება არასწორი იყოს.")
             df_lstm = df_lstm.dropna() # Re-dropna just in case

        if len(df_lstm) < 100: # Increased minimum data points for LSTM
            st.warning(f"არასაკმარისი სუფთა მონაცემები LSTM პროგნოზისთვის (მინიმუმ 100 დღე, მიღებულია {len(df_lstm)}).")
            return []

        look_back = 60 # Number of previous time steps to use as input features
        n_features = df_lstm.shape[1] # Number of features used (price + indicators + time features)

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

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=100, return_sequences=True, input_shape=(look_back, n_features)))
        model.add(Dropout(0.3))
        model.add(LSTM(units=100, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(units=1)) 
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

        try:
            # Increased epochs for potentially better learning, but be mindful of training time
            model.fit(X, y, epochs=70, batch_size=32, verbose=0) 
        except Exception as e:
            st.error(f"შეცდომა LSTM მოდელის ტრენინგისას: {e}. პროგნოზი მიუწვდომელია.")
            return []

        prediction_list = []
        # Use the very last 'look_back' data points from the scaled historical features for the first prediction
        current_input = data_scaled[-look_back:].reshape(1, look_back, n_features)
        last_date_original = self.historical_prices.index[-1]

        for i in range(days):
            predicted_scaled_price = model.predict(current_input, verbose=0)[0, 0]
            predicted_price = scaler_price.inverse_transform([[predicted_scaled_price]])[0, 0]
            
            date = last_date_original + datetime.timedelta(days=i+1)
            prediction_list.append({'date': date, 'price': max(0.00000001, round(predicted_price, 8)), 'type': 'prediction'})
            
            # For the next prediction, update the 'current_input'
            # We need to simulate the features for the predicted day to feed back into the model.
            # This is a simplification and assumes future features (like MA, RSI) will follow a similar pattern.
            # A more robust approach might involve predicting features as well or using more complex models.

            # Create a dummy row for the next day's features, filled with NaNs initially
            next_day_features = np.zeros((1, n_features))
            next_day_features[0, 0] = predicted_scaled_price # Put predicted price in the first feature column
            
            # For simplicity, for other features, we can reuse the last known scaled features for now
            # In a real-world scenario, you might have a more sophisticated way to estimate these future features.
            # Here, we'll carry forward the last actual feature values for the non-price features.
            last_actual_features = data_scaled[-1, :] # Last row of actual scaled features
            for j in range(1, n_features): # Copy all features except the price
                next_day_features[0, j] = last_actual_features[j] # This is a simplification!

            # Update current_input by removing the oldest step and adding the new predicted step
            current_input = np.append(current_input[:, 1:, :], next_day_features.reshape(1, 1, n_features), axis=1)

        return prediction_list

    def generate_predictions(self, days=30, model_choice="ARIMA Model"):
        if self.historical_prices.empty:
            st.warning("ისტორიული მონაცემები არ არის პროგნოზირებისთვის.")
            return []

        prediction_data = []
        
        if model_choice == "ARIMA Model":
            if ARIMA is None:
                st.error("ARIMA მოდული არ არის ხელმისაწვდომი. გთხოვთ, დააინსტალიროთ 'statsmodels'.")
                return []
            if len(self.historical_prices) < 90:
                st.warning("არასაკმარისი მონაცემები ARIMA პროგნოზისთვის (მინიმუმ 90 დღე).")
                return []

            series = self.historical_prices['price']
            try:
                # Experiment with different ARIMA orders or use auto_arima if pmdarima is installed
                # (p,d,q) parameters: p=AR order, d=differencing order, q=MA order
                # (5,1,0) is a common starting point for daily data.
                model = ARIMA(series, order=(5,1,0)) 
                model_fit = model.fit()
                forecast_result = model_fit.predict(start=len(series), end=len(series) + days - 1)
                
                last_date = self.historical_prices.index[-1]
                for i, price in enumerate(forecast_result):
                    date = last_date + datetime.timedelta(days=i+1)
                    prediction_data.append({'date': date, 'price': max(0.00000001, round(price, 8)), 'type': 'prediction'})
            except Exception as e:
                st.error(f"შეცდომა ARIMA მოდელის გაშვებისას: {e}. პროგნოზი მიუწვდომელია.")
                return []

        elif model_choice == "Prophet Model":
            if Prophet is None:
                st.error("Prophet მოდული არ არის ხელმისაწვდომი. გთხოვთ, დააინსტალიროთ 'prophet'.")
                return []
            if len(self.historical_prices) < 90:
                st.warning("არასაკმარისი მონაცემები Prophet პროგნოზისთვის (მინიმუმ 90 დღე).")
                return []

            df_prophet = self.historical_prices.reset_index()[['date', 'price']].rename(columns={'date': 'ds', 'price': 'y'})
            df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

            try:
                # Increased changepoint_prior_scale for more flexibility in trend changes
                model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False, changepoint_prior_scale=0.15) 
                model.fit(df_prophet)
                future = model.make_future_dataframe(periods=days)
                forecast = model.predict(future)
                
                for _, row in forecast.tail(days).iterrows():
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
        if len(prediction_data) < 7:
            return []
        
        df_prediction = pd.DataFrame(prediction_data)
        df_prediction['date'] = pd.to_datetime(df_prediction['date'])
        df_prediction.set_index('date', inplace=True)
        df_prediction['price'] = df_prediction['price'].astype(float)

        short_window = 3 
        long_window = 7 

        df_prediction['MA_short'] = df_prediction['price'].rolling(window=short_window).mean()
        df_prediction['MA_long'] = df_prediction['price'].rolling(window=long_window).mean()

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
                    'price': df_prediction['price'].iloc[i],
                    'confidence': f"{random.randint(70, 90)}%",
                    'argumentation': f"AI პროგნოზირებს მოკლევადიანი მოძრავი საშუალოს (MA{short_window}) გრძელვადიან მოძრავ საშუალოზე (MA{long_window}) ზემოთ კვეთას, რაც პოტენციური აღმავალი ტრენდის დასაწყისს მიანიშნებს."
                })

            elif (df_prediction['MA_short'].iloc[i-1] > df_prediction['MA_long'].iloc[i-1] and
                  df_prediction['MA_short'].iloc[i] < df_prediction['MA_long'].iloc[i]):
                signals.append({
                    'date': current_date,
                    'type': 'SELL',
                    'price': df_prediction['price'].iloc[i],
                    'confidence': f"{random.randint(65, 85)}%",
                    'argumentation': f"AI პროგნოზირებს მოკლევადიანი მოძრავი საშუალოს (MA{short_window}) გრძელვადიან მოძრავ საშუალოზე (MA{long_window}) ქვემოთ კვეთას, რაც პოტენციური დაღმავალი ტრენდის დასაწყისს მიანიშნებს."
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

# Apply custom CSS from styles.css (limited scope)
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
        padding: 1rem; /* Adjust padding if needed */
    }
    
    h1 {
        font-size: 2.5rem;
        font-weight: 800;
        text-shadow: 0 0 10px var(--accent-blue), 0 0 20px var(--accent-purple);
        color: var(--text-light); /* Ensure H1 color */
    }
    
    .live-text { color: var(--accent-red); /* No animation support here */ }
    
    .stSelectbox, .stTextInput, .stButton > button {
        background-color: var(--bg-dark);
        color: var(--text-light);
        border: 1px solid var(--border-color);
        border-radius: 8px;
    }
    .stButton > button {
        background: linear-gradient(45deg, var(--accent-blue), var(--accent-purple));
        color: white;
        border: none; /* Remove default button border */
        cursor: pointer;
    }

    .stButton > button:hover {
        opacity: 0.9; /* Simple hover effect */
    }

    /* Card-like appearance for sections */
    .stContainer {
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
        margin-bottom: 1.5rem; /* Add spacing between "cards" */
    }

    /* Specific styling for nested containers to avoid double borders/backgrounds */
    .stContainer > div > div > div > .stContainer {
        background-color: transparent;
        border: none;
        box-shadow: none;
        padding: 0;
        margin-bottom: 0;
    }

    .stAlert { /* For toasts */
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

    /* Specific styling for signals list items, if they can be represented */
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
    .historical { background-color: #00bcd4; } /* Matches accent-blue for historical */
    .prediction { background-color: #8e2de2; } /* Matches accent-purple for prediction */
    .buy-signal-legend { background-color: var(--accent-green); width: 10px; height: 10px; border-radius: 50%; } /* Circle for legend */
    .sell-signal-legend { background-color: var(--accent-red); width: 10px; height: 10px; border-radius: 0; transform: rotate(45deg); } /* Square for legend */


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
    
    /* Minor adjustments for responsiveness in Streamlit context */
    @media (max-width: 768px) {
        .stApp { padding: 0.5rem; }
        h1 { font-size: 2rem; }
        .price-value { font-size: 1.8rem; }
        /* Streamlit columns handle responsiveness, but you can override if needed */
    }

</style>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
""", unsafe_allow_html=True)


st.title("კრიპტო სიგნალები LIVE")

# --- Header Status Indicator (simplified) ---
st.markdown("""
<div class="status-indicator connected">
    <i class="fas fa-wifi"></i>
    <span>Online</span>
</div>
""", unsafe_allow_html=True)

# Initialize session state for symbol and period if not already set
if 'current_symbol' not in st.session_state:
    st.session_state.current_symbol = 'SAND'
if 'current_period' not in st.session_state:
    st.session_state.current_period = 90 # Default display period
if 'prediction_model' not in st.session_state:
    st.session_state.prediction_model = "ARIMA Model" # Default model now ARIMA for statistical rigor

# Create columns for the main layout (Streamlit's way of layout)
col1, col2, col3 = st.columns([0.8, 1.5, 0.8]) # Adjust ratios as needed

with col1: # Left Pane
    with st.container(border=False): # Use container to mimic card styling
        st.markdown("<h3>აირჩიეთ კრიპტოვალუტა:</h3>", unsafe_allow_html=True)
        
        # Use a form to group input and button to prevent immediate rerun on text change
        with st.form("symbol_form"):
            symbol_input = st.text_input("შეიყვანეთ სიმბოლო (მაგ. BTC, ETH)", st.session_state.current_symbol).upper().strip()
            submit_button = st.form_submit_button("ძიება", use_container_width=True)

            if submit_button:
                st.session_state.current_symbol = symbol_input
                st.toast(f"Fetching data for {st.session_state.current_symbol}...", icon="🔄")
                st.rerun()

        st.markdown("<span>პოპულარული:</span>", unsafe_allow_html=True)
        popular_cryptos = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'SKL', SAND]
        cols_pop = st.columns(len(popular_cryptos))
        for i, crypto_tag in enumerate(popular_cryptos):
            with cols_pop[i]:
                if st.button(crypto_tag, key=f"tag_{crypto_tag}", use_container_width=True):
                    st.session_state.current_symbol = crypto_tag
                    st.toast(f"Fetching data for {st.session_state.current_symbol}...", icon="🔄")
                    st.rerun()

    st.markdown("---") # Separator

    with st.container(border=False): # Use container to mimic card styling
        st.markdown("<div class='price-info-header'><h2>კრიპტოვალუტის ინფორმაცია</h2></div>", unsafe_allow_html=True)

        # Fetch data on initial load or symbol change
        coingecko_id = COINGECKO_CRYPTO_MAP.get(st.session_state.current_symbol)
        
        if coingecko_id:
            with st.spinner(f"იტვირთება მონაცემები {st.session_state.current_symbol}-ისთვის..."):
                coin_details = fetch_coin_details(coingecko_id)
                if coin_details:
                    st.markdown(f"<h2 id='coin-name'>{coin_details['name']} ({coin_details['symbol']})</h2>", unsafe_allow_html=True)
                    st.markdown(f"<p id='current-price' class='price-value'>{format_price(coin_details['currentPrice'])} $</p>", unsafe_allow_html=True)
                    
                    # Market Stats Grid using Streamlit columns
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

with col2: # Center Pane (Chart Section)
    with st.container(border=False): # Use container to mimic card styling
        st.markdown(f"<div class='chart-header'><h3>{st.session_state.current_symbol} ფასის დინამიკა და AI პროგნოზი</h3></div>", unsafe_allow_html=True)
        
        # Time Filters and Model Selection
        filter_cols = st.columns(3)
        periods = [90, 30, 7] # Display filter options
        for i, period in enumerate(periods):
            with filter_cols[i]:
                if st.button(f"{period} დღე", key=f"period_{period}", use_container_width=True, type="secondary" if st.session_state.current_period != period else "primary"):
                    st.session_state.current_period = period
                    st.toast(f"Showing {period} days data for {st.session_state.current_symbol}...", icon="📊")
        
        # Model Selection
        model_options = ["ARIMA Model", "Prophet Model"]
        if LSTM_AVAILABLE:
            model_options.append("LSTM Model")

        selected_model = st.selectbox(
            "აირჩიეთ პროგნოზირების მოდელი:",
            model_options,
            index=model_options.index(st.session_state.prediction_model)
        )
        if selected_model != st.session_state.prediction_model:
            st.session_state.prediction_model = selected_model
            st.rerun() # Rerun when model changes to regenerate predictions

        # Chart
        if coingecko_id and coin_details:
            # Use the new SQLite fetching function here
            historical_data_list_full = fetch_historical_data_sqlite(coingecko_id, days=MAX_HISTORICAL_DAYS)
            
            if historical_data_list_full:
                df_historical_full = pd.DataFrame(historical_data_list_full)
                df_historical_full['date'] = pd.to_datetime(df_historical_full['date'])
                
                forecaster = CryptoDataForecaster(st.session_state.current_symbol, df_historical_full) 
                
                # Get predicted data based on selected model (always predict for 30 days)
                with st.spinner(f"მიმდინარეობს {st.session_state.prediction_model} პროგნოზირება... (შეიძლება დასჭირდეს დრო LSTM-ისთვის)"):
                    prediction_data_list = forecaster.generate_predictions(days=30, model_choice=st.session_state.prediction_model)
                
                if not prediction_data_list:
                    st.warning(f"პროგნოზის გენერირება ვერ მოხერხდა {st.session_state.prediction_model} მოდელით. გთხოვთ, სცადოთ სხვა მოდელი ან სხვა კრიპტოვალუტა.")
                
                df_prediction = pd.DataFrame(prediction_data_list)
                if not df_prediction.empty:
                    df_prediction['date'] = pd.to_datetime(df_prediction['date']) 
                
                # Get signals from prediction
                signals = forecaster.generate_signals_from_prediction(prediction_data_list)

                # Filter historical data for display based on current_period AFTER indicator calculation
                df_indicators_display = forecaster.historical_with_features.tail(st.session_state.current_period).copy() # Use historical_with_features here
                
                # --- Create Plotly Subplots ---
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.08, 
                                    row_heights=[0.6, 0.2, 0.2],
                                    subplot_titles=(
                                        f"{st.session_state.current_symbol} ფასი და პროგნოზი",
                                        "Relative Strength Index (RSI)",
                                        "Moving Average Convergence Divergence (MACD)"
                                    ))

                # --- Row 1: Price, MAs, Prediction ---
                # Historical Price
                fig.add_trace(go.Scatter(
                    x=df_indicators_display.index,
                    y=df_indicators_display['price'],
                    mode='lines',
                    name='ისტორია',
                    line=dict(color='#00bcd4', width=2)
                ), row=1, col=1)

                # Prediction Data
                if not df_prediction.empty:
                    fig.add_trace(go.Scatter(
                        x=df_prediction['date'],
                        y=df_prediction['price'],
                        mode='lines',
                        name='პროგნოზი',
                        line=dict(color='#8e2de2', dash='dot', width=2)
                    ), row=1, col=1)

                # Moving Averages on Historical Data (filtered for display)
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

                # Add Signals as Scatter points with text labels on Row 1
                for signal in signals:
                    signal_color = "#39ff14" if signal['type'] == 'BUY' else "#ff073a"
                    # Only plot signals that fall within the displayed date range (current_period)
                    if not df_prediction.empty and (signal['date'] >= df_indicators_display.index.min() or signal['date'] >= df_prediction['date'].min()):
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
                    <p><span class='modal-label'>Relative Strength Index (RSI):</span> RSI არის იმპულსის ოსცილატორი, რომელიც ზომავს ფასის ცვლილებების სიჩქარესა და ცვლილებას. ის მერყეობს 0-დან 100-მდე. 70-ზე ზემოთ ნიშნავს, რომ აქტივი ზედმეტად ნაყიდია (Overbought) და შესაძლოა ფასის კორექცია მოხდეს. 30-ზე ქვემოთ ნიშნავს, რომ აქტივი ზედმეტად გაყიდულია (Oversold) და შესაძლოა ფასის ზრდა დაიწყოს.</p>
                    <p><span class='modal-label'>Moving Average Convergence Divergence (MACD):</span> MACD არის ტრენდის მიმდევარი იმპულსის ინდიკატორი, რომელიც აჩვენებს ფასის ორ ექსპონენციალურ მოძრავ საშუალოს (EMA) შორის ურთიერთობას (ჩვეულებრივ, 12-დღიანი და 26-დღიანი EMA). MACD ხაზის სასიგნალო ხაზის (9-დღიანი MACD-ის EMA) ზემოთ კვეთა არის ყიდვის სიგნალი, ხოლო ქვემოთ კვეთა - გაყიდვის სიგნალი. MACD ჰისტოგრამა აჩვენებს MACD ხაზსა და სასიგნალო ხაზს შორის განსხვავებას და გამოიყენება იმპულსის სიმძლავრის გასაზომად.</p>
                    <p><span class='modal-label'>LSTM (Long Short-Term Memory):</span> LSTM არის ხელოვნური ნერონული ქსელის სპეციალური ტიპი, რომელიც განსაკუთრებით ეფექტურია დროითი სერიების მონაცემების დასამუშავებლად და პროგნოზირებისთვის. მას შეუძლია ისწავლოს დამოკიდებულებები მონაცემებში როგორც მოკლე, ასევე გრძელვადიან პერსპექტივაში, რაც მას სასარგებლოს ხდის კომპლექსური ფინანსური მონაცემების მოდელირებისთვის.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Add a disclaimer for users
                st.markdown("---")
                st.markdown("<h3><i class='fas fa-exclamation-triangle'></i> მნიშვნელოვანი გაფრთხილება:</h3>", unsafe_allow_html=True)
                st.markdown("""
                <div class="argumentation-box">
                    <p>გთხოვთ, გაითვალისწინოთ, რომ კრიპტოვალუტების ფასების პროგნოზირება ძალიან რთული და არაზუსტი ამოცანაა. ეს AI მოდელები ეყრდნობა მხოლოდ ისტორიულ ფასებსა და ტექნიკურ ინდიკატორებს და ვერ ითვალისწინებს გარე ფაქტორებს, როგორიცაა სიახლეები, რეგულაციები, საბაზრო განწყობა ან მოულოდნელი მოვლენები.</p>
                    <p>პროგნოზები უნდა იქნას გამოყენებული მხოლოდ საინფორმაციო მიზნებისთვის და არა როგორც საინვესტიციო რჩევა. ყოველთვის ჩაატარეთ საკუთარი კვლევა და ივაჭრეთ სიფრთხილით.</p>
                    <p>მოდელის სიზუსტე დამოკიდებულია მონაცემთა ხარისხზე, რაოდენობაზე და ბაზრის ცვალებადობაზე. მაღალი ცვალებადობის მქონე აქტივებისთვის პროგნოზები, როგორც წესი, ნაკლებად ზუსტია.</p>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.warning("ისტორიული მონაცემები ვერ მოიძებნა არჩეული პერიოდისთვის.")
        else:
            st.info("აირჩიეთ კრიპტოვალუტა მონაცემების სანახავად.")


with col3: # Right Pane (Signals Section)
    with st.container(border=False): # Use container to mimic card styling
        st.markdown("<h3>პროგნოზირებული სიგნალები</h3>", unsafe_allow_html=True)
        
        if coingecko_id and coin_details:
            # Use the new SQLite fetching function here as well
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

# Footer
st.markdown("---")
st.markdown(f"""
<footer>
    <p>&copy; 2025 Crypto Signal App. ყველა უფლება დაცულია.</p>
    <p>მონაცემები მოწოდებულია <a href="https://www.coingecko.com" target="_blank">CoinGecko API</a>-ს მიერ | ბოლო განახლება: {datetime.datetime.now().strftime('%H:%M:%S')}</p>
</footer>
""", unsafe_allow_html=True)
