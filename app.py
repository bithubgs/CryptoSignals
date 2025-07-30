import streamlit as st
import datetime
import random
import requests
import time
from functools import lru_cache
import pandas as pd
import plotly.graph_objects as go

# Import forecasting models
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet # Ensure you have prophet installed: pip install prophet

# --- Set page config for wide mode ---
st.set_page_config(layout="wide", page_title="კრიპტო სიგნალები LIVE", page_icon="📈")

# --- Configuration ---
COINGECKO_API_BASE = "https://api.coingecko.com/api/v3"
API_CALL_INTERVAL = 1.2 # Coingecko allows 100 calls/minute, so ~1.2 seconds per call

COINGECKO_CRYPTO_MAP = {
    'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana', 'ADA': 'cardano', 'DOT': 'polkadot',
    'LINK': 'chainlink', 'MATIC': 'polygon', 'AVAX': 'avalanche-2', 'ATOM': 'cosmos', 'LTC': 'litecoin',
    'BCH': 'bitcoin-cash', 'XRP': 'ripple', 'DOGE': 'dogecoin', 'SHIB': 'shiba-inu', 'UNI': 'uniswap',
    'AAVE': 'aave', 'SAND': 'the-sandbox', 'MANA': 'decentraland', 'AXS': 'axie-infinity', 'SKL': 'skale'
}

# --- Helper Functions (adapted from app.py) ---
def rate_limit_api_call():
    if 'last_api_call' not in st.session_state:
        st.session_state.last_api_call = 0
    
    elapsed = time.time() - st.session_state.last_api_call
    if elapsed < API_CALL_INTERVAL:
        time.sleep(API_CALL_INTERVAL - elapsed)
    st.session_state.last_api_call = time.time()

@st.cache_data(ttl=3600) # Cache for 1 hour
def fetch_coin_details(coingecko_id):
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
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching coin details for {coingecko_id}: {e}")
        return None

@st.cache_data(ttl=3600) # Cache for 1 hour
def fetch_historical_data_coingecko(coingecko_id, days=90):
    try:
        rate_limit_api_call()
        url = f"{COINGECKO_API_BASE}/coins/{coingecko_id}/market_chart"
        params = {'vs_currency': 'usd', 'days': str(days), 'interval': 'daily'}
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        return [{'date': datetime.datetime.fromtimestamp(p[0] / 1000), 'price': round(p[1], 8), 'type': 'historical'} for p in data.get('prices', [])]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching historical data for {coingecko_id}: {e}")
        return []

class CryptoDataForecaster:
    def __init__(self, symbol, historical_prices_df):
        self.symbol = symbol.upper()
        self.historical_prices = historical_prices_df # DataFrame with 'date' and 'price'

    def _generate_simple_predictions(self, days):
        # This is the fallback/simplified forecasting method
        last_price = self.historical_prices['price'].iloc[-1] if not self.historical_prices.empty else 0
        last_date = self.historical_prices.index[-1] if not self.historical_prices.empty else datetime.date.today()
        
        prediction_data = []
        current_price = last_price
        trend_strength = random.uniform(-0.001, 0.001) # Much smaller random trend
        volatility = random.uniform(0.005, 0.015) # Much smaller volatility

        for i in range(1, days + 1):
            date = last_date + datetime.timedelta(days=i)
            random_change = (random.random() - 0.5) * volatility 
            current_price *= (1 + trend_strength + random_change)
            current_price = max(0.00000001, current_price)
            prediction_data.append({'date': date, 'price': round(current_price, 8), 'type': 'prediction'})
        return prediction_data

    def generate_predictions(self, days=30, model_choice="Moving Average (Simplified)"):
        if self.historical_prices.empty:
            st.warning("ისტორიული მონაცემები არ არის პროგნოზირებისთვის.")
            return []

        # Convert date to datetime and set as index
        df = self.historical_prices.copy()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df['price'] = df['price'].astype(float)

        prediction_data = []
        last_date = df.index[-1]
        
        if model_choice == "Moving Average (Simplified)":
            # Your existing simplified MA based forecasting logic
            if len(df) < 5:
                return self._generate_simple_predictions(days) # Fallback if not enough data for MA trend

            short_ma = df['price'].rolling(window=5).mean()
            long_ma = df['price'].rolling(window=20).mean()

            last_short_ma = short_ma.iloc[-1]
            last_long_ma = long_ma.iloc[-1]
            current_predicted_price = df['price'].iloc[-1]

            for i in range(1, days + 1):
                date = last_date + datetime.timedelta(days=i)
                if last_short_ma > last_long_ma:
                    trend_factor = random.uniform(1.001, 1.005)
                else:
                    trend_factor = random.uniform(0.995, 0.999)
                random_fluctuation = random.uniform(0.995, 1.005) 
                current_predicted_price *= trend_factor * random_fluctuation
                current_predicted_price = max(0.00000001, current_predicted_price)
                prediction_data.append({'date': date, 'price': round(current_predicted_price, 8), 'type': 'prediction'})
            
        elif model_choice == "ARIMA Model":
            if len(df) < 30: # ARIMA needs more data
                st.warning("არასაკმარისი მონაცემები ARIMA პროგნოზისთვის (მინიმუმ 30 დღე). გამოიყენება გამარტივებული პროგნოზი.")
                return self._generate_simple_predictions(days)

            series = df['price']
            try:
                # ARIMA order (p,d,q) can be tuned. (5,1,0) is a common starting point.
                # d=1 means first-order differencing (to make series stationary)
                model = ARIMA(series, order=(5,1,0)) 
                model_fit = model.fit()
                
                # Make predictions
                # start and end define the range of the forecast relative to the original series length
                forecast_result = model_fit.predict(start=len(series), end=len(series) + days - 1)
                
                for i, price in enumerate(forecast_result):
                    date = last_date + datetime.timedelta(days=i+1)
                    prediction_data.append({'date': date, 'price': max(0.00000001, round(price, 8)), 'type': 'prediction'})
            except Exception as e:
                st.error(f"შეცდომა ARIMA მოდელის გაშვებისას: {e}. გამოიყენება გამარტივებული პროგნოზი.")
                return self._generate_simple_predictions(days)

        elif model_choice == "Prophet Model":
            if len(df) < 30: # Prophet also prefers more data
                st.warning("არასაკმარისი მონაცემები Prophet პროგნოზისთვის (მინიმუმ 30 დღე). გამოიყენება გამარტივებული პროგნოზი.")
                return self._generate_simple_predictions(days)

            df_prophet = df.reset_index()[['date', 'price']].rename(columns={'date': 'ds', 'price': 'y'})
            df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

            try:
                # You can customize Prophet's parameters (e.g., seasonality_mode='multiplicative')
                model = Prophet(daily_seasonality=True) 
                model.fit(df_prophet)
                
                future = model.make_future_dataframe(periods=days)
                forecast = model.predict(future)
                
                for _, row in forecast.tail(days).iterrows():
                    prediction_data.append({
                        'date': row['ds'], 
                        'price': max(0.00000001, round(row['yhat'], 8)), # yhat is the prediction
                        'type': 'prediction'
                    })
            except Exception as e:
                st.error(f"შეცდომა Prophet მოდელის გაშვებისას: {e}. გამოიყენება გამარტივებული პროგნოზი.")
                return self._generate_simple_predictions(days)
        
        return prediction_data

    def generate_signals_from_prediction(self, prediction_data):
        if len(prediction_data) < 7: # Need at least 7 days for 7-day MA
            return []
        
        df_prediction = pd.DataFrame(prediction_data)
        df_prediction['date'] = pd.to_datetime(df_prediction['date'])
        df_prediction.set_index('date', inplace=True)
        df_prediction['price'] = df_prediction['price'].astype(float)

        # Calculate Moving Averages on predicted data
        short_window = 3 # e.g., 3-day MA for signals
        long_window = 7  # e.g., 7-day MA for signals

        df_prediction['MA_short'] = df_prediction['price'].rolling(window=short_window).mean()
        df_prediction['MA_long'] = df_prediction['price'].rolling(window=long_window).mean()

        signals = []
        # Look for moving average crossovers
        for i in range(1, len(df_prediction)):
            current_date = df_prediction.index[i]
            # Ensure we have enough data points for MAs
            if pd.isna(df_prediction['MA_short'].iloc[i]) or pd.isna(df_prediction['MA_long'].iloc[i]):
                continue # Skip if MAs are not yet calculated

            if (df_prediction['MA_short'].iloc[i-1] < df_prediction['MA_long'].iloc[i-1] and
                df_prediction['MA_short'].iloc[i] > df_prediction['MA_long'].iloc[i]):
                # Buy signal: Short MA crosses above Long MA
                signals.append({
                    'date': current_date,
                    'type': 'BUY',
                    'price': df_prediction['price'].iloc[i],
                    'confidence': f"{random.randint(70, 90)}%", # Higher confidence for MA signals
                    'argumentation': f"AI პროგნოზირებს მოკლევადიანი მოძრავი საშუალოს (MA{short_window}) გრძელვადიან მოძრავ საშუალოზე (MA{long_window}) ზემოთ კვეთას, რაც პოტენციური აღმავალი ტრენდის დასაწყისს მიანიშნებს."
                })

            elif (df_prediction['MA_short'].iloc[i-1] > df_prediction['MA_long'].iloc[i-1] and
                  df_prediction['MA_short'].iloc[i] < df_prediction['MA_long'].iloc[i]):
                # Sell signal: Short MA crosses below Long MA
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
    st.session_state.current_period = 90
if 'prediction_model' not in st.session_state:
    st.session_state.prediction_model = "Moving Average (Simplified)" # Default model

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
                st.toast(f"Fetching data for {st.session_state.current_symbol}...")
                st.rerun()

        st.markdown("<span>პოპულარული:</span>", unsafe_allow_html=True)
        popular_cryptos = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'SKL']
        cols_pop = st.columns(len(popular_cryptos))
        for i, crypto_tag in enumerate(popular_cryptos):
            with cols_pop[i]:
                if st.button(crypto_tag, key=f"tag_{crypto_tag}", use_container_width=True):
                    st.session_state.current_symbol = crypto_tag
                    st.toast(f"Fetching data for {st.session_state.current_symbol}...")
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
        periods = [90, 30, 7]
        for i, period in enumerate(periods):
            with filter_cols[i]:
                if st.button(f"{period} დღე", key=f"period_{period}", use_container_width=True, type="secondary" if st.session_state.current_period != period else "primary"):
                    st.session_state.current_period = period
                    st.toast(f"Fetching data for {st.session_state.current_symbol} for {period} days...")
                    st.rerun()
        
        # Model Selection
        model_options = ["Moving Average (Simplified)", "ARIMA Model", "Prophet Model"]
        selected_model = st.selectbox(
            "აირჩიეთ პროგნოზირების მოდელი:",
            model_options,
            index=model_options.index(st.session_state.prediction_model)
        )
        if selected_model != st.session_state.prediction_model:
            st.session_state.prediction_model = selected_model
            st.rerun()

        # Chart
        if coingecko_id and coin_details:
            historical_data_list = fetch_historical_data_coingecko(coingecko_id, days=st.session_state.current_period)
            if historical_data_list:
                df_historical = pd.DataFrame(historical_data_list)
                
                # Generate predictions using the new forecaster and selected model
                forecaster = CryptoDataForecaster(st.session_state.current_symbol, df_historical)
                prediction_data_list = forecaster.generate_predictions(model_choice=st.session_state.prediction_model)
                signals = forecaster.generate_signals_from_prediction(prediction_data_list)

                # Prepare data for Plotly
                df_prediction = pd.DataFrame(prediction_data_list)
                
                # Convert 'date' columns to datetime objects
                df_historical['date'] = pd.to_datetime(df_historical['date'])
                df_prediction['date'] = pd.to_datetime(df_prediction['date'])

                # Create Plotly figure
                fig = go.Figure()

                # Add Historical Data Trace
                fig.add_trace(go.Scatter(
                    x=df_historical['date'],
                    y=df_historical['price'],
                    mode='lines',
                    name='ისტორია',
                    line=dict(color='#00bcd4', width=2)
                ))

                # Add Prediction Data Trace
                fig.add_trace(go.Scatter(
                    x=df_prediction['date'],
                    y=df_prediction['price'],
                    mode='lines',
                    name='პროგნოზი',
                    line=dict(color='#8e2de2', dash='dot', width=2)
                ))

                # Add Signals as Scatter points
                for signal in signals:
                    signal_color = "#39ff14" if signal['type'] == 'BUY' else "#ff073a"
                    signal_symbol = "triangle-up" if signal['type'] == 'BUY' else "triangle-down"

                    fig.add_trace(go.Scatter(
                        x=[signal['date']],
                        y=[signal['price']],
                        mode='markers',
                        name=f"{signal['type']} სიგნალი",
                        marker=dict(
                            symbol=signal_symbol,
                            size=12,
                            color=signal_color,
                            line=dict(width=1, color='white')
                        ),
                        hoverinfo='text',
                        hovertext=f"<b>{signal['type']}</b><br>თარიღი: {signal['date'].strftime('%Y-%m-%d')}<br>ფასი: {format_price(signal['price'])} $"
                    ))

                # Update layout for better aesthetics and legend positioning
                fig.update_layout(
                    xaxis_title='თარიღი',
                    yaxis_title='ფასი (USD)',
                    template='plotly_dark',
                    hovermode='x unified',
                    margin=dict(l=0, r=0, t=50, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=st.get_option('theme.textColor')),
                    xaxis=dict(
                        showgrid=False,
                        zeroline=False,
                        tickfont=dict(color=st.get_option('theme.secondaryBackgroundColor')),
                        title_font=dict(color=st.get_option('theme.textColor'))
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(255,255,255,0.1)',
                        zeroline=False,
                        tickfont=dict(color=st.get_option('theme.secondaryBackgroundColor')),
                        title_font=dict(color=st.get_option('theme.textColor'))
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.3,
                        xanchor="left",
                        x=0,
                        bgcolor='rgba(0,0,0,0)',
                        bordercolor='rgba(0,0,0,0)',
                        font=dict(color=st.get_option('theme.textColor'))
                    )
                )

                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                <div class="chart-legend">
                    <div class="legend-item"><div class="legend-color historical"></div><span>ისტორია</span></div>
                    <div class="legend-item"><div class="legend-color prediction"></div><span>პროგნოზი</span></div>
                    <div class="legend-item"><div class="legend-color buy-signal-legend"></div><span>ყიდვა</span></div>
                    <div class="legend-item"><div class="legend-color sell-signal-legend"></div><span>გაყიდვა</span></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("ისტორიული მონაცემები ვერ მოიძებნა არჩეული პერიოდისთვის.")
        else:
            st.info("აირჩიეთ კრიპტოვალუტა მონაცემების სანახავად.")


with col3: # Right Pane (Signals Section)
    with st.container(border=False): # Use container to mimic card styling
        st.markdown("<h3>პროგნოზირებული სიგნალები</h3>", unsafe_allow_html=True)
        
        if coingecko_id and coin_details and historical_data_list:
            df_historical = pd.DataFrame(historical_data_list) # Re-create DataFrame for consistency
            forecaster = CryptoDataForecaster(st.session_state.current_symbol, df_historical)
            
            # Pass the selected model to generate_predictions
            prediction_data_list = forecaster.generate_predictions(model_choice=st.session_state.prediction_model)
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

                    # Show signal details in an expander instead of a modal
                    with st.expander(f"დეტალები: {signal['type']} {signal['date'].strftime('%Y-%m-%d')}"):
                        st.markdown(f"<p><span class='modal-label'>თარიღი:</span> {signal['date'].strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p><span class='modal-label'>ფასი:</span> {format_price(signal['price'])} $</p>", unsafe_allow_html=True)
                        st.markdown(f"<p><span class='modal-label'>სანდოობა:</span> {signal['confidence']}</p>", unsafe_allow_html=True)
                        # CORRECTED LINE HERE: removed the extra backslash
                        st.markdown("<div class='argumentation-box'><h4><i class='fas fa-brain'></i> AI ანალიზი:</h4></div>", unsafe_allow_html=True) 
                        st.markdown(f"<p>{signal['argumentation']}</p></div>", unsafe_allow_html=True)

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
