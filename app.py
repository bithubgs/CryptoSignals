import streamlit as st
import datetime
import requests
import time
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import os
import numpy as np

# --- ტექნიკური ანალიზის ბიბლიოთეკების იმპორტი ---
try:
    import pandas_ta as ta
except ImportError:
    st.error("Pandas TA ბიბლიოთეკა ვერ მოიძებნა. დააინსტალირეთ: pip install pandas_ta")
    ta = None

try:
    from scipy.signal import find_peaks
except ImportError:
    st.error("SciPy ბიბლიოთეკა ვერ მოიძებნა. დააინსტალირეთ: pip install scipy")
    find_peaks = None

# --- გვერდის კონფიგურაცია ---
st.set_page_config(layout="wide", page_title="კრიპტო ანალიტიკა LIVE", page_icon="🕯️")

# --- კონფიგურაცია ---
COINGECKO_API_BASE = "https://api.coingecko.com/api/v3"
API_CALL_INTERVAL = 2.5 # API ლიმიტის დასაცავად

COINGECKO_CRYPTO_MAP = {
    'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana', 'ADA': 'cardano', 'DOT': 'polkadot',
    'LINK': 'chainlink', 'MATIC': 'polygon', 'AVAX': 'avalanche-2', 'ATOM': 'cosmos', 'LTC': 'litecoin',
    'BCH': 'bitcoin-cash', 'XRP': 'ripple', 'DOGE': 'dogecoin', 'SHIB': 'shiba-inu', 'UNI': 'uniswap',
    'AAVE': 'aave', 'SAND': 'the-sandbox', 'MANA': 'decentraland', 'AXS': 'axie-infinity', 'SKL': 'skale'
}

# --- SQLite კონფიგურაცია ---
DB_FILE = 'crypto_ohlc_data.db'

def init_db():
    """მონაცემთა ბაზის ინიციალიზაცია OHLC მონაცემებისთვის."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ohlc_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coingecko_id TEXT NOT NULL,
            interval TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            UNIQUE(coingecko_id, interval, timestamp)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- დამხმარე ფუნქციები ---
def rate_limit_api_call():
    """API მოთხოვნების სიხშირის რეგულირება."""
    if 'last_api_call' not in st.session_state:
        st.session_state.last_api_call = 0
    
    elapsed = time.time() - st.session_state.last_api_call
    if elapsed < API_CALL_INTERVAL:
        time.sleep(API_CALL_INTERVAL - elapsed)
    st.session_state.last_api_call = time.time()

def fetch_coin_details(coingecko_id):
    """კრიპტოვალუტის მიმდინარე დეტალების ჩატვირთვა."""
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
        }
    except Exception:
        return None

def fetch_ohlc_data(coingecko_id, days, interval_str):
    """OHLC (სანთლების) მონაცემების ჩატვირთვა API-დან და ქეშირება SQLite-ში."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    try:
        rate_limit_api_call()
        url = f"{COINGECKO_API_BASE}/coins/{coingecko_id}/ohlc"
        params = {'vs_currency': 'usd', 'days': str(days)}
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        api_data = response.json()

        df = pd.DataFrame(api_data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # მონაცემების შენახვა ბაზაში
        for _, row in df.iterrows():
            cursor.execute('''
                INSERT OR REPLACE INTO ohlc_prices (coingecko_id, interval, timestamp, open, high, low, close)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (coingecko_id, interval_str, row['timestamp'], row['open'], row['high'], row['low'], row['close']))
        conn.commit()
        
        st.toast(f"{interval_str} ინტერვალის მონაცემები განახლდა", icon="✅")
        return df

    except requests.exceptions.RequestException as e:
        st.error(f"OHLC მონაცემების ჩატვირთვის შეცდომა: {e}")
        return pd.DataFrame() # დააბრუნე ცარიელი DataFrame შეცდომისას
    finally:
        conn.close()

# --- ტექნიკური ანალიზის კლასი ---
class CryptoTechnicalAnalyzer:
    def __init__(self, ohlc_df):
        if ohlc_df.empty:
            raise ValueError("OHLC მონაცემები ცარიელია.")
        self.df = ohlc_df.copy()
        self.df.set_index('date', inplace=True)
        self.sr_levels = []
        self.patterns = pd.DataFrame(index=self.df.index)

    def analyze(self):
        """ყველა ანალიტიკური მეთოდის გაშვება."""
        if ta and find_peaks:
            self.find_support_resistance()
            self.find_candle_patterns()

    def find_support_resistance(self, prominence_factor=0.015, width=3, n_levels=7):
        """მხარდაჭერისა და წინაღობის დონეების პოვნა."""
        price_range = self.df['high'].max() - self.df['low'].min()
        prominence = price_range * prominence_factor

        high_peaks, _ = find_peaks(self.df['high'], prominence=prominence, width=width)
        low_peaks, _ = find_peaks(-self.df['low'], prominence=prominence, width=width)
        
        levels = []
        for peak in high_peaks:
            levels.append({'price': self.df['high'].iloc[peak], 'type': 'resistance'})
        for peak in low_peaks:
            levels.append({'price': self.df['low'].iloc[peak], 'type': 'support'})
        
        # დონეების დაჯგუფება და გაფილტვრა
        if not levels:
            self.sr_levels = []
            return

        levels_df = pd.DataFrame(levels).sort_values('price', ascending=False)
        grouped_levels = []
        
        while not levels_df.empty:
            current_level = levels_df.iloc[0]['price']
            cluster_range = current_level * 0.02 # 2% კლასტერი
            
            cluster = levels_df[(levels_df['price'] >= current_level - cluster_range) & 
                                (levels_df['price'] <= current_level + cluster_range)]
            
            if not cluster.empty:
                avg_price = cluster['price'].mean()
                level_type = cluster['type'].mode()[0]
                grouped_levels.append({'price': avg_price, 'type': level_type, 'strength': len(cluster)})
                levels_df = levels_df.drop(cluster.index)

        # დახარისხება სიძლიერის მიხედვით და საუკეთესოების არჩევა
        self.sr_levels = sorted(grouped_levels, key=lambda x: x['strength'], reverse=True)[:n_levels]

    def find_candle_patterns(self):
        """სანთლების პატერნების იდენტიფიცირება."""
        # pandas_ta იყენებს `cdl_pattern` ფუნქციას ყველა პატერნის მოსაძებნად
        self.df.ta.cdl_pattern(name="all", append=True)
        
        pattern_cols = [col for col in self.df.columns if col.startswith('CDL_')]
        for col in pattern_cols:
            # ვინახავთ მხოლოდ იმ პატერნებს, რომლებიც დაფიქსირდა
            found_patterns = self.df[self.df[col] != 0]
            if not found_patterns.empty:
                pattern_name = col.replace('CDL_', '').replace('_', ' ')
                pattern_type = 'bullish' if found_patterns[col].iloc[0] > 0 else 'bearish'
                for date, row in found_patterns.iterrows():
                    self.patterns.loc[date, 'pattern_name'] = pattern_name
                    self.patterns.loc[date, 'pattern_type'] = pattern_type

    def generate_signals(self, proximity_factor=0.01):
        """სიგნალების გენერირება ანალიზის საფუძველზე."""
        if not self.sr_levels or self.patterns.empty:
            return []

        signals = []
        last_row = self.df.iloc[-1]
        last_date = self.df.index[-1]
        
        # შემოწმება, არის თუ არა ბოლო სანთელზე პატერნი
        if last_date in self.patterns.index:
            last_pattern = self.patterns.loc[last_date]
            pattern_name = last_pattern['pattern_name']
            pattern_type = last_pattern['pattern_type']

            for level in self.sr_levels:
                proximity = abs(last_row['close'] - level['price']) / level['price']
                
                # BUY სიგნალი: აღმავალი პატერნი მხარდაჭერის დონესთან
                if level['type'] == 'support' and pattern_type == 'bullish' and proximity < proximity_factor:
                    signals.append({
                        'date': last_date, 'type': 'BUY', 'price': last_row['close'],
                        'confidence': "Medium",
                        'argumentation': f"დაფიქსირდა '{pattern_name}' პატერნი ძლიერ მხარდაჭერის დონესთან ({format_price(level['price'])}$)."
                    })
                
                # SELL სიგნალი: დაღმავალი პატერნი წინაღობის დონესთან
                elif level['type'] == 'resistance' and pattern_type == 'bearish' and proximity < proximity_factor:
                    signals.append({
                        'date': last_date, 'type': 'SELL', 'price': last_row['close'],
                        'confidence': "Medium",
                        'argumentation': f"დაფიქსირდა '{pattern_name}' პატერნი ძლიერ წინაღობის დონესთან ({format_price(level['price'])}$)."
                    })
        return signals

def format_price(price):
    if price is None: return 'N/A'
    return f"{price:,.2f}" if price >= 1 else f"{price:.8f}".rstrip('0').rstrip('.')

# --- აპლიკაციის სტრუქტურა ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    :root {
        --bg-dark: #0a0a0a; --bg-medium: #1a1a1a; --text-light: #e0e0e0;
        --accent-blue: #00bcd4; --accent-purple: #8e2de2; --accent-green: #4caf50;
        --accent-red: #f44336; --accent-yellow: #fdd835;
        --card-bg: #1c1c1c; --border-color: #333;
    }
    .stApp { font-family: 'Inter', sans-serif; background: var(--bg-dark); color: var(--text-light); }
    h1, h5, h6 { color: var(--text-light); }
    h1 { font-size: 2.5rem; font-weight: 800; text-shadow: 0 0 10px var(--accent-blue), 0 0 20px var(--accent-purple); }
    .st-emotion-cache-1fplhmq { background-color: var(--card-bg); border: 1px solid var(--border-color); border-radius: 15px; padding: 1.5rem; margin-bottom: 1.5rem; }
    .price-value { font-size: 2.2rem; font-weight: 700; color: var(--accent-yellow); margin: 0.25rem 0 1rem 0; }
    .signal-item { background-color: var(--bg-dark); padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 0.75rem; border-left: 4px solid var(--border-color); }
    .buy-signal-item { border-left-color: var(--accent-green) !important; }
    .sell-signal-item { border-left-color: var(--accent-red) !important; }
    .signal-type { font-weight: 700; }
    .buy-signal { color: var(--accent-green); }
    .sell-signal { color: var(--accent-red); }
</style>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
""", unsafe_allow_html=True)

st.title("კრიპტო ანალიტიკა LIVE")

# --- Session State ინიციალიზაცია ---
if 'current_symbol' not in st.session_state:
    st.session_state.current_symbol = 'BTC'
if 'time_interval' not in st.session_state:
    st.session_state.time_interval = '1D'

# --- მთავარი სტრუქტურა ---
col1, col2 = st.columns([0.8, 2.2])

with col1: # მარცხენა პანელი
    with st.container(border=True):
        st.markdown("<h5>კრიპტოვალუტის არჩევა</h5>", unsafe_allow_html=True)
        symbol_input = st.text_input("სიმბოლო", st.session_state.current_symbol, label_visibility="collapsed").upper().strip()
        if symbol_input != st.session_state.current_symbol:
            st.session_state.current_symbol = symbol_input
            st.rerun()

    coingecko_id = COINGECKO_CRYPTO_MAP.get(st.session_state.current_symbol)
    if coingecko_id:
        with st.container(border=True):
            coin_details = fetch_coin_details(coingecko_id)
            if coin_details:
                st.markdown(f"<h6>{coin_details['name']} ({coin_details['symbol']})</h6>", unsafe_allow_html=True)
                st.markdown(f"<p class='price-value'>{format_price(coin_details['currentPrice'])} $</p>", unsafe_allow_html=True)
                change = coin_details.get('dailyChange')
                st.metric("24სთ ცვლილება", f"{change:.2f}%" if change is not None else "N/A", f"{change:.2f}%" if change is not None else None)
            else:
                st.error("მონაცემები ვერ მოიძებნა.")
        
        with st.container(border=True):
            st.markdown("<h5>სიგნალები</h5>", unsafe_allow_html=True)
            signals_placeholder = st.empty()
            signals_placeholder.info("აირჩიეთ ინტერვალი ანალიზისთვის.")

    else:
        st.warning("გთხოვთ, აირჩიოთ ვალიდური კრიპტოვალუტის სიმბოლო.")


with col2: # მარჯვენა პანელი (ჩარტი)
    if coingecko_id:
        with st.container(border=True):
            # დროის ინტერვალის ამრჩევი
            intervals = {'1H': 7, '4H': 30, '1D': 365}
            interval_keys = list(intervals.keys())
            selected_interval_key = st.radio(
                "აირჩიეთ ინტერვალი:", interval_keys,
                index=interval_keys.index(st.session_state.time_interval),
                horizontal=True, label_visibility="collapsed"
            )
            if selected_interval_key != st.session_state.time_interval:
                st.session_state.time_interval = selected_interval_key
                st.rerun()
            
            days_to_fetch = intervals[st.session_state.time_interval]
            
            # OHLC მონაცემების ჩატვირთვა
            with st.spinner(f"იტვირთება {st.session_state.time_interval} მონაცემები..."):
                ohlc_df = fetch_ohlc_data(coingecko_id, days=days_to_fetch, interval_str=st.session_state.time_interval)

            if not ohlc_df.empty:
                # 4H ინტერვალის რესემპლინგი
                if st.session_state.time_interval == '4H':
                    ohlc_df.set_index('date', inplace=True)
                    ohlc_df = ohlc_df.resample('4H').agg({
                        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
                    }).dropna()
                    ohlc_df.reset_index(inplace=True)

                # ტექნიკური ანალიზი
                analyzer = CryptoTechnicalAnalyzer(ohlc_df)
                analyzer.analyze()
                signals = analyzer.generate_signals()

                # სიგნალების ჩვენება მარცხენა პანელში
                with signals_placeholder.container():
                    if signals:
                        for signal in signals:
                            st.markdown(f"""
                            <div class="signal-item {'buy-signal-item' if signal['type'] == 'BUY' else 'sell-signal-item'}">
                                <span class="signal-type {'buy-signal' if signal['type'] == 'BUY' else 'sell-signal'}">
                                    <i class="fas fa-{'arrow-up' if signal['type'] == 'BUY' else 'arrow-down'}"></i> {signal['type']}
                                </span>
                                <p style="font-size:0.9rem; margin: 5px 0 0 0;">{signal['argumentation']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("აქტიური სიგნალები არ მოიძებნა.")

                # ჩარტის აგება
                fig = go.Figure(data=[go.Candlestick(
                    x=analyzer.df.index,
                    open=analyzer.df['open'], high=analyzer.df['high'],
                    low=analyzer.df['low'], close=analyzer.df['close'],
                    name=st.session_state.current_symbol
                )])

                # მხარდაჭერისა და წინაღობის დონეების დახატვა
                for level in analyzer.sr_levels:
                    fig.add_hline(
                        y=level['price'],
                        line_width=1,
                        line_dash="dash",
                        line_color="yellow" if level['type'] == 'support' else "cyan",
                        annotation_text=f"{level['type'].capitalize()} ({format_price(level['price'])})",
                        annotation_position="bottom right",
                        annotation_font_size=10
                    )
                
                # სანთლების პატერნების დახატვა
                patterns_to_draw = analyzer.patterns.dropna()
                if not patterns_to_draw.empty:
                    fig.add_trace(go.Scatter(
                        x=patterns_to_draw.index,
                        y=analyzer.df.loc[patterns_to_draw.index, 'high'] * 1.01, # პატერნის მარკერი სანთლის ზემოთ
                        mode='text',
                        text=patterns_to_draw['pattern_name'].apply(lambda x: ''.join([c for c in x if c.isupper()])), # აბრევიატურა
                        textfont=dict(size=9, color='white'),
                        hoverinfo='text',
                        hovertext=patterns_to_draw['pattern_name'],
                        name='Patterns',
                        showlegend=False
                    ))

                fig.update_layout(
                    height=700, template='plotly_dark',
                    xaxis_rangeslider_visible=False,
                    margin=dict(l=20, r=20, t=20, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("მონაცემების ჩატვირთვა ვერ მოხერხდა. სცადეთ სხვა კრიპტოვალუტა ან ინტერვალი.")
