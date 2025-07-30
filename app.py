import streamlit as st
import datetime
import random
import requests
import time
import pandas as pd
import os # For file operations

# ... (other imports and configurations remain the same) ...

# Define a directory for data storage
DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# File path for historical data
def get_historical_data_filepath(coingecko_id):
    return os.path.join(DATA_DIR, f"{coingecko_id}_historical_data.csv")

# Function to check if data is "fresh enough" - e.g., updated today
def is_data_fresh(filepath):
    if not os.path.exists(filepath):
        return False
    # Check modification date of the file
    last_modified_timestamp = os.path.getmtime(filepath)
    last_modified_date = datetime.datetime.fromtimestamp(last_modified_timestamp).date()
    # Data is fresh if it was modified today
    return last_modified_date == datetime.date.today()

# Modified fetch_historical_data_coingecko to use file caching
def fetch_historical_data_coingecko_persistent(coingecko_id, days=MAX_HISTORICAL_DAYS):
    filepath = get_historical_data_filepath(coingecko_id)

    # 1. Try to load from file if it's fresh enough
    if is_data_fresh(filepath):
        try:
            df = pd.read_csv(filepath, parse_dates=['date'])
            st.toast(f"ისტორიული მონაცემები ჩაიტვირთა ქეშიდან: {coingecko_id}", icon="💾")
            return df.to_dict('records') # Convert back to list of dicts for consistency
        except Exception as e:
            st.warning(f"შეცდომა ქეშირებული მონაცემების ჩატვირთვისას, ვცდილობ API-ს: {e}")
            # If error loading cached file, proceed to fetch from API

    # 2. If no fresh data or error loading, fetch from API
    st.toast(f"მიმდინარეობს ისტორიული მონაცემების ჩატვირთვა API-დან: {coingecko_id}", icon="🌐")
    try:
        rate_limit_api_call() # Wait before making the call
        url = f"{COINGECKO_API_BASE}/coins/{coingecko_id}/market_chart"
        params = {'vs_currency': 'usd', 'days': str(days), 'interval': 'daily'}
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        historical_data = [{'date': datetime.datetime.fromtimestamp(p[0] / 1000), 'price': round(p[1], 8), 'type': 'historical'} for p in data.get('prices', [])]
        
        # 3. Save to file
        df_to_save = pd.DataFrame(historical_data)
        df_to_save.to_csv(filepath, index=False)
        st.toast(f"ისტორიული მონაცემები შენახულია ქეშში: {filepath}", icon="💾")
        
        return historical_data

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            st.error(f"Error: Coingecko API ლიმიტი გადაჭარბებულია ისტორიული მონაცემებისთვის. გთხოვთ, სცადოთ მოგვიანებით. ({e})")
        else:
            st.error(f"Error fetching historical data for {coingecko_id}: {e}")
        return []
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching historical data for {coingecko_id}: {e}")
        return []

# Replace existing call in your app.py:
# historical_data_list_full = fetch_historical_data_coingecko(coingecko_id, days=MAX_HISTORICAL_DAYS)
# with this:
# historical_data_list_full = fetch_historical_data_coingecko_persistent(coingecko_id, days=MAX_HISTORICAL_DAYS)
