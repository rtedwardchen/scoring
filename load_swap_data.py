"""
Connect to TimescaleDB using environment variables for security.
"""
import os
import pandas as pd
from pandas import DataFrame
import psycopg2
from typing import Optional, Union, Tuple
from decimal import Decimal
from datetime import datetime, timedelta

# Use environment variables for sensitive credentials
TIMESCALE_HOST_AUTONITY_CHAINDATA = os.getenv(
    "TIMESCALE_HOST_AUTONITY_CHAINDATA", 
    "xpp0ph5sfb.pqxltlgf6n.tsdb.cloud.timescale.com"
)
TIMESCALE_PORT_AUTONITY_CHAINDATA = int(os.getenv(
    "TIMESCALE_PORT_AUTONITY_CHAINDATA", 
    "32139"
))
TIMESCALE_DATABASE_AUTONITY_CHAINDATA = os.getenv(
    "TIMESCALE_DATABASE_AUTONITY_CHAINDATA", 
    "tsdb"
)
TIMESCALE_USER_AUTONITY_CHAINDATA = os.getenv(
    "TIMESCALE_USER_AUTONITY_CHAINDATA", 
    "readonly_user"
)
TIMESCALE_PASSWORD_AUTONITY_CHAINDATA = os.getenv(
    "TIMESCALE_PASSWORD_AUTONITY_CHAINDATA", 
    "qHt&HTh&o4#tw76K"
)

# Create connection
conn = psycopg2.connect(
    host=TIMESCALE_HOST_AUTONITY_CHAINDATA,
    database=TIMESCALE_DATABASE_AUTONITY_CHAINDATA,
    user=TIMESCALE_USER_AUTONITY_CHAINDATA,
    password=TIMESCALE_PASSWORD_AUTONITY_CHAINDATA,
    port=TIMESCALE_PORT_AUTONITY_CHAINDATA,
    sslmode="require",
)

def run_query(query: str, conn) -> DataFrame:
    """
    Runs a query and returns results as a DataFrame.
    """
    cursor = conn.cursor()
    cursor.execute(query)  # Fixed: removed psycopg2.sql.SQL()
    rows = cursor.fetchall()
    colnames = [desc[0] for desc in cursor.description]
    return DataFrame(rows, columns=colnames)

"""
Pull ATN and NTN data from AMM pools.
"""

def get_atn_price(date: Optional[str] = None) -> DataFrame:
    """
    Get ATN-USDC pool data for a specific date.
    
    Args:
        date: Date string in format 'YYYY-MM-DD' (e.g., '2025-01-01')
              If None, fetches data for today
    """
    if date is None:
        # Get today's date
        date_condition = "timestamp >= CURRENT_DATE AND timestamp < CURRENT_DATE + INTERVAL '1 DAY'"
    else:
        # Get data for the specific date (00:00:00 to 23:59:59)
        date_condition = f"timestamp >= '{date}'::date AND timestamp < '{date}'::date + INTERVAL '1 DAY'"
    
    atn_usdc_state = run_query(
        f"""
        SELECT *
        FROM piccadilly_65100004.contract_2073d57cae6642223876ba3bf56868cc736d977c_state
        WHERE {date_condition}
        ORDER BY timestamp
        """,
        conn
    )
    
    try:
        # Calculate price: USDC/ATN (assuming token0 is USDC, token1 is ATN)
        atn_usdc_state["price"] = (
            atn_usdc_state["token0_reserve"]
            / atn_usdc_state["token1_reserve"]
            * Decimal(10**12)  # Adjust for decimal places difference
        )
        atn_usdc_state["price"] = [float(p) for p in atn_usdc_state["price"]]
    except ZeroDivisionError:
        atn_usdc_state["price"] = float("nan")
    except Exception as e:
        print(f"Warning: price calculation in get_atn_price failed: {e}")
        atn_usdc_state["price"] = float("nan")
    
    return atn_usdc_state

def get_ntn_price(date: Optional[str] = None) -> DataFrame:
    """
    Get NTN-USDC pool data for a specific date.
    
    Args:
        date: Date string in format 'YYYY-MM-DD' (e.g., '2025-01-01')
              If None, fetches data for today
    """
    if date is None:
        # Get today's date
        date_condition = "timestamp >= CURRENT_DATE AND timestamp < CURRENT_DATE + INTERVAL '1 DAY'"
    else:
        # Get data for the specific date (00:00:00 to 23:59:59)
        date_condition = f"timestamp >= '{date}'::date AND timestamp < '{date}'::date + INTERVAL '1 DAY'"
    
    ntn_usdc_state = run_query(
        f"""
        SELECT *
        FROM piccadilly_65100004.contract_caf123b55375e3ce1f20368d605086b5b0b767ed_state
        WHERE {date_condition}
        ORDER BY timestamp
        """,
        conn
    )
    
    try:
        # Calculate price: USDC/NTN (assuming token0 is USDC, token1 is NTN)
        ntn_usdc_state["price"] = (
            ntn_usdc_state["token0_reserve"]
            / ntn_usdc_state["token1_reserve"]
            * Decimal(10**12)  # Adjust for decimal places difference
        )
        ntn_usdc_state["price"] = [float(p) for p in ntn_usdc_state["price"]]
    except ZeroDivisionError:
        ntn_usdc_state["price"] = float("nan")
    except Exception as e:
        print(f"Warning: price calculation in get_ntn_price failed: {e}")
        ntn_usdc_state["price"] = float("nan")
    
    return ntn_usdc_state

# Helper function to get data for multiple days
def get_price_data_range(start_date: str, end_date: str, token: str = "ATN") -> DataFrame:
    """
    Get price data for a range of dates.
    
    Args:
        start_date: Start date in format 'YYYY-MM-DD'
        end_date: End date in format 'YYYY-MM-DD'
        token: Either 'ATN' or 'NTN'
    
    Returns:
        DataFrame with all data in the date range
    """
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    all_data = []
    current_date = start
    
    get_price_func = get_atn_price if token.upper() == "ATN" else get_ntn_price
    
    while current_date <= end:
        date_str = current_date.strftime('%Y-%m-%d')
        print(f"Fetching {token} data for {date_str}...")
        daily_data = get_price_func(date_str)
        if not daily_data.empty:
            all_data.append(daily_data)
        current_date += timedelta(days=1)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()

# Function to fetch and save daily data as CSV files
def fetch_and_save_price_data(start_date: str, end_date: str, tokens: list = ["ATN", "NTN"], 
                              base_dir: str = "swap-data") -> dict:
    """
    Fetch price data for a date range and save each day's data as a CSV file.
    
    Args:
        start_date: Start date in format 'YYYY-MM-DD'
        end_date: End date in format 'YYYY-MM-DD'
        tokens: List of tokens to fetch data for (default: ["ATN", "NTN"])
        base_dir: Base directory for saving CSV files (default: "swap-data")
    
    Returns:
        Dictionary with statistics about saved files
    """
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    stats = {
        'total_files': 0,
        'empty_days': 0,
        'saved_files': []
    }
    
    for token in tokens:
        token_upper = token.upper()
        token_lower = token.lower()
        
        # Create directory structure
        dir_path = os.path.join(base_dir, f"{token_lower}-usdc")
        os.makedirs(dir_path, exist_ok=True)
        
        # Get the appropriate function
        get_price_func = get_atn_price if token_upper == "ATN" else get_ntn_price
        
        current_date = start
        while current_date <= end:
            date_str = current_date.strftime('%Y-%m-%d')
            
            print(f"Fetching {token_upper} data for {date_str}...")
            daily_data = get_price_func(date_str)
            
            if not daily_data.empty:
                # Create filename
                filename = f"{token_lower}-usdc-{date_str}.csv"
                filepath = os.path.join(dir_path, filename)
                
                # Save to CSV
                daily_data.to_csv(filepath, index=False)
                print(f"  Saved {len(daily_data)} records to {filepath}")
                
                stats['total_files'] += 1
                stats['saved_files'].append(filepath)
            else:
                print(f"  No data found for {date_str}")
                stats['empty_days'] += 1
            
            current_date += timedelta(days=1)
    
    return stats

# Function to load saved CSV files
def load_saved_data(date_or_range: Union[str, Tuple[str, str]], token: str = "ATN", 
                    base_dir: str = "swap-data") -> DataFrame:
    """
    Load saved CSV files for a specific date or date range.
    
    Args:
        date_or_range: Either a single date string 'YYYY-MM-DD' or tuple of (start_date, end_date)
        token: Either 'ATN' or 'NTN'
        base_dir: Base directory where CSV files are saved
    
    Returns:
        DataFrame with loaded data
    """
    token_lower = token.lower()
    dir_path = os.path.join(base_dir, f"{token_lower}-usdc")
    
    all_data = []
    
    # Handle single date or date range
    if isinstance(date_or_range, str):
        # Single date
        filename = f"{token_lower}-usdc-{date_or_range}.csv"
        filepath = os.path.join(dir_path, filename)
        if os.path.exists(filepath):
            data = pd.read_csv(filepath)
            all_data.append(data)
    else:
        # Date range (tuple)
        start_date, end_date = date_or_range
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        current_date = start
        while current_date <= end:
            date_str = current_date.strftime('%Y-%m-%d')
            filename = f"{token_lower}-usdc-{date_str}.csv"
            filepath = os.path.join(dir_path, filename)
            
            if os.path.exists(filepath):
                data = pd.read_csv(filepath)
                all_data.append(data)
            
            current_date += timedelta(days=1)
    
    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp')
    else:
        return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    # Save only ATN data for a specific range
    stats_atn = fetch_and_save_price_data("2024-12-18", "2025-01-02", tokens=["ATN"])

    # Save only NTN data for a specific range
    stats_ntn = fetch_and_save_price_data("2024-12-18", "2025-01-02", tokens=["NTN"])
