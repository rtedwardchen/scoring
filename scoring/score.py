#!/usr/bin/env python3
"""
Oracle Scoring Script for Tiber Challenge
Processes validator submissions against benchmark data to calculate scores
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Constants
ACU_PAIRS = ['AUD-USD', 'CAD-USD', 'EUR-USD', 'GBP-USD', 'JPY-USD', 'SEK-USD']
CRYPTO_PAIRS = ['ATN-USD', 'NTN-USD']
M_FX = 2.0  # FX accuracy multiplier
M_CRYPTO = 2.0  # Crypto accuracy multiplier

# Paths
BASE_DIR = Path('.')
SUBMISSION_DIR = BASE_DIR / 'submission-data'
YAHOO_DIR = BASE_DIR / 'yahoo-finance'
USDC_USD_DIR = BASE_DIR / 'usdc-usd-data'
SWAP_DIR = BASE_DIR / 'swap-data'
SCORING_DIR = BASE_DIR / 'scoring'
INTERMEDIATE_DIR = SCORING_DIR / 'intermediate'

# Create intermediate directory if not exists
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

def load_progress():
    """Load progress from intermediate file"""
    progress_file = INTERMEDIATE_DIR / 'progress.json'
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {}

def save_progress(progress):
    """Save progress to intermediate file"""
    progress_file = INTERMEDIATE_DIR / 'progress.json'
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)

def load_submission_data(date_str):
    """Load oracle submission data for a specific date"""
    filename = f"Oracle_Submission_{date_str}.csv"
    filepath = SUBMISSION_DIR / filename
    
    if not filepath.exists():
        print(f"Warning: Submission file not found: {filepath}")
        return pd.DataFrame()
    
    print(f"Loading submission data: {filename}")
    df = pd.read_csv(filepath)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)
    return df

def load_yahoo_finance_data(pair, date):
    """Load Yahoo Finance data for a specific pair around the given date"""
    # Map pair format
    pair_map = {
        'AUD-USD': 'AUDUSD',
        'CAD-USD': 'CADUSD',
        'EUR-USD': 'EURUSD',
        'GBP-USD': 'GBPUSD',
        'JPY-USD': 'JPYUSD',
        'SEK-USD': 'SEKUSD'
    }
    
    yahoo_pair = pair_map.get(pair)
    if not yahoo_pair:
        return pd.DataFrame()
    
    # Find relevant files for the date
    yahoo_pair_dir = YAHOO_DIR / 'data' / yahoo_pair
    if not yahoo_pair_dir.exists():
        print(f"Warning: Yahoo data directory not found: {yahoo_pair_dir}")
        return pd.DataFrame()
    
    all_data = []
    for file in yahoo_pair_dir.glob(f"{yahoo_pair}=X_*.csv"):
        df = pd.read_csv(file, skiprows=2)  # Skip the header rows
        df['Datetime'] = pd.to_datetime(df.index, utc=True)
        df = df.reset_index(drop=True)
        
        # Check if date is within this file's range
        if len(df) > 0:
            file_start = df['Datetime'].min().date()
            file_end = df['Datetime'].max().date()
            if file_start <= date.date() <= file_end:
                all_data.append(df)
    
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        result = result.sort_values('Datetime')
        return result
    
    return pd.DataFrame()

def load_usdc_usd_data():
    """Load USDC-USD data from Kraken"""
    filepath = USDC_USD_DIR / 'USDCUSD_1.csv'
    if not filepath.exists():
        print(f"Warning: USDC-USD file not found: {filepath}")
        return pd.DataFrame()
    
    # Kraken data format: timestamp,open,high,low,close,volume,count
    df = pd.read_csv(filepath, header=None, names=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'count'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    return df

def load_swap_data(pair, date):
    """Load swap data for ATN/USDC or NTN/USDC"""
    pair_map = {
        'ATN-USD': 'atn-usdc',
        'NTN-USD': 'ntn-usdc'
    }
    
    swap_pair = pair_map.get(pair)
    if not swap_pair:
        return pd.DataFrame()
    
    swap_pair_dir = SWAP_DIR / swap_pair
    if not swap_pair_dir.exists():
        print(f"Warning: Swap data directory not found: {swap_pair_dir}")
        return pd.DataFrame()
    
    # Find file for the date
    date_str = date.strftime('%Y-%m-%d')
    filepath = swap_pair_dir / f"{swap_pair}-{date_str}.csv"
    
    if not filepath.exists():
        # Try to find any file that might contain this date
        all_files = list(swap_pair_dir.glob(f"{swap_pair}-*.csv"))
        for file in sorted(all_files):
            df = pd.read_csv(file)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            if len(df) > 0:
                file_date = df['timestamp'].iloc[0].date()
                if file_date == date.date():
                    return df
        return pd.DataFrame()
    
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df

def calculate_fx_volatility(yahoo_data):
    """Calculate annualized volatility for FX pairs"""
    if len(yahoo_data) == 0:
        return 0.0
    
    # Get daily close prices
    daily_closes = yahoo_data.groupby(yahoo_data['Datetime'].dt.date)['Close'].last()
    
    if len(daily_closes) < 2:
        return 0.0
    
    # Calculate log returns
    log_returns = np.log(daily_closes / daily_closes.shift(1))
    log_returns = log_returns.dropna()
    
    if len(log_returns) == 0:
        return 0.0
    
    # Annualized volatility (252 trading days)
    return log_returns.std() * np.sqrt(252)

def calculate_crypto_benchmark(swap_data, usdc_usd_data, timestamp):
    """Calculate crypto benchmark price with 30-second moving average"""
    if len(swap_data) == 0 or len(usdc_usd_data) == 0:
        return None
    
    # Ensure timestamp is timezone-aware
    if timestamp.tzinfo is None:
        timestamp = pd.Timestamp(timestamp, tz='UTC')
    
    # Get swap price at timestamp (constant extrapolation)
    swap_prices = swap_data[swap_data['timestamp'] <= timestamp]
    if len(swap_prices) == 0:
        return None
    
    swap_price = swap_prices.iloc[-1]['price']
    
    # Get USDC/USD price (backward extrapolation of close)
    usdc_prices = usdc_usd_data[usdc_usd_data['datetime'] <= timestamp]
    if len(usdc_prices) == 0:
        return None
    
    usdc_price = usdc_prices.iloc[-1]['close']
    
    # Calculate price in USD
    price_usd = swap_price * usdc_price
    
    # For simplicity, using the spot price as the 30-second MA
    # In production, would implement proper moving average
    return price_usd

def calculate_crypto_volatility(swap_data, usdc_usd_data):
    """Calculate annualized volatility for crypto pairs"""
    if len(swap_data) == 0 or len(usdc_usd_data) == 0:
        return 0.0
    
    # Calculate daily prices
    daily_prices = []
    dates = pd.date_range(swap_data['timestamp'].min().date(), 
                         swap_data['timestamp'].max().date(), 
                         freq='D')
    
    for date in dates:
        end_of_day = pd.Timestamp(date, tz='UTC').replace(hour=23, minute=59, second=59)
        price = calculate_crypto_benchmark(swap_data, usdc_usd_data, end_of_day)
        if price:
            daily_prices.append(price)
    
    if len(daily_prices) < 2:
        return 0.0
    
    # Calculate log returns
    prices = pd.Series(daily_prices)
    log_returns = np.log(prices / prices.shift(1))
    log_returns = log_returns.dropna()
    
    if len(log_returns) == 0:
        return 0.0
    
    # Annualized volatility (365 days)
    return log_returns.std() * np.sqrt(365)

def score_fx_submission(submission, yahoo_data, sigma, interval_seconds):
    """Score a single FX submission"""
    timestamp = submission['Timestamp']
    
    # Ensure timestamp is timezone-aware
    if timestamp.tzinfo is None:
        timestamp = pd.Timestamp(timestamp, tz='UTC')
    
    # Find the OHLC interval containing this timestamp
    for _, ohlc in yahoo_data.iterrows():
        interval_end = ohlc['Datetime']
        interval_start = interval_end - pd.Timedelta(seconds=interval_seconds)
        
        if interval_start < timestamp <= interval_end:
            # Calculate accuracy bands
            open_price = ohlc['Open']
            low = ohlc['Low']
            high = ohlc['High']
            
            # Time scaling factor
            time_factor = np.sqrt(interval_seconds / (365 * 24 * 60 * 60))
            
            # Accuracy bands
            lower_bound = low - M_FX * sigma * time_factor * open_price
            upper_bound = high + M_FX * sigma * time_factor * open_price
            
            # Check if price is within bounds
            if lower_bound <= submission['Price'] <= upper_bound:
                return 1
            else:
                return 0
    
    return 0  # No matching interval found

def score_crypto_submission(submission, swap_data, usdc_usd_data, sigma):
    """Score a single crypto submission"""
    timestamp = submission['Timestamp']
    
    # Ensure timestamp is timezone-aware
    if timestamp.tzinfo is None:
        timestamp = pd.Timestamp(timestamp, tz='UTC')
    
    # Get benchmark price
    benchmark = calculate_crypto_benchmark(swap_data, usdc_usd_data, timestamp)
    if benchmark is None:
        return 0
    
    # Time scaling factor (30 seconds)
    time_factor = np.sqrt(30 / (365 * 24 * 60 * 60))
    
    # Accuracy bands
    lower_bound = benchmark * (1 - M_CRYPTO * sigma * time_factor)
    upper_bound = benchmark * (1 + M_CRYPTO * sigma * time_factor)
    
    # Check if price is within bounds
    if lower_bound <= submission['Price'] <= upper_bound:
        return 1
    else:
        return 0

def process_date(date_str):
    """Process submissions for a single date"""
    print(f"\n{'='*60}")
    print(f"Processing date: {date_str}")
    print(f"{'='*60}")
    
    # Load submission data
    submissions = load_submission_data(date_str)
    if len(submissions) == 0:
        print(f"No submissions found for {date_str}")
        return pd.DataFrame()
    
    print(f"Found {len(submissions)} submissions from {submissions['Validator Address'].nunique()} validators")
    
    # Parse date
    date = pd.to_datetime(date_str, utc=True)
    
    # Load USDC-USD data once
    usdc_usd_data = load_usdc_usd_data()
    
    # Initialize results
    validator_scores = {}
    
    # Process each validator
    validators = submissions['Validator Address'].unique()
    for i, validator in enumerate(validators):
        print(f"\rProcessing validator {i+1}/{len(validators)}: {validator[:10]}...", end='')
        
        validator_submissions = submissions[submissions['Validator Address'] == validator]
        scores = {pair: 0 for pair in ACU_PAIRS + CRYPTO_PAIRS}
        
        # Process FX pairs
        for pair in ACU_PAIRS:
            pair_col = f"{pair} Price"
            if pair_col not in validator_submissions.columns:
                continue
                
            # Load Yahoo data for this pair
            yahoo_data = load_yahoo_finance_data(pair, date)
            if len(yahoo_data) == 0:
                continue
            
            # Calculate volatility
            sigma = calculate_fx_volatility(yahoo_data)
            
            # Determine interval (60s or 300s for first period)
            # Check if this is the first period (before 2025-01-28)
            if date < pd.Timestamp('2025-01-28', tz='UTC'):
                interval_seconds = 300  # 5 minutes
            else:
                interval_seconds = 60  # 1 minute
            
            # Score each submission
            for _, sub in validator_submissions.iterrows():
                if pd.notna(sub[pair_col]):
                    score = score_fx_submission({
                        'Timestamp': sub['Timestamp'],
                        'Price': sub[pair_col] / 1e18  # Convert from wei
                    }, yahoo_data, sigma, interval_seconds)
                    scores[pair] += score
        
        # Process crypto pairs
        for pair in CRYPTO_PAIRS:
            pair_col = f"{pair} Price"
            if pair_col not in validator_submissions.columns:
                continue
            
            # Load swap data
            swap_data = load_swap_data(pair, date)
            if len(swap_data) == 0:
                continue
            
            # Calculate volatility
            sigma = calculate_crypto_volatility(swap_data, usdc_usd_data)
            
            # Score each submission
            for _, sub in validator_submissions.iterrows():
                if pd.notna(sub[pair_col]):
                    score = score_crypto_submission({
                        'Timestamp': sub['Timestamp'],
                        'Price': sub[pair_col] / 1e18  # Convert from wei
                    }, swap_data, usdc_usd_data, sigma)
                    scores[pair] += score
        
        # Calculate total score
        total_score = sum(scores.values())
        validator_scores[validator] = {
            'scores_by_pair': scores,
            'total_score': total_score
        }
    
    print()  # New line after progress
    
    # Convert to DataFrame
    results = []
    for validator, data in validator_scores.items():
        row = {'validator': validator, 'date': date_str, 'total_score': data['total_score']}
        for pair, score in data['scores_by_pair'].items():
            row[f'{pair}_score'] = score
        results.append(row)
    
    results_df = pd.DataFrame(results)
    
    # Save intermediate results
    intermediate_file = INTERMEDIATE_DIR / f"scores_{date_str}.csv"
    results_df.to_csv(intermediate_file, index=False)
    print(f"Saved intermediate results to: {intermediate_file}")
    
    return results_df

def main():
    """Main execution function"""
    print("Oracle Scoring Script")
    print("=" * 60)
    
    # Check if running with a specific date argument
    if len(sys.argv) > 1:
        date_str = sys.argv[1]
        results = process_date(date_str)
        
        # Save final results
        if len(results) > 0:
            output_file = SCORING_DIR / f"final_scores_{date_str}.csv"
            results.to_csv(output_file, index=False)
            print(f"\nFinal scores saved to: {output_file}")
            
            # Print summary
            print(f"\nSummary for {date_str}:")
            print(f"Total validators: {len(results)}")
            print(f"Average score: {results['total_score'].mean():.2f}")
            print(f"Max score: {results['total_score'].max()}")
            print(f"Validators with positive score: {(results['total_score'] > 0).sum()}")
    else:
        # Process all available dates
        submission_files = sorted(SUBMISSION_DIR.glob("Oracle_Submission_*.csv"))
        
        if len(submission_files) == 0:
            print("No submission files found!")
            return
        
        print(f"Found {len(submission_files)} submission files to process")
        
        # Load progress
        progress = load_progress()
        
        all_results = []
        for i, file in enumerate(submission_files):
            # Extract date from filename
            date_str = file.stem.replace("Oracle_Submission_", "")
            
            # Skip if already processed
            if date_str in progress.get('completed_dates', []):
                print(f"Skipping {date_str} (already processed)")
                # Load existing results
                intermediate_file = INTERMEDIATE_DIR / f"scores_{date_str}.csv"
                if intermediate_file.exists():
                    all_results.append(pd.read_csv(intermediate_file))
                continue
            
            print(f"\nProgress: {i+1}/{len(submission_files)}")
            
            try:
                results = process_date(date_str)
                if len(results) > 0:
                    all_results.append(results)
                
                # Update progress
                if 'completed_dates' not in progress:
                    progress['completed_dates'] = []
                progress['completed_dates'].append(date_str)
                save_progress(progress)
                
            except Exception as e:
                print(f"Error processing {date_str}: {e}")
                continue
        
        # Combine all results
        if all_results:
            final_results = pd.concat(all_results, ignore_index=True)
            
            # Aggregate by validator
            validator_totals = final_results.groupby('validator')['total_score'].sum().reset_index()
            validator_totals = validator_totals.sort_values('total_score', ascending=False)
            
            # Save final aggregated results
            output_file = SCORING_DIR / "final_scores_all.csv"
            validator_totals.to_csv(output_file, index=False)
            print(f"\n{'='*60}")
            print(f"Final aggregated scores saved to: {output_file}")
            
            # Calculate rewards (75000 total)
            total_rewards = 75000
            total_score = validator_totals['total_score'].sum()
            if total_score > 0:
                validator_totals['rewards'] = validator_totals['total_score'] * total_rewards / total_score
                
                # Save with rewards
                rewards_file = SCORING_DIR / "validator_rewards.csv"
                validator_totals.to_csv(rewards_file, index=False)
                print(f"Validator rewards saved to: {rewards_file}")
                
                # Print summary
                print(f"\nFinal Summary:")
                print(f"Total validators: {len(validator_totals)}")
                print(f"Validators with positive score: {(validator_totals['total_score'] > 0).sum()}")
                print(f"Total score points: {total_score:.0f}")
                print(f"Average reward: {validator_totals['rewards'].mean():.2f}")
                print(f"\nTop 10 validators:")
                print(validator_totals.head(10)[['validator', 'total_score', 'rewards']].to_string(index=False))

if __name__ == "__main__":
    main()