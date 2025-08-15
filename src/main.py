import yfinance as yf
import pandas as pd
import itertools
import csv
from collections import defaultdict
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

# Top 100 tech companies
tickers = ['NVDA', 'MSFT', 'AAPL', 
                    'GOOG', 'AMZN', 'META', 
                    'AVGO', 'TSM', 'TSLA', 
                    'ORCL', 'TCEHY', 'NFLX', 
                    'PLTR', 'SAP', 'BABA', 
                    'CSCO', 'ASML', 'AMD', 
                    'CRM', 'IBM', 'INTU',
                    'SHOP', 'UBER', 'NOW', 
                    'BKNG', 'ANET', 'XIACF', 
                    'TXN', 'SONY', 'PDD', 
                    'ARM', 'SU.PA', 'SPOT', 
                    'ADBE', 'MU', '000660.KS', 
                    'LRCX', 'ADP', 'KLAC', 
                    'MELI', 'SNPS', '7974.T',
                    'MSTR', 'PANW', 'ADI', 
                    'DASH', 'CRWD', 'HOOD', 
                    'CDNS', '3690.HK', 'DELL', 
                    '6561.T', '2317.TW', 'SE', 
                    'INTC', 'NTES', 'COIN', 
                    'EQIX', 'ABNB', 'FI', 
                    '2454.TW', 'NET', 'CSU.TO',
                    'MRVL', '8035.T', '0981.HK', 
                    'PYPL', 'SNOW', 'CRWV', 
                    'ADSK', 'TEL', 'ROP', 
                    'FTNT', 'DELTA.BK', 'ADYEN.AS', 
                    'IFX.DE', 'NXPI', '6857.T', 
                    'CPNG', 'VEEV', 'JD', 
                    'DDOG', 'XYZ', 'GRMN', 
                    'TEAM', '1024.HK', 'EBAY', 
                    'DSY.PA', 'ZS', 'EA', 
                    '6701.T', '688256.SS', 'RDDT',]

# Downloading 10yr data (only needed for first run)
def download_data():
    data = yf.download(tickers, period='10y')
    data.to_csv("data/stock_data_10yr.csv")

# Only run when data is updated
def load_data():
    df = pd.read_csv("../data/stock_data_10yr.csv", index_col=0)
    df.index = pd.to_datetime(df.index)
    return df

def calculate(df, selected_stock, lag_days):
    returns = df.pct_change().dropna()
    corr = returns.corr()
    shifted_corr = returns[selected_stock].corr(returns['MSFT'].shift(lag_days))
    return corr, shifted_corr

# Data analysis
def analyze_relationship(returns, stock1, stock2, lag):
    stock2_lagged = returns[stock2].shift(lag)
    
    df = pd.DataFrame({
        stock1: returns[stock1],
        f'{stock2}_lagged': stock2_lagged
    }).dropna()
    
    corr = df[stock1].corr(df[f'{stock2}_lagged'])
    
    # Opposite direction moves count and frequency when stock1 dips
    stock1_down_stock2_up = ((df[stock1] < 0) & (df[f'{stock2}_lagged'] > 0)).sum()
    stock1_down_total = (df[stock1] < 0).sum()
    freq_percent = (stock1_down_stock2_up / stock1_down_total) * 100 if stock1_down_total > 0 else float('nan')
    
    # Same direction moves count and frequency when stock1 dips
    stock1_down_stock2_down = ((df[stock1] < 0) & (df[f'{stock2}_lagged'] < 0)).sum()
    same_dir_freq = (stock1_down_stock2_down / stock1_down_total) * 100 if stock1_down_total > 0 else float('nan')
    
    # Average % gap when they move opposite directions
    opposite_moves = df[(df[stock1] * df[f'{stock2}_lagged']) < 0]
    avg_gap = (opposite_moves[stock1] - opposite_moves[f'{stock2}_lagged']).abs().mean() * 100 if len(opposite_moves) > 0 else float('nan')
    
    # Win rate for stock2 when stock1 dips and they move opposite
    stock2_wins = (opposite_moves[f'{stock2}_lagged'] > 0).sum()
    win_rate = (stock2_wins / len(opposite_moves)) * 100 if len(opposite_moves) > 0 else float('nan')
    
    # Mean return for stock2 during opposite moves
    avg_profit = opposite_moves[f'{stock2}_lagged'].mean() * 100 if len(opposite_moves) > 0 else float('nan')
    
    # Decay rate
    from numpy import polyfit
    lags = list(range(1, lag+1))
    corrs = [df[stock1].corr(returns[stock2].shift(l)) for l in lags]
    if len(corrs) > 1:
        slope, _ = polyfit(lags, corrs, 1)
    else:
        slope = float('nan')
    
    return corr, freq_percent, same_dir_freq, avg_gap, win_rate, avg_profit, slope

# Stock rank generator
def compute_rankings(returns, tickers, max_lag=10):
    results = []
    for stock1 in tickers:
        for stock2 in tickers:
            if stock1 == stock2:
                continue
            for lag in range(1, max_lag + 1):
                try:
                    stock2_lagged = returns[stock2].shift(lag)
                    df = pd.DataFrame({
                        stock1: returns[stock1],
                        f'{stock2}_lagged': stock2_lagged
                    }).dropna()

                    if df.empty:
                        continue

                    corr = df[stock1].corr(df[f'{stock2}_lagged'])

                    # Frequency opposite moves when stock1 dips
                    stock1_down_stock2_up = ((df[stock1] < 0) & (df[f'{stock2}_lagged'] > 0)).sum()
                    stock1_down_total = (df[stock1] < 0).sum()
                    freq_percent = (stock1_down_stock2_up / stock1_down_total * 100) if stock1_down_total else 0

                    # Average % gap during opposite moves
                    opposite_moves = df[(df[stock1] * df[f'{stock2}_lagged']) < 0]
                    avg_gap = (opposite_moves[stock1] - opposite_moves[f'{stock2}_lagged']).abs().mean() * 100 if not opposite_moves.empty else 0

                    # Win rate
                    stock1_up_stock2_down = ((df[stock1] > 0) & (df[f'{stock2}_lagged'] < 0)).sum()
                    stock2_down_total = (df[f'{stock2}_lagged'] < 0).sum()
                    win_rate = (stock1_up_stock2_down / stock2_down_total * 100) if stock2_down_total else 0

                    # Avg profit
                    win_cases = df[(df[stock1] > 0) & (df[f'{stock2}_lagged'] < 0)]
                    avg_profit = win_cases[stock1].mean() * 100 if not win_cases.empty else 0

                    # Same direction moves
                    same_moves = ((df[stock1] > 0) & (df[f'{stock2}_lagged'] > 0)).sum() + \
                                 ((df[stock1] < 0) & (df[f'{stock2}_lagged'] < 0)).sum()
                    freq_same_percent = (same_moves / len(df) * 100) if len(df) else 0

                    # Final score formula
                    score = (corr * 0.275) + (freq_percent * 0.065) + (avg_gap * 0.05) \
                            + (win_rate * 0.275) + (avg_profit * 0.1) - (freq_same_percent * 0.225)

                    results.append({
                        'Stock1': stock1,
                        'Stock2': stock2,
                        'Lag': lag,
                        'Corr': corr,
                        'Freq%': freq_percent,
                        'AvgGap': avg_gap,
                        'WinRate': win_rate,
                        'AvgProfit': avg_profit,
                        'FreqSame%': freq_same_percent,
                        'Score': score
                    })
                except Exception as e:
                    print(f"Error processing {stock1}-{stock2} lag {lag}: {e}")

    return pd.DataFrame(results)

# Get top 50 rankings per lag day
def generate_rankings(returns):
    results = []
    stocks = returns.columns.tolist()

    for lag in range(1, 11):
        for stock1, stock2 in itertools.permutations(stocks, 2):
            try:
                corr, freq_percent, same_dir_freq, avg_gap, win_rate, avg_profit, freq_same_percent = analyze_relationship(
                    returns, stock1, stock2, lag
                )

                score = (corr * 0.275) + (freq_percent * 0.065) + (avg_gap * 0.05) + \
                        (win_rate * 0.275) + (avg_profit * 0.1) - (freq_same_percent * 0.15) - (same_dir_freq * 0.085)

                results.append({
                    "lag": lag,
                    "stock1": stock1,
                    "stock2": stock2,
                    "corr": corr,
                    "freq_percent": freq_percent,
                    "same_dir_freq": same_dir_freq,
                    "avg_gap": avg_gap,
                    "win_rate": win_rate,
                    "avg_profit": avg_profit,
                    "freq_same_percent": freq_same_percent,
                    "score": score
                })

            except Exception as e:
                print(f"Error for {stock1} vs {stock2} lag {lag}: {e}")

    df = pd.DataFrame(results)
    df.to_csv("../data/lag_rankings.csv", index=False)

# Creates top 50 list per lag day
def final_rankings(raw_file="../data/lag_rankings.csv", output_file="../data/lag_rankings_clean.csv"):
    df = pd.read_csv(raw_file)
    top50_list = []

    for lag in range(1, 11):
        lag_df = df[df["Lag"] == lag]
        top50 = lag_df.nlargest(50, "Score")
        top50_list.append(top50)

    top50_df = pd.concat(top50_list, ignore_index=True)
    top50_df.to_csv(output_file, index=False)
    return top50_df

def main():
    # Uncomment top 3 to reset all data
    # download_data()  # Download fresh 10-year stock dataset

    df = load_data()
    returns = df.pct_change(fill_method=None).dropna()

    # compute_rankings()  # Recalculate and save rankings for all lag values
    # generate_rankings(returns)  # Uncomment to regenerate rankings
    # top50_df = final_rankings() # Uncomment to rerank

    top50_df = pd.read_csv("../data/lag_rankings_clean.csv")

    # GUI
    root = tk.Tk()
    notebook = ttk.Notebook(root)
    notebook.grid(row=0, column=0, columnspan=2)

    # Analysis tab
    frame_analysis = ttk.Frame(notebook)
    notebook.add(frame_analysis, text="Analysis")

    # Rankings tab
    frame_rankings = ttk.Frame(notebook)
    notebook.add(frame_rankings, text="Rankings")
    root.title("Inverse Lead/Lag Engine")

    ttk.Label(frame_rankings, text="Enter Lag Day (1-10):").grid(row=0, column=0, padx=10, pady=5)
    lag_entry_rankings_var = tk.StringVar()
    lag_entry_rankings = ttk.Entry(frame_rankings, textvariable=lag_entry_rankings_var)
    lag_entry_rankings.grid(row=0, column=1, padx=10, pady=5)
    
    def validate_lag_rankings(new_value):
        return new_value.isdigit() and 1 <= int(new_value) <= 10 or new_value == ""
    vcmd_rankings = (root.register(validate_lag_rankings), '%P')
    lag_entry_rankings.config(validate='key', validatecommand=vcmd_rankings)
    
    # Text box for displaying top 50
    top50_text = tk.Text(frame_rankings, width=80, height=20)
    top50_text.grid(row=2, column=0, columnspan=2, padx=10, pady=10)
    
    # Button to load top 50 for entered  lag
    def show_top50():
        try:
            lag = int(lag_entry_rankings_var.get())
            df = pd.read_csv("../data/lag_rankings_clean.csv")  # Top 50 cleaned file
            lag_df = df[df["Lag"] == lag]
            top50 = lag_df.nlargest(50, "Score")
    
            top50_text.delete(1.0, tk.END)
            top50_text.insert(tk.END, f"Top 50 rankings for lag {lag}:\n\n")
            for idx, (_, row) in enumerate(top50.iterrows(), start=1):
                top50_text.insert(tk.END, f"{idx}. {row['Stock1']} -> {row['Stock2']} | Score: {row['Score']:.2f}\n")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    show_button = ttk.Button(frame_rankings, text="Show Top 50", command=show_top50)
    show_button.grid(row=1, column=0, columnspan=2, pady=5)
    
    # Stock 1
    ttk.Label(frame_analysis, text="Select Stock:").grid(row=0, column=0, padx=10, pady=5)
    stock_var = tk.StringVar(value=tickers[0])
    stock_combo = ttk.Combobox(frame_analysis, values=tickers, textvariable=stock_var, state="readonly")
    stock_combo.grid(row=0, column=1, padx=10, pady=5)

    # Stock 2
    ttk.Label(frame_analysis, text="Compare With:").grid(row=1, column=0, padx=10, pady=5)
    compare_var = tk.StringVar(value=tickers[1])
    compare_combo = ttk.Combobox(frame_analysis, values=tickers, textvariable=compare_var, state="readonly")
    compare_combo.grid(row=1, column=1, padx=10, pady=5)
    
    # Lag days
    ttk.Label(frame_analysis, text="Lag Days:").grid(row=2, column=0, padx=10, pady=5)
    lag_var = tk.StringVar(value="1")
    lag_entry = ttk.Entry(frame_analysis, textvariable=lag_var)
    lag_entry.grid(row=2, column=1, padx=10, pady=5)

    def validate_lag(new_value):
        return new_value.isdigit() or new_value == ""
    vcmd = (root.register(validate_lag), '%P')
    lag_entry.config(validate='key', validatecommand=vcmd)
    
    # Results
    result_text = tk.Text(frame_analysis, width=60, height=15)
    result_text.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

    def on_run():
        stock1 = stock_var.get()
        stock2 = compare_var.get()
        lag = int(lag_var.get())
        try:
            corr, freq_percent, same_dir_freq, avg_gap, win_rate, avg_profit, decay_rate = analyze_relationship(returns, stock1, stock2, lag)
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, f"Analysis for {stock1} vs {stock2} lagged by {lag} days:\n\n")
            result_text.insert(tk.END, f"Correlation coefficient: {corr:.4f}\n")
            result_text.insert(tk.END, f"Frequency of opposite moves when {stock1} dips: {freq_percent:.2f}%\n")
            result_text.insert(tk.END, f"Frequency of same direction moves when {stock1} dips: {same_dir_freq:.2f}%\n")
            result_text.insert(tk.END, f"Average % gap during opposite moves: {avg_gap:.2f}%\n")
            result_text.insert(tk.END, f"Win rate for {stock2} during opposite moves: {win_rate:.2f}%\n")
            result_text.insert(tk.END, f"Average profit % for {stock2} during opposite moves: {avg_profit:.2f}%\n")
            result_text.insert(tk.END, f"Decay rate of correlation: {decay_rate:.6f}\n")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # Run button
    run_button = ttk.Button(frame_analysis, text="Run Analysis", command=on_run)
    run_button.grid(row=3, column=0, columnspan=2, pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
