import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# === File Paths ===
etf_dir = Path(r"C:\Users\Ozimek\Documents\india_etf_data")
gdp_file = Path(r"C:\Users\Ozimek\Documents\processed_gdp_quarterly.csv")

etf_files = [
    "FLIN_daily.csv", "EPI_daily.csv", "SMIN_daily.csv",
    "PIN_daily.csv", "NFTY_daily.csv", "INDY_daily.csv", "INDA_daily.csv"
]

# === Load and process GDP data ===
gdp_df = pd.read_csv(gdp_file)
gdp_df.columns = gdp_df.columns.str.strip()
gdp_df['Date'] = pd.to_datetime(gdp_df['Date'], errors='coerce')
gdp_df['GDP_USD'] = pd.to_numeric(gdp_df['GDP_USD'], errors='coerce')
gdp_df = gdp_df.dropna(subset=['Date', 'GDP_USD'])
gdp_df = gdp_df.set_index('Date').resample('Q').mean()

# === Store correlations ===
results = []

# === Loop over all ETFs ===
for file in etf_files:
    etf_path = etf_dir / file
    etf_name = file.replace("_daily.csv", "")

    try:
        etf_df = pd.read_csv(etf_path, skiprows=[1])
        etf_df.columns = etf_df.columns.str.strip()

        if 'Date' not in etf_df.columns or 'Price' not in etf_df.columns:
            print(f"[SKIPPED] {etf_name}: Missing expected columns.")
            continue

        etf_df['Date'] = pd.to_datetime(etf_df['Date'], errors='coerce')
        etf_df['Price'] = pd.to_numeric(etf_df['Price'], errors='coerce')
        etf_df = etf_df.dropna(subset=['Date', 'Price'])
        etf_df = etf_df.set_index('Date').resample('Q').mean()

        # Align GDP and ETF data
        merged = pd.merge(etf_df[['Price']], gdp_df[['GDP_USD']], left_index=True, right_index=True, how='inner')
        merged = merged.dropna()

        if len(merged) < 8:
            print(f"[SKIPPED] {etf_name}: Only {len(merged)} overlapping quarters.")
            continue

        # Pearson correlation
        corr = merged['Price'].corr(merged['GDP_USD'])
        results.append({
            "ETF": etf_name,
            "Pearson_Correlation": round(corr, 4),
            "Quarters_Used": len(merged),
            "Start": merged.index.min().date(),
            "End": merged.index.max().date()
        })

        # === Plot 1: Normalized Time Series ===
        merged['Price_norm'] = merged['Price'] / merged['Price'].iloc[0]
        merged['GDP_norm'] = merged['GDP_USD'] / merged['GDP_USD'].iloc[0]
        plt.figure(figsize=(10, 5))
        plt.plot(merged.index, merged['Price_norm'], label=f"{etf_name} (Normalized)")
        plt.plot(merged.index, merged['GDP_norm'], '--', label="GDP (Normalized)")
        plt.title(f"{etf_name} vs GDP: Normalized Series")
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

        # === Plot 2: Scatter Plot ===
        plt.figure(figsize=(6, 5))
        plt.scatter(merged['GDP_USD'], merged['Price'], alpha=0.7)
        plt.title(f"{etf_name}: Price vs GDP")
        plt.xlabel("GDP (USD)"); plt.ylabel("ETF Price")
        plt.grid(True); plt.tight_layout(); plt.show()

        # === Plot 3: Rolling Pearson + Spearman ===
        window = 8
        rolling_pearson = merged['Price'].rolling(window=window).corr(merged['GDP_USD'])

        def spearman_corr(x):
            price_rank = x.rank()
            gdp_rank = merged.loc[x.index, 'GDP_USD'].rank()
            return price_rank.corr(gdp_rank)

        rolling_spearman = merged['Price'].rolling(window=window).apply(spearman_corr, raw=False)

        plt.figure(figsize=(10, 4))
        plt.plot(merged.index, rolling_pearson, label="Pearson", color='steelblue')
        plt.plot(merged.index, rolling_spearman, label="Spearman", color='darkorange', linestyle='--')
        plt.axhline(0, linestyle=':', color='gray')
        plt.title(f"{etf_name}: Rolling Correlation with GDP (Window={window})")
        plt.ylabel("Correlation Coefficient")
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

        # === Plot 4: Cumulative Growth Overlay ===
        merged['Price_pct'] = merged['Price'].pct_change().add(1).cumprod()
        merged['GDP_pct'] = merged['GDP_USD'].pct_change().add(1).cumprod()
        plt.figure(figsize=(10, 5))
        plt.plot(merged.index, merged['Price_pct'], label=f"{etf_name} Growth")
        plt.plot(merged.index, merged['GDP_pct'], '--', label="GDP Growth")
        plt.title(f"{etf_name}: Cumulative Growth vs GDP")
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

        # === Plot 5: Scatter of % Changes ===
        merged['Price_change'] = merged['Price'].pct_change()
        merged['GDP_change'] = merged['GDP_USD'].pct_change()
        plt.figure(figsize=(6, 5))
        plt.scatter(merged['GDP_change'], merged['Price_change'], alpha=0.6)
        plt.axhline(0, color='gray', linestyle=':')
        plt.vlines(0, ymin=merged['Price_change'].min(), ymax=merged['Price_change'].max(), color='gray', linestyle=':')
        plt.title(f"{etf_name}: % Change Correlation")
        plt.xlabel("GDP % Change"); plt.ylabel("ETF % Change")
        plt.grid(True); plt.tight_layout(); plt.show()

    except Exception as e:
        print(f"[ERROR] {etf_name}: {e}")

# === Final Plot: Summary Bar Chart of Correlation ===
df_results = pd.DataFrame(results).sort_values(by="Pearson_Correlation", ascending=False)

plt.figure(figsize=(10, 6))
bars = plt.barh(df_results['ETF'], df_results['Pearson_Correlation'], color='teal', edgecolor='black')
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
             f"{width:.2f}", va='center')
plt.axvline(0, color='gray', linestyle='--')
plt.title("Pearson Correlation of India ETFs vs GDP (USD)")
plt.xlabel("Correlation Coefficient")
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()
