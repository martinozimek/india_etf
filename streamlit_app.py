import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# === Paths ===
etf_dir = os.path.join("data")  # ETF CSVs live here
gdp_path = os.path.join("data", "processed_gdp_quarterly.csv")

# === Functions ===
def load_etf_csv(path):
    df = pd.read_csv(path, skiprows=[1])  # skip metadata row
    df.columns = df.columns.str.strip()
    if 'Date' not in df.columns or 'Price' not in df.columns:
        raise ValueError(f"Expected 'Date' and 'Price' columns in {path}, got: {df.columns}")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df.dropna(subset=['Date', 'Price'], inplace=True)
    df = df.set_index('Date').resample('QE').mean().reset_index()
    return df

def load_gdp_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['GDP_USD'] = pd.to_numeric(df['GDP_USD'], errors='coerce')
    df.dropna(subset=['Date', 'GDP_USD'], inplace=True)
    df = df.set_index('Date').resample('QE').mean().reset_index()
    return df

# === Load GDP Data ===
gdp_df = load_gdp_csv(gdp_path)

# === Streamlit UI ===
st.title("India ETF vs GDP Correlation Explorer")

correlations = {}
fig_norm, ax_norm = plt.subplots(figsize=(9, 3.5))
fig_corr, ax_corr = plt.subplots(figsize=(9, 3.5))

# === Process Each ETF ===
for file in sorted(f for f in os.listdir(etf_dir) if f.endswith("_daily.csv")):
    if not file.endswith(".csv"):
        continue
    etf_name = file.replace("_daily.csv", "")
    try:
        etf_df = load_etf_csv(os.path.join(etf_dir, file))
    except Exception as e:
        st.warning(f"Skipping {file}: {e}")
        continue

    merged = pd.merge(etf_df, gdp_df, on="Date", how="inner")
    if merged.empty:
        st.warning(f"No overlapping dates found for {etf_name}. Skipping.")
        continue

    # Calculate correlation
    pearson_corr = merged["Price"].corr(merged["GDP_USD"], method="pearson")
    spearman_corr = merged["Price"].corr(merged["GDP_USD"], method="spearman")
    correlations[etf_name] = {"pearson": pearson_corr, "spearman": spearman_corr}

    # Normalized price & GDP plot
    norm_price = merged["Price"] / merged["Price"].iloc[0]
    norm_gdp = merged["GDP_USD"] / merged["GDP_USD"].iloc[0]
    ax_norm.plot(merged["Date"], norm_price, label=f"{etf_name} (ETF)")
    ax_norm.plot(merged["Date"], norm_gdp, linestyle="--", alpha=0.5, label=f"{etf_name} (GDP)")

    # Rolling correlations
    rolling_window = 8  # quarters
    rolling_pearson = merged["Price"].rolling(rolling_window).corr(merged["GDP_USD"])
    ax_corr.plot(merged["Date"], rolling_pearson, label=f"{etf_name}")

# === Show Normalized Trends Plot ===
ax_norm.set_title("Normalized ETF Prices vs GDP Over Time", fontsize=14)
ax_norm.set_ylabel("Normalized Value", fontsize=12)
ax_norm.legend(loc="upper left", fontsize="small")
ax_norm.tick_params(axis='x', labelsize=10)
ax_norm.tick_params(axis='y', labelsize=10)
fig_norm.tight_layout()
col1, col2, col3 = st.columns([1, 4, 1])  # Center with margins
with col2:
    st.pyplot(fig_norm)


# === Show Rolling Correlations ===
ax_corr.set_title("Rolling Pearson Correlation (8Q Window)", fontsize=14)
ax_corr.set_ylabel("Pearson r", fontsize=12)
ax_corr.set_xlabel("Date", fontsize=12)
ax_corr.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax_corr.legend(loc="upper right", fontsize="small")
ax_corr.tick_params(axis='x', labelsize=10)
ax_corr.tick_params(axis='y', labelsize=10)
fig_corr.tight_layout()
scol1, col2, col3 = st.columns([1, 4, 1])  # Center with margins
with col2:
    st.pyplot(fig_norm)


# === Show Summary Table ===
st.subheader("Correlation Table")
cor_df = pd.DataFrame.from_dict(correlations, orient="index").round(3)
st.dataframe(cor_df)
