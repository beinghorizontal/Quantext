"""
TPO Market Profile End-of-Day Visualization Script

This script generates a Time Price Opportunity (TPO) Market Profile chart for
end-of-day analysis. It downloads intraday price data and creates an interactive
Plotly visualization with TPO data, Value Area, Point of Control, and contextual
information.

Features:
    - Automatic data download from yfinance
    - Customizable resampling frequency
    - Dynamic Value Area calculation
    - Previous day context visualization
    - Market data export to text file
    - Interactive Plotly chart with hover information

Author: Alex ( https://www.youtube.com/@quantext )
Date: 2025-11-14
"""

from datetime import timedelta

import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from plotly.offline import plot

from tpo_helper_v2 import (
    get_context,
    get_dayrank,
    get_ibrank,
    get_mean,
    get_rf,
    get_ticksize,
)

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Data Source Configuration
NIFTY_TICKER = "^NSEI"  # NSE Nifty 50 Index ticker
TPO_SPACING = 2  # Additional spacing between TPO levels
DATA_PERIOD = "7d"  # Historical period to download ("7d", "30d", "1y", etc.)
DATA_INTERVAL = "1m"  # Intraday interval ("1m", "5m", "15m", "30m", "60m")

# NOTE: To use local CSV data instead of yfinance:
# Uncomment the lines below and adjust the file path
# filePath = 'd:/anaconda/Scripts/niftyf.csv'
# data = pd.read_csv(filePath, header=0)
# data['DateTime'] = pd.to_datetime(data['DateTime'], format='%Y-%m-%d %H:%M:%S')
# data = data.set_index('DateTime', drop=True, inplace=False)

# Chart Configuration
RESAMPLING_FREQ = 30  # Resampling frequency in minutes (30 min optimal for TPO)
AVERAGE_LENGTH = 7  # Number of days for calculating mean values
DAYS_TO_DISPLAY = 7  # Number of recent days to show on chart
CHART_MODE = "tpo"  # "tpo" for TPO mode, "vol" for Volume mode
# NOTE: TPO mode is recommended for Indian markets
# Volume mode is better for global markets as Indian volume data is unreliable

# Output Configuration
MARKET_DATA_FILE = "e:/envs/market_data.txt"  # File to export market data
OUTPUT_IMAGE_FILE = "e:/envs/nifty_tpo_plot.png"  # PNG export location
OUTPUT_IMAGE_WIDTH = 1920  # Image width in pixels
OUTPUT_IMAGE_HEIGHT = 1920  # Image height in pixels
SYMBOL_NAME = "NiftyF"  # Symbol display name

# ============================================================================
# DATA PREPARATION
# ============================================================================

# Download intraday data
print(f"Downloading {NIFTY_TICKER} data for {DATA_PERIOD} period...")
data = yf.download(
    tickers=NIFTY_TICKER, period=DATA_PERIOD, interval=DATA_INTERVAL, ignore_tz=True
)

# Standardize column names
data_columns = ["Open", "High", "Low", "Close", "Volume"]
data.columns = data_columns

print(f"Downloaded {len(data)} records")
print(f"Latest data:\n{data.tail()}")

# Calculate tick size and apply spacing
tick_size = get_ticksize(data, freq=RESAMPLING_FREQ)
tick_size = tick_size + TPO_SPACING

# Calculate mean values for session analysis
mean_values = get_mean(data, avglen=AVERAGE_LENGTH, freq=RESAMPLING_FREQ)
session_hours = mean_values["session_hr"]

# ============================================================================
# DATA TRANSFORMATION
# ============================================================================

# Apply rotational factor calculation
data = get_rf(data.copy())

# Resample data to desired frequency (30 min is optimal for TPO)
print(f"\nResampling data to {RESAMPLING_FREQ} minute intervals...")
df_resampled = data.copy()
df_resampled["datetime"] = df_resampled.index
df_resampled = df_resampled.resample(f"{RESAMPLING_FREQ}min").agg(
    {
        "datetime": "last",
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
        "rf": "sum",
    }
)
df_resampled = df_resampled.dropna()

# Filter data based on days_to_display parameter
latest_datetime = df_resampled.index[-1]
start_date = latest_datetime - timedelta(DAYS_TO_DISPLAY)
df_resampled = df_resampled[(df_resampled.index.date > start_date.date())]

print(f"Displaying last {len(df_resampled)} resampled bars")

# ============================================================================
# TPO & MARKET PROFILE ANALYSIS
# ============================================================================

# Split dataframe by trading day
daily_df_list = [group[1] for group in df_resampled.groupby(df_resampled.index.date)]

# Generate market profile context (TPO/Volume distribution)
print("Calculating market profile context...")
df_context = get_context(
    df_resampled,
    freq=RESAMPLING_FREQ,
    ticksize=tick_size,
    style=CHART_MODE,
    session_hr=session_hours,
)
df_mp_list = df_context[0]  # Market Profile for each day
df_distribution = df_context[1]  # Price distribution

# Calculate daily rankings and strength metrics
df_day_ranking = get_dayrank(df_distribution.copy(), mean_values)
daily_ranking = df_day_ranking[0]
daily_breakdown = df_day_ranking[1]

# Extract daily strength and highs/lows
daily_power_raw = daily_ranking.power1  # Non-normalized day strength
daily_power = daily_ranking.power  # Normalized day strength for visual markers
daily_high_list = daily_ranking.highd
daily_low_list = daily_ranking.lowd

# Calculate IB (Initial Balance) context
# Note: IB is the first 1 hour of session - not applicable for 24x7 instruments
print("Calculating Initial Balance context...")
ib_context = get_ibrank(mean_values, daily_ranking)
ib_power_raw = ib_context[0].ibpower1  # Non-normalized IB strength
ib_power = ib_context[0].IB_power  # Normalized IB strength
ib_breakdown = ib_context[1]
ib_high_list = ib_context[0].ibh
ib_low_list = ib_context[0].ibl

# ============================================================================
# EXPORT MARKET DATA
# ============================================================================

# Clear previous data file
print(f"Exporting market data to {MARKET_DATA_FILE}...")
with open(MARKET_DATA_FILE, "w") as f:
    pass

# ============================================================================
# CREATE PLOTLY VISUALIZATION
# ============================================================================

print("Creating interactive chart...")

# Initialize figure with candlestick chart
fig = go.Figure(
    data=[
        go.Candlestick(
            x=df_resampled["datetime"],
            open=df_resampled["Open"],
            high=df_resampled["High"],
            low=df_resampled["Low"],
            close=df_resampled["Close"],
            showlegend=False,
            name=SYMBOL_NAME,
            opacity=0.8,
        )
    ]
)

# ============================================================================
# ADD TPO LETTERS AND MARKET PROFILE DATA
# ============================================================================

# Iterate through each trading day and add TPO visualization
for day_index in range(len(df_mp_list)):
    daily_data = daily_df_list[day_index].copy()
    df_market_profile = df_mp_list[day_index]
    current_rank = daily_ranking.iloc[day_index]
    previous_rank = daily_ranking.iloc[day_index - 1] if day_index > 0 else None

    # Set date identifier
    df_market_profile["i_date"] = current_rank.date

    # Color TPO letters based on Value Area
    # Green if within Value Area, White otherwise
    df_market_profile["color"] = np.where(
        np.logical_and(
            df_market_profile["close"] > current_rank.vallist,
            df_market_profile["close"] < current_rank.vahlist,
        ),
        "green",
        "white",
    )

    df_market_profile = df_market_profile.set_index("i_date", inplace=False)

    # Add TPO letters to chart
    fig.add_trace(
        go.Scatter(
            x=df_market_profile.index,
            y=df_market_profile.close,
            mode="text",
            name=str(df_market_profile.index[0]),
            text=df_market_profile.alphabets,
            showlegend=False,
            textposition="top right",
            textfont=dict(family="verdana", size=18, color=df_market_profile.color),
        )
    )

    # Add daily context information (only from day 1 onwards)
    if day_index > 0:
        # Format Low Value Nodes list
        lvn_list_str = list(map(str, current_rank.lvnlist))

        # Create detailed hover information text
        context_text = (
            f"<br />Date: {current_rank.date}"
            f"<br />VAH: {int(current_rank.vahlist)}"
            f"<br />POC: {int(current_rank.poclist)}"
            f"<br />VAL: {int(current_rank.vallist)}"
            f"<br />yVAH: {int(previous_rank.vahlist)}"
            f"<br />yVAL: {int(previous_rank.vallist)}"
            f"<br />yPOC: {int(previous_rank.poclist)}"
            f"<br />Open: {int(daily_data.iloc[0]['Open'])}"
            f"<br />High: {int(current_rank.highd)}"
            f"<br />Low: {int(current_rank.lowd)}"
            f"<br />Close: {int(current_rank.close)}"
        )

        # Add context box above each day's chart
        fig.add_trace(
            go.Scatter(
                x=df_market_profile.index,
                y=[daily_data["High"].max() * 1.0005],  # Position with slight offset
                mode="text",
                text=[context_text],
                textposition="top right",
                textfont=dict(size=16, color="white"),
                showlegend=False,
            )
        )

        # Prepare market data for export
        market_data = {
            "Date": current_rank.date,
            "VAH": int(current_rank.vahlist),
            "POC": int(current_rank.poclist),
            "VAL": int(current_rank.vallist),
            "Yesterday's VAH": int(previous_rank.vahlist),
            "Yesterday's VAL": int(previous_rank.vallist),
            "Yesterday's POC": int(previous_rank.poclist),
            "Open": int(daily_data.iloc[0]["Open"]),
            "High": int(current_rank.highd),
            "Low": int(current_rank.lowd),
            "Close": int(current_rank.close),
            "Daily_Range": int(current_rank.ranged),
            "Daily_Rotation_factor": int(current_rank.rfd),
            "IBR_High": int(current_rank.ibh),
            "IBR_Low": int(current_rank.ibl),
            "LVNs": lvn_list_str,
        }

        # Write market data to file
        with open(MARKET_DATA_FILE, "a") as f:
            for key, value in market_data.items():
                f.write(f"{key}: {value}\n")
            f.write("---------------------\n")

# ============================================================================
# ADD LAST TRADED PRICE INDICATOR
# ============================================================================

# Get the last price and determine color (green if above POC, magenta if below)
last_price = int(daily_data.iloc[-1]["Close"])
last_price_color = "lightgreen" if last_price >= current_rank.poclist else "magenta"

fig.add_trace(
    go.Scatter(
        x=[daily_data.iloc[-1]["datetime"]],
        y=[daily_data.iloc[-1]["Close"]],
        mode="text",
        name="last traded price",
        text=[f"last {last_price}"],
        textposition="bottom right",
        textfont=dict(size=16, color=last_price_color),
        showlegend=False,
    )
)

# ============================================================================
# CONFIGURE CHART LAYOUT AND STYLING
# ============================================================================

# Set axis colors
fig.layout.xaxis.color = "white"
fig.layout.yaxis.color = "white"
fig.layout.autosize = True

# Configure Y-axis (Price axis)
fig.update_yaxes(
    title_text="Nifty",
    tickformat="d",
    title_font=dict(size=18, color="white"),
    tickfont=dict(size=12, color="white"),
    showgrid=False,
    # Add 0.5% extra space at the top for text visibility
    range=[df_resampled["Low"].min(), df_resampled["High"].max() * 1.005],
)

# Configure X-axis (Time axis)
fig.update_xaxes(
    showgrid=False,
    zeroline=False,
    rangeslider_visible=False,
    showticklabels=False,
    color="white",
    type="category",
    tickangle=90,
    dtick=30,  # Control spacing between ticks
)

# Configure overall layout
fig.update_layout(
    paper_bgcolor="black",  # Background color
    plot_bgcolor="black",  # Plot area background
    autosize=True,
    uirevision=True,
)

# ============================================================================
# DISPLAY AND EXPORT RESULTS
# ============================================================================

# Display interactive chart in browser
print("Displaying chart...")
plot(fig, auto_open=True)
fig.show()

# Export chart as PNG image
# NOTE: Requires kaleido package - install with: pip install kaleido
print(f"Saving chart to {OUTPUT_IMAGE_FILE}...")
fig.write_image(OUTPUT_IMAGE_FILE, width=OUTPUT_IMAGE_WIDTH, height=OUTPUT_IMAGE_HEIGHT)

print("Complete!")


