"""
TPO Market Profile Helper Functions - Version 2

A comprehensive library of functions for calculating Time Price Opportunity (TPO)
Market Profile metrics. This module provides tools for:

- Market Profile generation and TPO/Volume distribution analysis
- Value Area calculation (VAH, POC, VAL)
- Day type classification based on distribution patterns
- Initial Balance (IB) analysis and context
- Daily ranking and strength calculations
- Statistical analysis of trading metrics

The Market Profile concept is based on the Steidlmayer method of market analysis,
which organizes price data by time spent at each price level, revealing market
structure and buyer/seller activity patterns.

Author: Alex ( https://www.youtube.com/@quantext )
Date: 2025-11-14
License: MIT License
"""

import math

import numpy as np
import pandas as pd

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Price levels for calculation
DEFAULT_FREQ = 30  # Default resampling frequency in minutes
DEFAULT_TICKSIZE = 10  # Default tick size for price levels
DEFAULT_SESSION_HOURS = 6.5  # Default trading session length in hours

# Value Area calculation
VALUE_AREA_PERCENTAGE = 0.70  # Include 70% of TPOs in value area

# Normalization parameters for strength calculations
STRENGTH_MIN = 25  # Minimum normalized strength value
STRENGTH_MAX = 100  # Maximum normalized strength value


# ============================================================================
# PRIMARY TPO/MARKET PROFILE FUNCTIONS
# ============================================================================


def get_ticksize(data, freq=DEFAULT_FREQ):
    """
    Calculate optimal tick size based on rolling standard deviation of closing prices.

    Tick size represents the price level granularity for the market profile. It's
    calculated as half of the mean standard deviation of price changes over the most
    recent period.

    Args:
        data (pd.DataFrame): OHLCV price data with 'Close' column
        freq (int): Frequency for rolling std calculation in minutes (default: 30)

    Returns:
        int: Calculated tick size (minimum 0.2)

    Notes:
        - Uses 50% of most recent data for calculation
        - Ensures minimum tick size of 0.2 to avoid too-fine granularity

    Example:
        >>> tick_size = get_ticksize(data, freq=30)
        >>> print(f"Optimal tick size: {tick_size}")
    """
    # Sample 50% of most recent data for tick size calculation
    sample_size = int(len(data) / 2)
    recent_data = data.tail(sample_size).copy()

    # Calculate rolling standard deviation
    recent_data["tz"] = recent_data.Close.rolling(freq).std()
    recent_data = recent_data.dropna()

    # Tick size is half of mean standard deviation
    tick_size = np.ceil(recent_data["tz"].mean() * 0.50)

    # Apply minimum limit
    if tick_size < 0.2:
        tick_size = 0.2

    return int(tick_size)


def abc(session_hr=DEFAULT_SESSION_HOURS, freq=DEFAULT_FREQ):
    """
    Generate alphabet sequence for TPO character labeling.

    Each 30-minute (or specified frequency) period gets an alphabetic character.
    Sequences cycle through A-Z and a-z, then repeat as needed for longer sessions.

    Args:
        session_hr (float): Trading session length in hours (default: 6.5)
        freq (int): Bar frequency in minutes (default: 30)

    Returns:
        tuple: (alphabets list for each bar, string of timekeeper indices)

    Notes:
        - A session of 6.5 hours with 30-min bars = ~13 bars
        - Returns 52 unique characters (A-Z, a-z)
        - Repeats pattern if session requires more characters

    Example:
        >>> alphabets, timekeeper = abc(session_hr=6.5, freq=30)
        >>> print(f"Using {len(alphabets)} characters for session")
    """
    # Define alphabet sequence
    capital_letters = [
        " A",
        " B",
        " C",
        " D",
        " E",
        " F",
        " G",
        " H",
        " I",
        " J",
        " K",
        " L",
        " M",
        " N",
        " O",
        " P",
        " Q",
        " R",
        " S",
        " T",
        " U",
        " V",
        " W",
        " X",
        " Y",
        " Z",
    ]
    lowercase_letters = [x.lower() for x in capital_letters]
    alphabet_sequence = capital_letters + lowercase_letters

    # Calculate number of bars in session
    bars_in_session = math.ceil(session_hr * (60 / freq)) + 3

    # Extend alphabet sequence if needed
    if bars_in_session > 52:
        repetitions = int(np.ceil((bars_in_session - 52) / 52)) + 1
        alphabets = alphabet_sequence * repetitions
    else:
        alphabets = alphabet_sequence[0:bars_in_session]

    # Generate timekeeper indices (marks specific times during session)
    timekeeper_indices = [28, 31, 35, 40, 33, 34, 41, 44, 35, 52, 41, 40, 46, 27, 38]
    timekeeper_chars = []
    for idx in timekeeper_indices:
        timekeeper_chars.append(alphabet_sequence[idx - 1])
    timekeeper_string = "".join(timekeeper_chars)

    return alphabets, timekeeper_string


def tpo(
    dft_rs,
    freq=DEFAULT_FREQ,
    ticksize=DEFAULT_TICKSIZE,
    style="tpo",
    session_hr=DEFAULT_SESSION_HOURS,
):
    """
    Calculate Time Price Opportunity (TPO) market profile for a single trading day.

    This is the core function that generates market profile data including:
    - TPO count (or volume) at each price level
    - Value Area High/Low (VAH/VAL)
    - Point of Control (POC) - price with highest activity
    - Low Value Nodes (LVN) - prices with minimal activity
    - Balance Target - theoretical equilibrium price

    The Value Area is calculated starting from POC and expanding up/down to include
    70% of total TPOs or volume.

    Args:
        dft_rs (pd.DataFrame): Resampled 1-minute OHLCV data for a single day
        freq (int): Bar frequency in minutes (default: 30)
        ticksize (int): Price level granularity (default: 10)
        style (str): "tpo" for TPO count, "vol" for volume-based (default: "tpo")
        session_hr (float): Session length in hours (default: 6.5)

    Returns:
        dict: Market profile metrics including:
            - 'df': DataFrame with TPO distribution
            - 'vah': Value Area High
            - 'poc': Point of Control
            - 'val': Value Area Low
            - 'lvn': List of Low Value Nodes (single prints)
            - 'bal_target': Balance Target price
            Returns empty dict if insufficient data

    Notes:
        - Requires minimum bars (typically > 2 for 30-min frequency)
        - VAL extends from POC downward
        - VAH extends from POC upward
        - LVN identifies support/resistance with low activity

    Example:
        >>> mp = tpo(daily_data, freq=30, ticksize=5, style="tpo")
        >>> print(f"POC: {mp['poc']}, VAH: {mp['vah']}, VAL: {mp['val']}")
    """
    if len(dft_rs) > int(60 / freq):
        # Remove duplicate timestamps and reset index
        dft_rs = dft_rs.drop_duplicates("datetime")
        dft_rs = dft_rs.reset_index(inplace=False, drop=True)

        # Calculate cumulative high/low for breakout detection
        dft_rs["rol_mx"] = dft_rs["High"].cummax()
        dft_rs["rol_mn"] = dft_rs["Low"].cummin()
        dft_rs["ext_up"] = dft_rs["rol_mn"] > dft_rs["rol_mx"].shift(2)
        dft_rs["ext_dn"] = dft_rs["rol_mx"] < dft_rs["rol_mn"].shift(2)

        # Get alphabet sequence for this session
        alphabets = abc(session_hr, freq)[0]
        alphabets = alphabets[0 : len(dft_rs)]

        # Get price range
        high_price = dft_rs["High"].max()
        low_price = dft_rs["Low"].min()
        dft_rs["abc"] = alphabets

        # Calculate number of price levels to analyze
        num_price_levels = int(np.ceil((high_price - low_price) / ticksize))

        # Initialize lists to store TPO data
        alphabets_at_level = []
        tpo_count_at_level = []
        price_levels = []
        volume_sum_at_level = []

        # Iterate through each price level
        for level_idx in range(num_price_levels):
            price_level = low_price + (level_idx * ticksize)
            alphabets_hit = []
            tpo_count = []
            volume_count = []

            # Find TPOs and volume at this price level
            for bar_idx in range(len(dft_rs)):
                if (
                    price_level >= dft_rs["Low"][bar_idx]
                    and price_level < dft_rs["High"][bar_idx]
                ):
                    alphabets_hit.append(dft_rs["abc"][bar_idx])
                    tpo_count.append(1)
                    volume_count.append((dft_rs["Volume"][bar_idx]) / freq)

            alphabets_at_level.append("".join(alphabets_hit))
            tpo_count_at_level.append(sum(tpo_count))
            volume_sum_at_level.append(sum(volume_count))
            price_levels.append(price_level)

        # Create TPO distribution DataFrame
        dftpo = pd.DataFrame(
            {
                "close": price_levels,
                "alphabets": alphabets_at_level,
                "tpocount": tpo_count_at_level,
                "volsum": volume_sum_at_level,
            }
        )

        # Remove empty price levels
        dftpo["alphabets"].replace("", np.nan, inplace=True)
        dftpo = dftpo.dropna()
        dftpo = dftpo.reset_index(inplace=False, drop=True)

        # Sort by price (high to low) for easier processing
        dftpo = dftpo.sort_index(ascending=False)
        dftpo = dftpo.reset_index(inplace=False, drop=True)

        # Select metric based on style
        metric_column = "tpocount" if style == "tpo" else "volsum"

        # Find Point of Control (POC)
        max_count = dftpo[metric_column].max()
        dfmax_candidates = dftpo[dftpo[metric_column] == max_count].copy()

        # If multiple POCs, select one closest to mid-price
        mid_price = low_price + ((high_price - low_price) / 2)
        dfmax_candidates["distance_to_mid"] = abs(dfmax_candidates["close"] - mid_price)
        poc_idx = dfmax_candidates["distance_to_mid"].idxmin()
        poc = dfmax_candidates["close"].loc[poc_idx]
        poc_tpo = dftpo[metric_column].max()

        # Initialize Value Area calculation
        value_area = set([poc])  # Start with POC
        current_tpo_sum = poc_tpo
        total_tpo = dftpo[metric_column].sum()
        target_tpo = total_tpo * VALUE_AREA_PERCENTAGE

        # Find POC index in sorted DataFrame
        poc_index = dftpo[dftpo["close"] == poc].index[0]

        # Indices for expanding above and below POC
        # Note: DataFrame is sorted high-to-low, so up means lower index
        up_index = poc_index - 1  # Above POC
        down_index = poc_index + 1  # Below POC

        # Expand Value Area up and down
        while current_tpo_sum < target_tpo and (up_index >= 0 or down_index < len(dftpo)):
            above_sum = 0
            below_sum = 0

            # Sum next 2 price levels above POC
            if up_index >= 0 and up_index - 1 >= 0:
                above_sum = (
                    dftpo.iloc[up_index][metric_column]
                    + dftpo.iloc[up_index - 1][metric_column]
                )
            elif up_index >= 0:
                above_sum = dftpo.iloc[up_index][metric_column]

            # Sum next 2 price levels below POC
            if down_index < len(dftpo) and down_index + 1 < len(dftpo):
                below_sum = (
                    dftpo.iloc[down_index][metric_column]
                    + dftpo.iloc[down_index + 1][metric_column]
                )
            elif down_index < len(dftpo):
                below_sum = dftpo.iloc[down_index][metric_column]

            # Add prices with higher activity to Value Area
            if above_sum >= below_sum and up_index >= 0:
                if up_index >= 0:
                    value_area.add(dftpo.iloc[up_index]["close"])
                    current_tpo_sum += dftpo.iloc[up_index][metric_column]
                    up_index -= 1
                if up_index >= 0 and current_tpo_sum < target_tpo:
                    value_area.add(dftpo.iloc[up_index]["close"])
                    current_tpo_sum += dftpo.iloc[up_index][metric_column]
                    up_index -= 1
            else:
                if down_index < len(dftpo):
                    value_area.add(dftpo.iloc[down_index]["close"])
                    current_tpo_sum += dftpo.iloc[down_index][metric_column]
                    down_index += 1
                if down_index < len(dftpo) and current_tpo_sum < target_tpo:
                    value_area.add(dftpo.iloc[down_index]["close"])
                    current_tpo_sum += dftpo.iloc[down_index][metric_column]
                    down_index += 1

        # Extract Value Area boundaries
        vah = max(value_area)
        val = min(value_area)

        # Identify Low Value Nodes (LVN) - prices with minimal activity
        # Exclude outer edges and find isolated low-activity zones
        tpo_middle_section = dftpo[ticksize:-(ticksize)]["tpocount"]
        low_tpo_mask = tpo_middle_section <= 2
        low_tpo_indices = [
            idx for idx, val in zip(tpo_middle_section.index, low_tpo_mask) if val
        ]

        # Find LVNs separated by minimum distance
        min_distance = ticksize * 3
        lvn_list = []
        for idx, ex_idx in enumerate(low_tpo_indices[:-1:min_distance]):
            lvn_list.append(dftpo["close"].iloc[ex_idx])

        # Calculate Balance Target
        area_above_poc = dft_rs.High.max() - poc
        area_below_poc = poc - dft_rs.Low.min()

        # Prevent division issues
        if area_above_poc == 0:
            area_above_poc = 1
        if area_below_poc == 0:
            area_below_poc = 1

        balance_ratio = area_above_poc / area_below_poc

        # Balance target is where symmetry would be restored
        if balance_ratio >= 0:
            bal_target = poc - area_above_poc
        else:
            bal_target = poc + area_below_poc

        # Return market profile results
        market_profile = {
            "df": dftpo,
            "vah": round(vah, 2),
            "poc": round(poc, 2),
            "val": round(val, 2),
            "lvn": lvn_list,
            "bal_target": round(bal_target, 2),
        }
    else:
        # Insufficient data
        print(f"Insufficient bars for date {dft_rs['datetime'].iloc[0]}")
        market_profile = {}

    return market_profile


# ============================================================================
# CONTEXT AND RANKING FUNCTIONS
# ============================================================================


def get_context(
    df_hi,
    freq=DEFAULT_FREQ,
    ticksize=DEFAULT_TICKSIZE,
    style="tpo",
    session_hr=DEFAULT_SESSION_HOURS,
):
    """
    Generate market profile context for multiple trading days.

    Processes resampled OHLCV data and generates market profile metrics for each
    trading day, then compiles statistical distributions.

    Args:
        df_hi (pd.DataFrame): Resampled OHLCV data (ideally 30-minute bars)
        freq (int): Bar frequency in minutes (default: 30)
        ticksize (int): Price level granularity (default: 10)
        style (str): "tpo" or "vol" for metric selection (default: "tpo")
        session_hr (float): Session length in hours (default: 6.5)

    Returns:
        tuple: (market_profile_list, distribution_dataframe)
            - market_profile_list: List of DataFrames with TPO distributions
            - distribution_dataframe: Combined metrics for all days with columns:
                - date, maxtpo, tpocount, vahlist, poclist, vallist
                - btlist (balance target), lvnlist (low value nodes)
                - volumed, rfd (rotation factor), highd, lowd, ranged
                - ibh, ibl, ibvol, ibrf (Initial Balance metrics)
                - close (closing price)

    Notes:
        - Handles exceptions gracefully, returns empty lists if error occurs
        - Calculates Initial Balance (first hour) metrics
        - Includes cumulative volume and rotation factor

    Example:
        >>> mp_list, dist_df = get_context(df_resampled, freq=30, ticksize=5)
        >>> print(f"Analyzed {len(mp_list)} trading days")
    """
    try:
        # Split data by trading day
        daily_groups = [group[1] for group in df_hi.groupby(df_hi.index.date)]

        # Initialize storage lists
        market_profile_list = []
        max_tpo_list = []
        total_tpo_list = []
        vah_list = []
        poc_list = []
        val_list = []
        balance_target_list = []
        lvn_list = []

        date_list = []
        volume_list = []
        rotation_factor_list = []
        ib_volume_list = []
        ib_rf_list = []
        ib_high_list = []
        ib_low_list = []
        close_list = []
        high_list = []
        low_list = []
        range_list = []

        # Process each trading day
        for day_idx in range(len(daily_groups)):
            daily_data = daily_groups[day_idx].copy()

            # Convert numeric columns
            daily_data.iloc[:, 2:6] = daily_data.iloc[:, 2:6].apply(pd.to_numeric)
            daily_data = daily_data.reset_index(inplace=False, drop=True)

            # Calculate TPO market profile
            market_profile = tpo(daily_data, freq, ticksize, style, session_hr)
            mp_df = market_profile["df"]
            market_profile_list.append(mp_df)

            # Extract market profile metrics
            max_tpo_list.append(mp_df["tpocount"].max())
            total_tpo_list.append(mp_df["tpocount"].sum())
            vah_list.append(market_profile["vah"])
            poc_list.append(market_profile["poc"])
            val_list.append(market_profile["val"])
            balance_target_list.append(market_profile["bal_target"])
            lvn_list.append(market_profile["lvn"])

            # Extract OHLCV metrics
            date_list.append(daily_data.datetime[0])
            close_list.append(daily_data.iloc[-1]["Close"])
            low_list.append(daily_data.Low.min())
            high_list.append(daily_data.High.max())
            range_list.append(daily_data.High.max() - daily_data.Low.min())

            volume_list.append(daily_data.Volume.sum())
            rotation_factor_list.append(daily_data.rf.sum())

            # Calculate Initial Balance metrics (first 60 minutes = 1 hour)
            daily_data["cumsumvol"] = daily_data.Volume.cumsum()
            daily_data["cumsumrf"] = daily_data.rf.cumsum()
            daily_data["cumsumhigh"] = daily_data.High.cummax()
            daily_data["cumsumlow"] = daily_data.Low.cummin()

            ib_bar_count = int(60 / freq)
            ib_volume_list.append(daily_data.cumsumvol[ib_bar_count])
            ib_rf_list.append(daily_data.cumsumrf[ib_bar_count])
            ib_low_list.append(daily_data.cumsumlow[ib_bar_count])
            ib_high_list.append(daily_data.cumsumhigh[ib_bar_count])

        # Compile all metrics into distribution DataFrame
        distribution_df = pd.DataFrame(
            {
                "date": date_list,
                "maxtpo": max_tpo_list,
                "tpocount": total_tpo_list,
                "vahlist": vah_list,
                "poclist": poc_list,
                "vallist": val_list,
                "btlist": balance_target_list,
                "lvnlist": lvn_list,
                "volumed": volume_list,
                "rfd": rotation_factor_list,
                "highd": high_list,
                "lowd": low_list,
                "ranged": range_list,
                "ibh": ib_high_list,
                "ibl": ib_low_list,
                "ibvol": ib_volume_list,
                "ibrf": ib_rf_list,
                "close": close_list,
            }
        )

    except Exception as e:
        print(f"Error in get_context: {str(e)}")
        market_profile_list = []
        distribution_df = []

    return (market_profile_list, distribution_df)


def get_dayrank(dist_df, mean_val):
    """
    Rank trading days based on multiple strength factors.

    Analyzes market profile distribution and calculates day strength based on:
    - Value Area movement (VAH/VAL vs previous day)
    - Price levels (High/Low vs previous day)
    - Price position relative to Value Area
    - Low Value Nodes (single print zones)
    - Volume and Rotation Factor zscore normalization
    - Day type classification

    Args:
        dist_df (pd.DataFrame): Distribution DataFrame from get_context()
        mean_val (dict): Mean values dictionary from get_mean() containing:
            - rf_mean: Mean daily rotation factor
            - volume_mean: Mean daily volume

    Returns:
        tuple: (ranking_df, breakdown_df)
            - ranking_df: Original data plus calculated strength metrics
            - breakdown_df: Transposed breakdown of individual strength components

    Notes:
        - Day types: Trend Day (4), Trend Dist Day (3), Normal Variation (2), Neutral (1)
        - Power values normalized to 25-100 range
        - Negative power indicates down bias

    Example:
        >>> ranking_df, breakdown = get_dayrank(dist_df, mean_values)
        >>> print(f"Day strength: {ranking_df['power'].iloc[-1]}")
    """
    ranking_df = dist_df.copy()

    # ===== LVN ANALYSIS =====
    # Calculate Single Print strength (LVN analysis)
    lvn_list = ranking_df["lvnlist"].to_list()
    close_list = ranking_df["close"].to_list()
    lvn_power_list = []
    total_lvn = 0

    for close_price, lvn_prices in zip(close_list, lvn_list):
        if len(lvn_prices) == 0:
            lvn_score = 0
        else:
            for lvn_price in lvn_prices:
                delta = close_price - lvn_price
                lvn_score = 1 if delta >= 0 else -1
                total_lvn += lvn_score

        lvn_power_list.append(total_lvn)
        total_lvn = 0

    ranking_df["Single_Prints"] = lvn_power_list

    # ===== DAY TYPE CLASSIFICATION =====
    # Classify days based on TPO distribution relative to mean
    ranking_df["distr"] = ranking_df.tpocount / ranking_df.maxtpo
    distr_mean = math.floor(ranking_df.distr.mean())
    distr_std = math.floor(ranking_df.distr.std())

    # Assign day types with numeric values
    # Trend Day (4): High distribution, significant directional bias
    # Trend Distribution Day (3): Moderately high distribution
    # Normal Variation Day (2): Average distribution around mean
    # Neutral Day (1): Lower distribution, more consolidation

    ranking_df["daytype"] = ""
    ranking_df["daytype_num"] = 0

    # Trend Distribution Day
    mask_trend_dist = (ranking_df.distr >= distr_mean) & (
        ranking_df.distr < distr_mean + distr_std
    )
    ranking_df.loc[mask_trend_dist, "daytype"] = "Trend Distribution Day"
    ranking_df.loc[mask_trend_dist, "daytype_num"] = 3

    # Normal Variation Day
    mask_normal = (ranking_df.distr < distr_mean) & (
        ranking_df.distr >= distr_mean - distr_std
    )
    ranking_df.loc[mask_normal, "daytype"] = "Normal Variation Day"
    ranking_df.loc[mask_normal, "daytype_num"] = 2

    # Neutral Day
    mask_neutral = ranking_df.distr < distr_mean - distr_std
    ranking_df.loc[mask_neutral, "daytype"] = "Neutral Day"
    ranking_df.loc[mask_neutral, "daytype_num"] = 1

    # Trend Day
    mask_trend = ranking_df.distr > distr_mean + distr_std
    ranking_df.loc[mask_trend, "daytype"] = "Trend Day"
    ranking_df.loc[mask_trend, "daytype_num"] = 4

    # Apply bias (sign) based on close vs POC
    ranking_df["daytype_num"] = np.where(
        ranking_df.close >= ranking_df.poclist,
        ranking_df.daytype_num * 1,
        ranking_df.daytype_num * -1,
    )

    # ===== VOLUME AND ROTATION ANALYSIS =====
    # Z-score normalize volume
    rf_mean = mean_val["rf_mean"]
    vol_mean = mean_val["volume_mean"]

    ranking_df["vold_zscore"] = (ranking_df.volumed - vol_mean) / ranking_df.volumed.std(
        ddof=0
    )
    ranking_df["rfd_zscore"] = (abs(ranking_df.rfd) - rf_mean) / abs(ranking_df.rfd).std(
        ddof=0
    )

    # Normalize to 1-4 range
    norm_min, norm_max = 1, 4

    # Normalize rotation factor
    rf_min, rf_max = ranking_df.rfd_zscore.min(), ranking_df.rfd_zscore.max()
    ranking_df["norm_rf"] = (ranking_df.rfd_zscore - rf_min) / (rf_max - rf_min) * (
        norm_max - norm_min
    ) + norm_min

    # Normalize volume
    vol_min, vol_max = ranking_df.vold_zscore.min(), ranking_df.vold_zscore.max()
    ranking_df["norm_volume"] = (ranking_df.vold_zscore - vol_min) / (
        vol_max - vol_min
    ) * (norm_max - norm_min) + norm_min

    # Apply bias based on close vs POC
    ranking_df["Volume_Factor"] = np.where(
        ranking_df.close >= ranking_df.poclist,
        ranking_df.norm_volume * 1,
        ranking_df.norm_volume * -1,
    )
    ranking_df["Rotation_Factor"] = np.where(
        ranking_df.rfd >= 0,
        ranking_df.norm_rf * 1,
        ranking_df.norm_rf * -1,
    )

    # ===== COMPARATIVE METRICS =====
    # Compare current day metrics with previous day
    ranking_df["VAH_vs_yVAH"] = np.where(
        ranking_df.vahlist >= ranking_df.vahlist.shift(), 1, -1
    )
    ranking_df["VAL_vs_yVAL"] = np.where(
        ranking_df.vallist >= ranking_df.vallist.shift(), 1, -1
    )
    ranking_df["POC_vs_yPOC"] = np.where(
        ranking_df.poclist >= ranking_df.poclist.shift(), 1, -1
    )
    ranking_df["H_vs_yH"] = np.where(ranking_df.highd >= ranking_df.highd.shift(), 1, -1)
    ranking_df["L_vs_yL"] = np.where(ranking_df.lowd >= ranking_df.lowd.shift(), 1, -1)
    ranking_df["Close_vs_yCL"] = np.where(
        ranking_df.close >= ranking_df.close.shift(), 1, -1
    )

    # ===== CLOSE POSITION RELATIVE TO VALUE AREA =====
    # Evaluate where the day closed within Value Area
    ranking_df["CL>POC<VAH"] = np.where(
        (ranking_df.close >= ranking_df.poclist)
        & (ranking_df.close < ranking_df.vahlist),
        1,
        0,
    )
    ranking_df["CL<poc>val"] = np.where(
        (ranking_df.close < ranking_df.poclist)
        & (ranking_df.close >= ranking_df.vallist),
        -1,
        0,
    )
    ranking_df["CL<VAL"] = np.where(ranking_df.close < ranking_df.vallist, -2, 0)
    ranking_df["CL>=VAH"] = np.where(ranking_df.close >= ranking_df.vahlist, 2, 0)

    # ===== HANDLE NaN VALUES =====
    # Replace NaN with 0 for safe calculations
    ranking_df["Volume_Factor"] = np.where(
        ranking_df["Volume_Factor"].isnull(), 0, ranking_df["Volume_Factor"]
    )
    ranking_df["Rotation_Factor"] = np.where(
        ranking_df["Rotation_Factor"].isnull(), 0, ranking_df["Rotation_Factor"]
    )
    ranking_df["daytype_num"] = np.where(
        ranking_df["daytype_num"].isnull(), 0, ranking_df["daytype_num"]
    )

    # ===== CALCULATE TOTAL POWER =====
    # Weighted average of all strength factors
    strength_factors = [
        "VAH_vs_yVAH",
        "VAL_vs_yVAL",
        "POC_vs_yPOC",
        "H_vs_yH",
        "L_vs_yL",
        "Close_vs_yCL",
        "CL>POC<VAH",
        "CL<poc>val",
        "Single_Prints",
        "CL<VAL",
        "CL>=VAH",
        "Volume_Factor",
        "Rotation_Factor",
        "daytype_num",
    ]

    num_factors = len(strength_factors)
    factor_sum = sum(ranking_df[factor] for factor in strength_factors)
    ranking_df["power1"] = 100 * (factor_sum / num_factors)

    # Normalize power to 25-100 range
    power_min = abs(ranking_df.power1).min()
    power_max = abs(ranking_df.power1).max()
    ranking_df["power"] = (abs(ranking_df.power1) - power_min) / (
        power_max - power_min
    ) * (STRENGTH_MAX - STRENGTH_MIN) + STRENGTH_MIN

    ranking_df = ranking_df.round(2)

    # Create breakdown DataFrame for detailed analysis
    breakdown_df = ranking_df[strength_factors].transpose()
    breakdown_df = breakdown_df.round(2)

    return (ranking_df, breakdown_df)


def get_ibrank(mean_val, ranking):
    """
    Calculate Initial Balance (IB) strength and context.

    Analyzes Initial Balance metrics (first 60 minutes of trading) to assess
    early session strength and directional bias.

    Args:
        mean_val (dict): Mean values from get_mean()
        ranking (pd.DataFrame): Ranking DataFrame from get_dayrank()

    Returns:
        tuple: (ib_ranking_df, ib_breakdown_df)
            - ib_ranking_df: Ranking data with IB-specific metrics
            - ib_breakdown_df: Breakdown of IB strength components

    Notes:
        - IB represents first 60 minutes of trading session
        - Not applicable for 24x7 markets (crypto, forex)
        - Useful for session-based markets (stocks, index futures)

    Example:
        >>> ib_df, ib_breakdown = get_ibrank(mean_values, ranking_df)
        >>> print(f"IB Power: {ib_df['IB_power'].iloc[-1]}")
    """
    ib_ranking_df = ranking.copy()

    # Extract mean values
    ib_rf_mean = mean_val["ibrf_mean"]
    vol_mean = mean_val["volume_mean"]

    # ===== IB ROTATION FACTOR ANALYSIS =====
    ib_ranking_df["ibrf_zscore"] = (abs(ib_ranking_df.ibrf) - ib_rf_mean) / abs(
        ib_ranking_df.ibrf
    ).std(ddof=0)

    norm_min, norm_max = 1, 4

    # Normalize IB rotation factor
    ib_rf_min = ib_ranking_df.ibrf_zscore.min()
    ib_rf_max = ib_ranking_df.ibrf_zscore.max()
    ib_ranking_df["ibnorm_rf"] = (ib_ranking_df.ibrf_zscore - ib_rf_min) / (
        ib_rf_max - ib_rf_min
    ) * (norm_max - norm_min) + norm_min

    # Apply bias based on rotation direction
    ib_ranking_df["IB_RotationFactor"] = np.where(
        ib_ranking_df.ibrf >= 0,
        ib_ranking_df.ibnorm_rf * 1,
        ib_ranking_df.ibnorm_rf * -1,
    )
    ib_ranking_df["IB_RotationFactor"] = ib_ranking_df["IB_RotationFactor"].round(2)

    # ===== IB VOLUME ANALYSIS =====
    if ib_ranking_df.ibvol.isnull().values.any():
        ib_ranking_df["ibvol_zscore"] = (abs(ib_ranking_df.ibvol) - vol_mean) / abs(
            ib_ranking_df.ibvol
        ).std(ddof=0)
    else:
        ib_ranking_df["ibvol_zscore"] = 0

    # Normalize IB volume
    ib_vol_min = ib_ranking_df.ibvol_zscore.min()
    ib_vol_max = ib_ranking_df.ibvol_zscore.max()

    if ib_ranking_df.ibvol.isnull().values.any():
        ib_ranking_df["ibnorm_vol"] = (ib_ranking_df.ibvol_zscore - ib_vol_min) / (
            ib_vol_max - ib_vol_min
        ) * (norm_max - norm_min) + norm_min
    else:
        ib_ranking_df["ibnorm_vol"] = 0

    # Apply bias based on close vs POC
    ib_ranking_df["IB_Volume_factor"] = np.where(
        ib_ranking_df.close >= ib_ranking_df.poclist,
        ib_ranking_df.ibnorm_vol * 1,
        ib_ranking_df.ibnorm_vol * -1,
    )
    ib_ranking_df["IB_Volume_factor"] = ib_ranking_df["IB_Volume_factor"].round()

    # ===== IB POSITION RELATIVE TO VALUE AREA =====
    # Calculate IB midpoint
    ib_ranking_df["ibmid"] = ib_ranking_df.ibl + (
        (ib_ranking_df.ibh - ib_ranking_df.ibl) / 2
    )

    # Position IB within previous day's Value Area
    # 1/-1: Inside Value Area (bullish/bearish)
    # 2/-2: Above/Below Value Area but within day's range
    # 4/-4: Beyond previous day's high/low

    ib_ranking_df["IB>yPOC<yVAH"] = np.where(
        (ib_ranking_df.ibmid >= ib_ranking_df.poclist.shift())
        & (ib_ranking_df.ibmid < ib_ranking_df.vahlist.shift()),
        1,
        0,
    )

    ib_ranking_df["IB<yPOC>yVAL"] = np.where(
        (ib_ranking_df.ibmid >= ib_ranking_df.vallist.shift())
        & (ib_ranking_df.ibmid < ib_ranking_df.poclist.shift()),
        -1,
        0,
    )

    ib_ranking_df["IB>yVAH<yH"] = np.where(
        (ib_ranking_df.ibmid >= ib_ranking_df.vahlist.shift())
        & (ib_ranking_df.ibmid < ib_ranking_df.highd.shift()),
        2,
        0,
    )

    ib_ranking_df["IB<yVAL>yL"] = np.where(
        (ib_ranking_df.ibmid < ib_ranking_df.vallist.shift())
        & (ib_ranking_df.ibmid > ib_ranking_df.lowd.shift()),
        -2,
        0,
    )

    ib_ranking_df["IB>yH"] = np.where(
        ib_ranking_df.ibmid >= ib_ranking_df.highd.shift(), 4, 0
    )

    ib_ranking_df["IB<yL"] = np.where(
        ib_ranking_df.ibmid < ib_ranking_df.lowd.shift(), -4, 0
    )

    # ===== CALCULATE IB TOTAL POWER =====
    ib_factors = [
        "IB>yPOC<yVAH",
        "IB<yPOC>yVAL",
        "IB>yVAH<yH",
        "IB<yVAL>yL",
        "IB>yH",
        "IB<yL",
        "IB_Volume_factor",
        "IB_RotationFactor",
    ]

    num_ib_factors = len(ib_factors)
    ib_sum = sum(ib_ranking_df[factor] for factor in ib_factors)
    ib_ranking_df["ibpower1"] = 100 * (ib_sum / num_ib_factors)

    # Normalize IB power to 25-100 range
    ib_power_min = abs(ib_ranking_df.ibpower1).min()
    ib_power_max = abs(ib_ranking_df.ibpower1).max()
    ib_ranking_df["IB_power"] = (abs(ib_ranking_df.ibpower1) - ib_power_min) / (
        ib_power_max - ib_power_min
    ) * (STRENGTH_MAX - STRENGTH_MIN) + STRENGTH_MIN

    ib_ranking_df = ib_ranking_df.round(2)

    # Create breakdown DataFrame
    ib_breakdown_df = ib_ranking_df[ib_factors].transpose()

    return (ib_ranking_df, ib_breakdown_df)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def get_rf(df):
    """
    Calculate Rotation Factor (RF) - directional strength indicator.

    RF measures directional bias by comparing current bar with previous bar:
    - Close up: +1
    - High up: +1
    - Low up: +1
    Range: -3 to +3

    Positive RF indicates bullish pressure, negative indicates bearish pressure.

    Args:
        df (pd.DataFrame): OHLCV DataFrame with 'Close', 'High', 'Low' columns

    Returns:
        pd.DataFrame: Original dataframe with 'rf' column added

    Example:
        >>> df = get_rf(data)
        >>> print(f"Average daily RF: {df.groupby(df.index.date)['rf'].sum().mean()}")
    """
    df["cup"] = np.where(df["Close"] >= df["Close"].shift(), 1, -1)
    df["hup"] = np.where(df["High"] >= df["High"].shift(), 1, -1)
    df["lup"] = np.where(df["Low"] >= df["Low"].shift(), 1, -1)

    df["rf"] = df["cup"] + df["hup"] + df["lup"]
    df = df.drop(["cup", "lup", "hup"], axis=1)

    return df


def get_mean(data, avglen=30, freq=DEFAULT_FREQ):
    """
    Calculate mean values for volume, rotation factor, and Initial Balance metrics.

    Aggregates 1-minute data into daily candles and computes rolling averages
    for normalization purposes.

    Args:
        data (pd.DataFrame): 1-minute OHLCV data
        avglen (int): Number of days for rolling average (default: 30)
        freq (int): Bar frequency in minutes (default: 30)

    Returns:
        dict: Mean values dictionary:
            - volume_mean: Average daily volume
            - rf_mean: Average daily rotation factor (absolute)
            - volib_mean: Average Initial Balance volume
            - ibrf_mean: Average Initial Balance rotation factor
            - session_hr: Trading session length in hours

    Notes:
        - IB is calculated as first 60 minutes of session
        - RF values are absolute (unsigned) for comparison purposes
        - Session length determines IB bar count

    Example:
        >>> means = get_mean(data, avglen=7, freq=30)
        >>> print(f"Average volume: {means['volume_mean']}")
    """
    # Add rotation factor to data
    df_history = get_rf(data.copy())

    # Resample to daily and aggregate
    df_daily = df_history.resample("D").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
            "rf": "sum",
        }
    )
    df_daily = df_daily.dropna()

    # Calculate rolling averages
    volume_rolling = df_daily["Volume"].rolling(avglen).mean()
    volume_mean = volume_rolling.iloc[-1]

    rf_rolling = abs(df_daily["rf"]).rolling(avglen).mean()
    rf_mean = rf_rolling.iloc[-1]

    # Calculate session length (hours in first trading day)
    second_day = df_daily.index[1].date()
    first_day_mask = df_history.index.date < second_day
    first_day_data = df_history.loc[first_day_mask]
    session_hours = math.ceil(len(first_day_data) / 60)

    # ===== INITIAL BALANCE CALCULATION =====
    # Get first hour of session
    ib_start_time = df_history.index.time[0]
    ib_end_bar = int(freq * (60 / freq))  # First 60 minutes
    ib_end_time = df_history.index.time[ib_end_bar]

    df_ib = df_history.between_time(ib_start_time, ib_end_time)

    # Resample IB to daily and aggregate
    df_ib_daily = df_ib.resample("D").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
            "rf": "sum",
        }
    )
    df_ib_daily = df_ib_daily.dropna()

    # Calculate IB rolling averages
    ib_volume_rolling = df_ib_daily["Volume"].rolling(avglen).mean()
    ib_volume_mean = ib_volume_rolling.iloc[-1]

    ib_rf_rolling = abs(df_ib_daily["rf"]).rolling(avglen).mean()
    ib_rf_mean = ib_rf_rolling.iloc[-1]

    # Return all mean values
    mean_values = {
        "volume_mean": volume_mean,
        "rf_mean": rf_mean,
        "volib_mean": ib_volume_mean,
        "ibrf_mean": ib_rf_mean,
        "session_hr": session_hours,
    }

    return mean_values
