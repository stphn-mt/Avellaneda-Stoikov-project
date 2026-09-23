import pandas as pd
from datetime import datetime, timedelta
import tqdm
import numpy as np

"""
Take the time-series data, and derive other import time series data that trading models will use
(e.g. volatility, mid-prices, spread)
"""

def analyse(market, time_res):
    df = market.copy()

    df["datetime"] = pd.to_datetime(df["event_time"], unit="ms")
    df = df.set_index("datetime")

    # mid prices
    df["mid_price"] = (df["best_bid"] + df["best_ask"]) / 2
    df["mid_price"] = df["mid_price"].resample(time_res).last()
    df["spread"] = df["best_ask"] - df["best_bid"]
    df["spread"] = df["spread"].resample(time_res).last()

    # Log returns
    df["log_return"] = np.log(df["mid_price"] / df["mid_price"].shift(1))
    df["log_return"] = df["log_return"].resample(time_res).sum()

    # Rolling 1-second volatility
    df["volatility"] = (df["log_return"].rolling(time_res).std())

# we calculate OFI seperately after we compare "inventory-aware" vs naive model results
def calculate_ofi(df, time_res):

    bid_price_prev = df["best_bid"].shift(1)
    ask_price_prev = df["best_ask"].shift(1)

    bid_qty_prev = df["best_bid_qty"].shift(1)
    ask_qty_prev = df["best_ask_qty"].shift(1)

    df["ofi"] = (
        (df["best_bid"] >= bid_price_prev) * df["best_bid_qty"]
        - (df["best_bid"] <= bid_price_prev) * bid_qty_prev
        - (df["best_ask"] <= ask_price_prev) * df["best_ask_qty"]
        + (df["best_ask"] >= ask_price_prev) * ask_qty_prev
    )
    df["ofi"].resample(time_res).sum()

    return df["ofi"]
