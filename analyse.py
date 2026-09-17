import pandas as pd
from datetime import datetime, timedelta
import tqdm
import numpy as np

market_ts = pd.read_parquet(
    "data/reconstructed_orderbook.parquet"
)

df = market_ts.copy()
"""
find volatility, returns, mid prices etc. from market_ts
"""
df["datetime"] = pd.to_datetime(
    df["event_time"],
    unit="ms"
)

# Use datetime as the index
df = df.set_index("datetime")

# mid prices
df["mid_price"] = (df["best_bid"] + df["best_ask"]) / 2
df["spread"] = df["best_ask"] - df["best_bid"]

# Log returns
df["log_return"] = np.log(df["mid_price"] / df["mid_price"].shift(1))

# Rolling 1-second volatility
df["volatility"] = (df["log_return"].rolling("1s").std())

# df = df.dropna()

df.to_parquet(
    "data/derived_quantities.parquet",
    index=False
)