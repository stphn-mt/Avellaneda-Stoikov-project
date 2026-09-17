import pandas as pd
from datetime import datetime, timedelta
import tqdm
import numpy as np

events = pd.read_parquet(
    "data/binance_spot/2026-08-31/00/BTCUSDT_orderbook.parquet"
)

df = events.copy()

# Reconstruct orderbook from events
# Load orderbook events

# Convert price and quantity to float
df["price"] = df["price"].astype(float)
df["quantity"] = df["quantity"].astype(float)


# Reconstruct the final orderbook
orderbook = {
    'bid': {},
    'ask': {}
}

was_prev_snapshot = False
# Create numpy arrays for faster processing
prices = np.array(df["price"])
quantities = np.array(df["quantity"])
event_types = np.array(df["event_type"])
sides = np.array(df["side"])
times = np.array(df["event_time"])
#time series
time_series = []
best_bids = []
best_asks = []
bb_qty = []
ba_qty = []

# derived qty's
mid_prices = []
# spread = []
# bid_depth = []
# ask_depth = []
# order_book_imbalance = []

prev_time = 0
for i in tqdm.tqdm(range(len(df))):
    price = prices[i]
    quantity = quantities[i]
    event_type = event_types[i]
    side = sides[i]
    time = times[i]
    if event_type == "snapshot":
        if not was_prev_snapshot:
            # Clear the orderbook as this is the start of a new snapshot
            orderbook = {
                'bid': {},
                'ask': {}
            }
        was_prev_snapshot = True
    else:
        was_prev_snapshot = False

    if quantity == 0:
        # Remove the price level if quantity is zero
        orderbook[side].pop(price, None)
    else:
        # add data to time series
        if event_type == "update":
            if  prev_time != time: #if multiple updates at a time, let all updates be added to orderbook, then calc bb/ba
                best_bids.append(max(orderbook["bid"]))
                bb_qty.append(orderbook["bid"][best_bids[-1]])

                best_asks.append(min(orderbook["ask"]))
                ba_qty.append(orderbook["ask"][best_asks[-1]])
                time_series.append(time) #record the time of best bid and ask
        # Update the orderbook with the new price level
        orderbook[side][price] = quantity

    prev_time = time #always update to new time.




market_data = pd.DataFrame({
    "event_time": time_series,
    "best_bid": best_bids,
    "best_bid_qty": bb_qty,
    "best_ask": best_asks,
    "best_ask_qty": ba_qty,
})

market_data.to_parquet(
    "data/reconstructed_orderbook.parquet",
    index=False
)

