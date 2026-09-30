import numpy as np
import pandas as pd

"""
Backtest and compare a naive benchmark strategy with Avellaneda Stoikov strategy.
"""
# market_data = pd.read_parquet(
#     "data/derived_quantities.parquet"
# )
# #first iteration of simulating fills, unused since slower than current
# def simulate_fill(cash, inventory, bidq, askq, best_bid, best_ask, bid_active, ask_active, order_size=0.001):
#     bid_filled = False
#     ask_filled = False
#     # Check whether bid fills only if bid is active
#     if bid_active:
#         if best_ask <= bidq:

#             # We buy order_size units
#             cash -= bidq * order_size
#             inventory += order_size

#             bid_filled = True
#     # Check whether ask fills only if ask is active
#     if ask_active:
#         if best_bid >= askq:

#             # We sell order_size units
#             cash += askq * order_size
#             inventory -= order_size

#             ask_filled = True
#     return cash, inventory, bid_filled, ask_filled

def calculate_metrics(results, initial_cash=1000000.0):
    df = results.copy()  # don't mutate the caller's frame

    df["portfolio_value"] = df["cash"] + df["inventory"] * df["mid_price"]
    df["pnl"] = df["portfolio_value"] - initial_cash

    # Sharpe on the LR grid, annualised (crypto trades 24/7)
    step_s = np.median(np.diff(df["time"].to_numpy())) / 1000
    steps_per_year = 365 * 24 * 3600 / step_s
    returns = df["portfolio_value"].pct_change().dropna()
    std = returns.std()
    sharpe = np.nan if (np.isnan(std) or std == 0) else (
        returns.mean() / std * np.sqrt(steps_per_year)
    )
    
    # inventory / exposure
    df["inventory_exposure"] = df["inventory"].abs() * df["mid_price"]

    # drawdown
    running_max = df["portfolio_value"].cummax()
    drawdown = df["portfolio_value"] - running_max
    drawdown_pct = df["portfolio_value"] / running_max - 1

    n_bid = int(df["bid_filled"].sum())
    n_ask = int(df["ask_filled"].sum())

    return {
        "final_pnl":                df["pnl"].iloc[-1],
        "sharpe_ratio":             sharpe,
        "max_inventory":            df["inventory"].abs().max(),
        "average_inventory":        df["inventory"].abs().mean(),
        "final_inventory":          df["inventory"].iloc[-1],
        "max_monetary_exposure":    df["inventory_exposure"].max(),
        "average_monetary_exposure": df["inventory_exposure"].mean(),
        "max_drawdown":             drawdown.min(),
        "max_drawdown_pct":         drawdown_pct.min(),
        "n_bid_fills":              n_bid,
        "n_ask_fills":              n_ask,
        "fills_per_quote":          (n_bid + n_ask) / len(df),
    }

def backtest(HR_df, LR_df, strategy, initial_cash=1000000.0, initial_inventory=0.0, order_size=0.1):
    """
    HR_df is the high-resolution data; use for simulating fills
    LR_df is the low-resolution data; use for strategy decisions.
    be careful about using index (datetime) from LR_df, and event_time from HR_df
    """
    LR_df = LR_df.dropna(subset=["volatility"]).reset_index(drop=True)
    HR_df = HR_df.sort_values("event_time").reset_index(drop=True)

    lr_t      = LR_df["sampletime"].to_numpy()
    lr_mid    = LR_df["mid_price"].to_numpy()
    lr_vol    = LR_df["volatility"].to_numpy()
    lr_spread = LR_df["spread"].to_numpy()

    hr_t   = HR_df["event_time"].to_numpy()
    hr_bid = HR_df["best_bid"].to_numpy()
    hr_ask = HR_df["best_ask"].to_numpy()

    end_time  = lr_t[-1]
    cash, inventory = initial_cash, initial_inventory
    rows = []
    for i in range(len(LR_df) - 1):
        t0, t1 = lr_t[i], lr_t[i + 1]
        time_horizon = (end_time - t0) / 1000  # ms -> s
        # 1) quote using only information available at t0
        bid_quote, ask_quote = strategy(
            mid_price=lr_mid[i],
            volatility=lr_vol[i],
            spread=lr_spread[i],
            inventory=inventory,
            time_horizon=time_horizon,
        )

        # 2) HR window strictly after t0, up to and including t1
        low = np.searchsorted(hr_t, t0, side="right")
        high = np.searchsorted(hr_t, t1, side="right")

        bid_filled = ask_filled = False
        bid_fill_time = ask_fill_time = np.nan
        if high > low:
            # our bid fills once the market ask trades down to it,
            # our ask fills once the market bid trades up to it
            bid_hits = hr_ask[low:high] <= bid_quote
            ask_hits = hr_bid[low:high] >= ask_quote

            if bid_hits.any():
                bid_filled = True
                bid_fill_time = hr_t[low + bid_hits.argmax()]
                cash -= bid_quote * order_size
                inventory += order_size

            if ask_hits.any():
                ask_filled = True
                ask_fill_time = hr_t[low + ask_hits.argmax()]
                cash += ask_quote * order_size
                inventory -= order_size

        rows.append({
            "time":          t1,    # when cash/inventory/mid are measured
            "quote_time":    t0,    # when quote is made     
            "bid_quote":     bid_quote,
            "ask_quote":     ask_quote,
            "bid_filled":    bid_filled,
            "ask_filled":    ask_filled,
            "bid_fill_time": bid_fill_time,
            "ask_fill_time": ask_fill_time,
            "cash":          cash,
            "inventory":     inventory,
            "mid_price":     lr_mid[i + 1],
            "mtm_P&L":        cash + inventory * lr_mid[i + 1],  # mark-to-market
        })

    results = pd.DataFrame(results)
    return results