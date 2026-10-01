import numpy as np
import pandas as pd

def calculate_metrics(results, initial_cash=1000000.0):
    df = results.copy()
    df["portfolio_value"] = df["cash"] + df["inventory"] * df["mid_price"]
    df["pnl"] = df["portfolio_value"] - initial_cash
    step_s = np.median(np.diff(df["time"].to_numpy())) / 1000
    steps_per_year = 365 * 24 * 3600 / step_s
    returns = df["portfolio_value"].pct_change().dropna()
    std = returns.std()
    sharpe = np.nan if (np.isnan(std) or std == 0) else (returns.mean() / std * np.sqrt(steps_per_year))
    df["inventory_exposure"] = df["inventory"].abs() * df["mid_price"]
    running_max = df["portfolio_value"].cummax()
    drawdown = df["portfolio_value"] - running_max
    drawdown_pct = df["portfolio_value"] / running_max - 1
    n_bid = int(df["bid_filled"].sum())
    n_ask = int(df["ask_filled"].sum())
    return {
        "final_pnl": df["pnl"].iloc[-1],
        "sharpe_ratio": sharpe,
        "max_inventory": df["inventory"].abs().max(),
        "average_inventory": df["inventory"].abs().mean(),
        "final_inventory": df["inventory"].iloc[-1],
        "max_monetary_exposure": df["inventory_exposure"].max(),
        "average_monetary_exposure": df["inventory_exposure"].mean(),
        "max_drawdown": drawdown.min(),
        "max_drawdown_pct": drawdown_pct.min(),
        "n_bid_fills": n_bid,
        "n_ask_fills": n_ask,
        "fills_per_quote": (n_bid + n_ask) / len(df),
    }

def _to_ms(series):
    """
    Int64 ms epoch, whether `series` is already a raw ms integer column or a
    datetime64 column (in any resolution: ns/us/ms). Casting to datetime64[ms]
    first forces the unit before converting to int64.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.astype("datetime64[ms]").astype("int64").to_numpy()
    return series.to_numpy().astype(np.int64)


def backtest(HR_df, LR_df, strategy, initial_cash=1000000.0, initial_inventory=0.0, order_size=0.1):
    LR_df = LR_df.dropna(subset=["volatility"]).reset_index()   # FIXED: was reset_index(drop=True)
    HR_df = HR_df.sort_values("event_time").reset_index(drop=True)

    lr_t = _to_ms(LR_df["sampletime"])
    lr_mid = LR_df["mid_price"].to_numpy()
    lr_vol = LR_df["volatility"].to_numpy()
    lr_spread = LR_df["spread"].to_numpy()

    hr_t = _to_ms(HR_df["event_time"])  # FIXED: event_time may already be datetime-like
    hr_bid = HR_df["best_bid"].to_numpy()
    hr_ask = HR_df["best_ask"].to_numpy()

    end_time = lr_t[-1]
    cash, inventory = initial_cash, initial_inventory
    rows = []
    for i in range(len(LR_df) - 1):
        t0, t1 = lr_t[i], lr_t[i + 1]
        time_horizon = (end_time - t0) / 1000
        bid_quote, ask_quote = strategy(
            mid_price=lr_mid[i], volatility=lr_vol[i], spread=lr_spread[i],
            inventory=inventory, time_horizon=time_horizon,
        )
        low = np.searchsorted(hr_t, t0, side="right")
        high = np.searchsorted(hr_t, t1, side="right")
        bid_filled = ask_filled = False
        bid_fill_time = ask_fill_time = np.nan
        if high > low:
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
            "time": t1, "quote_time": t0, "bid_quote": bid_quote, "ask_quote": ask_quote,
            "bid_filled": bid_filled, "ask_filled": ask_filled,
            "bid_fill_time": bid_fill_time, "ask_fill_time": ask_fill_time,
            "cash": cash, "inventory": inventory, "mid_price": lr_mid[i + 1],
        })
    results = pd.DataFrame(rows)   # FIXED: was pd.DataFrame(results)
    return results