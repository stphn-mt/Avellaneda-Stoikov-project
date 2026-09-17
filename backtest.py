import numpy as np
import pandas as pd

from models import naive_strategy, as_strategy

market_data = pd.read_parquet(
    "data/derived_quantities.parquet"
)
def simulate_fill(cash, inventory, bidq, askq, best_bid, best_ask, bid_active, ask_active, order_size=0.001):
    bid_filled = False
    ask_filled = False
    # Check whether bid fills only if bid is active
    if bid_active:
        if best_ask <= bidq:

            # We buy order_size units
            cash -= bidq * order_size
            inventory += order_size

            bid_filled = True
    # Check whether ask fills only if ask is active
    if ask_active:
        if best_bid >= askq:

            # We sell order_size units
            cash += askq * order_size
            inventory -= order_size

            ask_filled = True
    return cash, inventory, bid_filled, ask_filled

def calculate_metrics(results, initial_cash=1000000.0):
    # inventory exposure 
    # mark-to-market Pnl
    results["portfolio_value"] = (results["cash"]+ results["inventory"] * results["mid_price"])
    results["pnl"] = (results["portfolio_value"] - initial_cash)
    results["strategy_returns"] = (results["portfolio_value"].pct_change())

    returns = results["strategy_returns"].dropna()
    # sharpe
    if returns.std() == 0:
        sharpe = np.nan
    else:
        sharpe = returns.mean() / returns.std()

    max_inventory = results["inventory"].abs().max()
    average_inventory = results["inventory"].abs().mean()
    results["inventory_exposure"] = (results["inventory"].abs() * results["mid_price"])
    max_monetary_exposure = results["inventory_exposure"].max()
    # drawdown
    results["running_max"] = (results["portfolio_value"].cummax())
    results["drawdown"] = (results["portfolio_value"] - results["running_max"])
    results["drawdown_pct"] = (results["portfolio_value"] / results["running_max"] - 1)

    max_drawdown = results["drawdown"].min()
    max_drawdown_pct = results["drawdown_pct"].min()

    return {
        "sharpe_ratio": sharpe,
        "max_inventory": max_inventory,
        "average_inventory": average_inventory,
        "max_monetary_exposure": max_monetary_exposure,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
    }

def backtest(df, strategy, initial_cash=1000000.0):

    cash = initial_cash
    inventory = 0.0
    results = []
    next_quote_time = df["event_time"].iloc[0]
    bid_active = False
    ask_active = False
    df = df.dropna(subset=["volatility"]).reset_index(drop=True)
    for i in range(len(df) - 1):

        current = df.iloc[i]
        future = df.iloc[i + 1]

        mid_price = current["mid_price"]
        volatility = current["volatility"]
        spread = current["spread"]

        # find time_horizon
        time_horizon = (df["event_time"].iloc[-1]-current["event_time"]) / 1000 #seconds
        # Generate our quotes at 1s resolution
        if current["event_time"] >= next_quote_time:
            bid_quote, ask_quote = strategy(
                mid_price=mid_price,
                volatility=volatility,
                spread=spread,
                inventory=inventory,
                time_horizon=time_horizon,
            )
            bid_active = True
            ask_active = True
            next_quote_time = current["event_time"] + 1000 # milliseconds, anchored to current time

        # default is not filled
        bid_filled = False
        ask_filled = False

        if bid_active or ask_active:
            cash, inventory, bid_filled, ask_filled = simulate_fill(
                                                                cash, 
                                                                inventory, 
                                                                bid_quote, 
                                                                ask_quote, 
                                                                future["best_bid"], 
                                                                future["best_ask"],
                                                                bid_active=bid_active,
                                                                ask_active=ask_active,
                                                            )
            # Remove filled quotes
            if bid_filled:
                bid_active = False

            if ask_filled:
                ask_active = False

        results.append({
            "event_time": future["event_time"],
            "bid_quote": bid_quote,
            "ask_quote": ask_quote,
            "bid_filled": bid_filled,
            "ask_filled": ask_filled,
            "cash": cash,
            "inventory": inventory,
            "mid_price": future["mid_price"],
        })

    results = pd.DataFrame(results)
    return results
naive_results = backtest(
    market_data,
    naive_strategy,
)

as_results = backtest(
    market_data,
    as_strategy,
)

naive_metrics = calculate_metrics(naive_results)
as_metrics = calculate_metrics(as_results)
print("naive model \n", pd.DataFrame([naive_metrics]))
print("Avellenada-Stoikov model \n", pd.DataFrame([as_metrics]))

# print(as_results[[
#     "event_time",
#     "bid_quote",
#     "ask_quote",
#     "bid_filled",
#     "ask_filled",
#     "cash",
#     "inventory",
# ]].head(10))

print("Bid fills:", as_results["bid_filled"].sum(), naive_results["bid_filled"].sum())
print("Ask fills:", as_results["ask_filled"].sum(), naive_results["ask_filled"].sum())
print("Final inventory:", as_results["inventory"].iloc[-1], naive_results["inventory"].iloc[-1])
print("Max inventory:", as_results["inventory"].abs().max(), naive_results["inventory"].abs().max())
print("Final PnL:", as_results["pnl"].iloc[-1], naive_results["pnl"].iloc[-1])