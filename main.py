from backtest import backtest, calculate_metrics
from reconstruct import reconstruct
from models import naive_strategy, as_strategy
from analyse import analyse
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


# TAKEN FROM START OF RECONSTRUCT.PY
# events = pd.read_parquet(
#     "data/raw/BTCUSDT_orderbook_20260918_00.parquet"
# )

# df = events.copy() #pandas dataframe

#TAKEN FROM END OF RECONSTRUCT.PY

# market_data.to_parquet(
#     "data/timeseries/hour00.parquet",
#     index=False -> we should save our index because it holds datetime!
# )
"""
from reconstruct: read parquet of raw data, convert into market data time series we can use
"""


RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/timeseries")
HOURS = sorted(RAW_DIR.glob("BTCUSDT_orderbook_20260918_*.parquet"))  
RESOLUTIONS = ["100ms", "1s", "5s", "1min"]

all_results = {}
def main():
    """
    backtest:
    current inventory + low res data + high res data -> sim fill + update P&L, inventory 
    
    calculate other metrics: max drawdown, sharpe, inventory exposure.
    """
    next_cash_as = 1000000.0
    next_cash_naive = 1000000.0
    next_inventory_naive = 0
    next_inventory_as = 0
    for resolution in RESOLUTIONS:
        for hour in HOURS:
            print(f"processing {hour.name}")
            timeseries = pd.read_parquet(hour)

            market = reconstruct(timeseries) #take raw data and turn into time series (HIGH RES)
            market_analysis = analyse(market, resolution) # use time series to derive time series (LOW RES)

            naive_results = backtest(market, market_analysis, naive_strategy, initial_cash=next_cash_naive, initial_inventory=next_inventory_naive) #quotes, fills, cash, inventory, mid_price,
            as_results = backtest(market, market_analysis, as_strategy, initial_cash=next_cash_as, initial_inventory=next_inventory_as) #we need to fix which cash goes where.

            naive_metrics = calculate_metrics(naive_results) # pandas DF
            as_metrics = calculate_metrics(as_results)

            record_data(hour, naive_metrics, resolution)
            record_data(hour, as_metrics, resolution)

            all_results.append({"time_res": resolution,
                                "hour": hour,
                                "results": results,
                                "metrics": metrics,
                                })
            # continue trading into the next hour
            next_cash_naive = naive_results["cash"].iloc[-1] 
            next_inventory_naive = naive_results["inventory"].iloc[-1]
            next_cash_as = as_results["cash"].iloc[-1]
            next_inventory_as = as_results["inventory"].iloc[-1]

    plot_data(all_results)

def plot_data(all_results):
    # plot_pnl(all_results)
    plt.plot(all_results["cash"], all_results["time_res"]) #change cash to something else?
    # plot_inventory(all_results)
    plt.plot(all_results["inventory"], all_results["time_res"])
    # plot_drawdown(all_results)
    plt.plot(all_results["max_drawdown"], all_results["time_res"])
    # plot_quotes(...)
          
def record_data(hour, metrics, resolution): #either compare each hour (24 samples) or compare different time resolution performances
    output_dir = OUTPUT_DIR / resolution
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{hour.stem}.parquet"
    metrics.to_parquet(output_file)
    return None


if __name__ == '__main__':
    main()