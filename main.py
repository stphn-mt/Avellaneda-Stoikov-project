from backtest import backtest, calculate_metrics
from reconstruct import reconstruct
from analyse import analysis
import matplotlib as plt
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
#     index=False
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
    next_cash = 100000
    for resolution in RESOLUTIONS:
        for hour in HOURS:
            print(f"processing {hour.name}")
            timeseries = pd.read_parquet(hour)

            market = reconstruct(timeseries) #take raw data and turn into time series (HIGH RES)
            market_analysis = analysis(market, resolution) # use time series to derive time series (LOW RES)

            results = backtest(market, market_analysis, initial_cash=next_cash) #quotes, fills, cash, inventory, mid_price,
            metrics = calculate_metrics(results) # pandas DF

            record_data(hour, metrics, resolution)

            all_results.append({"time_res": resolution,
                                "hour": hour,
                                "results": results,
                                "metrics": metrics,
                                })
            
            
            next_cash = results["cash"].iloc[-1] # continue trading into the next hour
            # the loop will be the backtest
            # loop:
            # current inventory + low res data + high res data -> sim fill + update P&L, inventory 
            # 
            # calculate other metrics: max drawdown, sharpe, inventory exposure.

def plot_data(all_results):
    # plot_pnl(all_results)
    # plot_inventory(all_results)
    # plot_drawdown(all_results)
    # plot_quotes(...)
          
def record_data(hour, metrics, resolution): #either compare each hour (24 samples) or compare different time resolution performances
    output_dir = OUTPUT_DIR / resolution
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{hour.stem}.parquet"
    metrics.to_parquet(output_file)
    return None


if __name__ == '__main__':
    main()