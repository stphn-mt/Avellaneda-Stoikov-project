from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyse import analyse
from backtest import backtest, calculate_metrics
from models import as_strategy, naive_strategy
from reconstruct import reconstruct

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/timeseries")
HOURS = sorted(RAW_DIR.glob("BTCUSDT_orderbook_20260918_*.parquet"))
RESOLUTIONS = ["100ms", "1s", "5s", "1min"]
COLOURS = {"naive": "tab:orange", "as": "tab:blue"}

STRATEGIES = {
    "naive": naive_strategy,
    "as": as_strategy,
}


# helpers
def _prep(df, initial_cash):
    """Add a datetime column and mark-to-market P&L."""
    df = df.copy()
    df["dt"] = pd.to_datetime(df["time"], unit="ms")
    df["pnl"] = df["cash"] + df["inventory"] * df["mid_price"] - initial_cash
    return df


def _thin(df, max_points=5000):
    """Downsample for full-day plots (100ms data is ~864k rows over 24 hours)."""
    step = max(1, len(df) // max_points) #works out step so that total points returned=maxpoints
    return df.iloc[::step] #returns the df at each step


def _grid(n):
    """ helps to format multiple plots into 2 columns"""
    ncols = 2 if n > 1 else 1
    nrows = -(-n // ncols) # round "down" so double negative flips to round up
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 3.5 * nrows),
                             squeeze=False)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False) # hide any leftover space
    return fig, axes


def _acf(x, nlags):
    """
    Autocorrelation via FFT. always returns
    exactly nlags+1 values, so caller's x-axis (np.arange(nlags+1))
    never mismatches: lags beyond what series supports come back as NaN
    (matplotlib skips them) instead of silently shortening array.
    """
    x = np.asarray(x, dtype=float) # make timeseries an np array
    n = len(x)
    out = np.full(nlags + 1, np.nan) # fill output with nan's first
    if n < 2:
        return out # exit if not enough data
    x = x - x.mean() #normalise around mean
    max_lag = min(nlags, n - 1) # constrict max lags
    f = np.fft.rfft(x, 2 * n) 
    ac = np.fft.irfft(f * np.conj(f))[: max_lag + 1] #recover acf from squared freqs
    if ac[0] != 0:  # else: constant series (e.g. no fills)
        out[: max_lag + 1] = ac / ac[0] #normalise so ac[0] = 1
    return out


# P&L and inventory curves
def _plot_series(full, column, title, ylabel, resolutions, initial_cash,
                 zero_line=False):
    """
    Shared engine for the P&L and inventory plots.
    get unique strats -> grid multiple plots ->  thin+prep data -> add title, format
    """
    # get all unique strategy names from results
    strategies = sorted({s for s, _ in full}) 
    fig, axes = _grid(len(resolutions))
    for ax, res in zip(axes, resolutions): #zip each axes to a resolution into list of tuples, iterate.
        for s in strategies:
            if (s, res) not in full: # ensure the data is actually there
                continue
            df = _thin(_prep(full[(s, res)], initial_cash)) #add dt, mtm, then thin down df
            ax.plot(df["dt"], df[column], label=s, color=COLOURS.get(s), lw=1)
        if zero_line:
            ax.axhline(0, color="k", lw=0.5) #add a horizontal zero line for comparison
        ax.set_title(res)
        ax.set_ylabel(ylabel)
        ax.legend()
    fig.suptitle(title) # give collection of plots a supertitle
    fig.autofmt_xdate() # format dates properly
    fig.tight_layout() # adjusts spacing to fit labels
    return fig

#each plot here is against time so only need ylabel
def plot_pnl(full, resolutions=RESOLUTIONS, initial_cash=1_000_000.0):
    return _plot_series(full, "pnl", "Mark-to-market P&L", "P&L",
                        resolutions, initial_cash, zero_line=True)


def plot_inventory(full, resolutions=RESOLUTIONS, initial_cash=1_000_000.0):
    return _plot_series(full, "inventory", "Inventory", "units",
                        resolutions, initial_cash, zero_line=True)


# zoomed quote plot
def plot_quotes(full, resolution, start=None, n_steps=300):
    """
    for each strategy plot a graph with the bids, asks, and mid price. 
    filter resolutions -> create subplots for each strategy in a window.
    for each strategy -> plot midprice, bid/ask quotes (stepped) -> mark fills in window
    """
    # filter out other resolutions: we just want 1 resolution for each strategy
    strategies = sorted(s for s, r in full if r == resolution) 
    # formatting subplots. Share same x axis, return 2D numpy array.
    fig, axes = plt.subplots(len(strategies), 1,
                             figsize=(11, 3.6 * len(strategies)),
                             sharex=True, squeeze=False) 
    axes = axes.ravel() #restructure to 1D list of axes

    for ax, s in zip(axes, strategies):
        df = full[(s, resolution)]
        i0 = len(df) // 2 if start is None else start #start from middle if a start isn't given.
        w = df.iloc[i0: i0 + n_steps] #take a "w" window of the df from i0

        qt = pd.to_datetime(w["quote_time"], unit="ms")
        mt = pd.to_datetime(w["time"], unit="ms")
        ax.plot(mt, w["mid_price"], color="k", lw=1, label="mid")
        #step functions to accurately represent strategy quotes in the exchange.
        ax.step(qt, w["bid_quote"], where="post", color="tab:green", lw=1,
                label="bid quote")
        ax.step(qt, w["ask_quote"], where="post", color="tab:red", lw=1,
                label="ask quote")
        # from the window, keep only rows where quote is filled
        bf = w[w["bid_filled"].astype(bool)]
        af = w[w["ask_filled"].astype(bool)]
        #overlay bids as ^ arrow, asks as v arrow
        ax.scatter(pd.to_datetime(bf["bid_fill_time"], unit="ms"),
                   bf["bid_quote"], marker="^", color="tab:green", s=45,
                   zorder=3, label="bid fill")
        ax.scatter(pd.to_datetime(af["ask_fill_time"], unit="ms"),
                   af["ask_quote"], marker="v", color="tab:red", s=45,
                   zorder=3, label="ask fill")

        ax.set_title(f"{s} - {resolution}")
        ax.set_ylabel("price")
        ax.ticklabel_format(axis="y", useOffset=False)
        ax.legend(loc="upper left", ncol=5, fontsize=8)

    fig.autofmt_xdate()
    fig.tight_layout()
    return fig

# diagnostics
def plot_acf(full, resolution, nlags=200, initial_cash=1_000_000.0):
    """
    find strategy at right res -> cap max_lags -> plot acf for inventory, P&L increments
    """
    strategies = sorted(s for s, r in full if r == resolution)
    min_len = min(len(full[(s, resolution)]) for s in strategies)
    nlags = max(1, min(nlags, min_len - 1))  # don't ask for more lags than the data supports
    fig, (ax_inv, ax_pnl) = plt.subplots(1, 2, figsize=(12, 3.8))
    for s in strategies:
        df = _prep(full[(s, resolution)], initial_cash)
        lags = np.arange(nlags + 1)
        ax_inv.plot(lags, _acf(df["inventory"], nlags), label=s,
                    color=COLOURS.get(s))
        ax_pnl.plot(lags, _acf(np.diff(df["pnl"]), nlags), label=s,
                    color=COLOURS.get(s))
    ax_inv.set_title(f"Inventory ACF ({resolution})")
    ax_pnl.set_title(f"P&L increment ACF ({resolution})")
    for ax in (ax_inv, ax_pnl):
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel(f"lag ({resolution} steps)")
        ax.legend()
    fig.tight_layout()
    return fig

def plot_fill_rates(metrics, resolutions=RESOLUTIONS):
    """
    fill-rate data -> into table -> into barchart
    """
    # take strategy and turn into separate columns
    rates = metrics["fills_per_quote"].astype(float).unstack(0)
    rates = rates.reindex([r for r in resolutions if r in rates.index]) #reorder rows, and get rid of extra resolutions
    ax = rates.plot.bar(color=[COLOURS.get(c) for c in rates.columns],
                        figsize=(7, 4), rot=0)
    ax.set_ylabel("fills per quote")
    ax.set_title("Fill rate by resolution")
    fig = ax.get_figure()
    fig.tight_layout()
    return fig

def plot_data(full, metrics=None, resolutions=RESOLUTIONS,
              initial_cash=1_000_000.0, save_dir=None, show=True):
    """
    builds dict of figures, shows them and saves them if requested.
    loop through resolutions -> plot figures -> plot metrics ?-> save -> show
    """
    #if there are any dfs at the requested resolution "k[1]", loop thru them
    resolutions = [r for r in resolutions if any(k[1] == r for k in full)]
    figs = {
        "pnl": plot_pnl(full, resolutions, initial_cash),
        "inventory": plot_inventory(full, resolutions, initial_cash),
    }
    for r in resolutions:
        figs[f"quotes_{r}"] = plot_quotes(full, r)
        figs[f"acf_{r}"] = plot_acf(full, r, initial_cash=initial_cash)
    if metrics is not None:
        figs["fill_rates"] = plot_fill_rates(metrics, resolutions)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(save_dir / f"{name}.png", dpi=150)
    if show:
        plt.show()
    return figs

# pipeline
def run(hours, resolutions=RESOLUTIONS, strategies=None,
       initial_cash=1_000_000.0, output_dir=OUTPUT_DIR,
       save_dir=None, show=True):
    """
    Runs reconstruct -> analyse -> backtest -> calculate_metrics -> plot_data
    over each hour, carrying cash/inventory forward per (strategy, resolution).
    Factored out of main() for quick test-call with 1-2 hours.
    """
    strategies = STRATEGIES if strategies is None else strategies # do all of them if specific one isn't chosen

    # pair cash + inventory tuple, each strategy at each resolution
    state = {(s, r): {"cash": initial_cash, "inventory": 0.0} 
             for s in strategies for r in resolutions}
    # key-value pair between tuple and results list.
    results = {(s, r): [] for s in strategies for r in resolutions}
    # results look like { ("as", "1s"): [hour1, hour2, ...], ("as", "5s"): [...], ("naive", "1s"): [...], ("naive", "5s"): [...] }
    for hour in hours:
        print(f"processing {hour.name}") # crash feedback
        market = reconstruct(pd.read_parquet(hour))  # 
        for r in resolutions:
            lr = analyse(market, r)  # once per hour + resolution
            print(f"  {r}: {len(lr)} LR rows") # print how many rows in the lower resolution df
            for name, strat in strategies.items():
                # identify which strategy (name + "r" resolution, strat -> backtest -> results)
                st = state[(name, r)] 
                res = backtest(market, lr, strat,
                               initial_cash=st["cash"],
                               initial_inventory=st["inventory"]) #backtest over 1 hour
                st["cash"] = res["cash"].iloc[-1] #record last cash
                st["inventory"] = res["inventory"].iloc[-1]# record last inventory
                # hour.stem is added as a column to "res" results df, then appended to results (so I can do multiple hours)
                results[(name, r)].append(res.assign(hour=hour.stem)) 

    # for each (strategy,resolution) concatenate all hours into 1 fresh dataframe "full", new index
    full = {k: pd.concat(v, ignore_index=True) for k, v in results.items()} 
    #e.g. full = {("as", "1s"): df24_hrs, ("naive", "1s"): df24_hrs}
    # for each (strategy, resolution) assign metrics (sharpe, max drawdown etc.), transpose dataframe so each (s,r) is a row; more convenient when plotting.
    metrics = pd.DataFrame({k: calculate_metrics(v) for k, v in full.items()}).T


    output_dir = Path(output_dir) # assign a given path 
    output_dir.mkdir(parents=True, exist_ok=True)  # create a directory (exist_ok creates any parent dirs if needed)
    metrics.to_parquet(output_dir / "metrics.parquet") #save metrics to compressed parquet file for later study if necessary.

    figs = plot_data(full, metrics, resolutions=resolutions,
                     initial_cash=initial_cash, save_dir=save_dir, show=show)
    return full, metrics, figs # return full results, result_metrics, and graphs


def main():
    run(HOURS, RESOLUTIONS)


if __name__ == '__main__':
    main()