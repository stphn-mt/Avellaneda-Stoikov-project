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

# FIX: this was referenced but never defined. If naive_strategy / as_strategy
# need fixed parameters (gamma, k, half_spread, ...), wrap them with
# functools.partial here, e.g.:
#   "as": partial(as_strategy, gamma=0.1, k=1.5)
# Using them unwrapped for now — check this matches your models.py signatures.
STRATEGIES = {
    "naive": naive_strategy,
    "as": as_strategy,
}


# ---------------------------------------------------------------- helpers
def _prep(df, initial_cash):
    """Add a datetime column and mark-to-market P&L."""
    df = df.copy()
    df["dt"] = pd.to_datetime(df["time"], unit="ms")
    df["pnl"] = df["cash"] + df["inventory"] * df["mid_price"] - initial_cash
    return df


def _thin(df, max_points=5000):
    """Downsample for full-day plots (100ms data is ~864k rows)."""
    step = max(1, len(df) // max_points)
    return df.iloc[::step]


def _grid(n):
    ncols = 2 if n > 1 else 1
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 3.5 * nrows),
                             squeeze=False)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes


def _acf(x, nlags):
    """
    Autocorrelation via FFT (fast enough for 100ms data). Always returns
    exactly nlags+1 values, so the caller's x-axis (np.arange(nlags+1))
    never mismatches: lags beyond what the series supports come back NaN
    (matplotlib skips them) instead of silently shortening the array.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.full(nlags + 1, np.nan)
    if n < 2:
        return out
    x = x - x.mean()
    max_lag = min(nlags, n - 1)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[: max_lag + 1]
    if ac[0] != 0:  # else: constant series (e.g. no fills)
        out[: max_lag + 1] = ac / ac[0]
    return out


# ------------------------------------------------- P&L and inventory curves
def _plot_series(full, column, title, ylabel, resolutions, initial_cash,
                 zero_line=False):
    strategies = sorted({s for s, _ in full})
    fig, axes = _grid(len(resolutions))
    for ax, res in zip(axes, resolutions):
        for s in strategies:
            if (s, res) not in full:
                continue
            df = _thin(_prep(full[(s, res)], initial_cash))
            ax.plot(df["dt"], df[column], label=s, color=COLOURS.get(s), lw=1)
        if zero_line:
            ax.axhline(0, color="k", lw=0.5)
        ax.set_title(res)
        ax.set_ylabel(ylabel)
        ax.legend()
    fig.suptitle(title)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_pnl(full, resolutions=RESOLUTIONS, initial_cash=1_000_000.0):
    return _plot_series(full, "pnl", "Mark-to-market P&L", "P&L",
                        resolutions, initial_cash, zero_line=True)


def plot_inventory(full, resolutions=RESOLUTIONS, initial_cash=1_000_000.0):
    return _plot_series(full, "inventory", "Inventory", "units",
                        resolutions, initial_cash, zero_line=True)


# ------------------------------------------------------ zoomed quote plot
def plot_quotes(full, resolution, start=None, n_steps=300):
    strategies = sorted(s for s, r in full if r == resolution)
    fig, axes = plt.subplots(len(strategies), 1,
                             figsize=(11, 3.6 * len(strategies)),
                             sharex=True, squeeze=False)
    axes = axes.ravel()

    for ax, s in zip(axes, strategies):
        df = full[(s, resolution)]
        i0 = len(df) // 2 if start is None else start
        w = df.iloc[i0: i0 + n_steps]

        qt = pd.to_datetime(w["quote_time"], unit="ms")
        mt = pd.to_datetime(w["time"], unit="ms")
        ax.plot(mt, w["mid_price"], color="k", lw=1, label="mid")
        ax.step(qt, w["bid_quote"], where="post", color="tab:green", lw=1,
                label="bid quote")
        ax.step(qt, w["ask_quote"], where="post", color="tab:red", lw=1,
                label="ask quote")

        bf = w[w["bid_filled"].astype(bool)]
        af = w[w["ask_filled"].astype(bool)]
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


# ------------------------------------------------------------ diagnostics
def plot_acf(full, resolution, nlags=200, initial_cash=1_000_000.0):
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
    rates = metrics["fills_per_quote"].astype(float).unstack(0)
    rates = rates.reindex([r for r in resolutions if r in rates.index])
    ax = rates.plot.bar(color=[COLOURS.get(c) for c in rates.columns],
                        figsize=(7, 4), rot=0)
    ax.set_ylabel("fills per quote")
    ax.set_title("Fill rate by resolution")
    fig = ax.get_figure()
    fig.tight_layout()
    return fig


# ------------------------------------------------------------- entry point
def plot_data(full, metrics=None, resolutions=RESOLUTIONS,
              initial_cash=1_000_000.0, save_dir=None, show=True):
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


# --------------------------------------------------------------- pipeline
def run(hours, resolutions=RESOLUTIONS, strategies=None,
       initial_cash=1_000_000.0, output_dir=OUTPUT_DIR,
       save_dir=None, show=True):
    """
    Runs reconstruct -> analyse -> backtest -> calculate_metrics -> plot_data
    over `hours`, carrying cash/inventory forward per (strategy, resolution).
    Factored out of main() so a quick test can call it on 1-2 hours.
    """
    strategies = STRATEGIES if strategies is None else strategies

    state = {(s, r): {"cash": initial_cash, "inventory": 0.0}
             for s in strategies for r in resolutions}
    results = {(s, r): [] for s in strategies for r in resolutions}

    for hour in hours:
        print(f"processing {hour.name}")
        market = reconstruct(pd.read_parquet(hour))  # once per hour
        for r in resolutions:
            lr = analyse(market, r)  # once per hour + resolution
            print(f"  {r}: {len(lr)} LR rows")
            for name, strat in strategies.items():
                st = state[(name, r)]
                res = backtest(market, lr, strat,
                               initial_cash=st["cash"],
                               initial_inventory=st["inventory"])
                st["cash"] = res["cash"].iloc[-1]
                st["inventory"] = res["inventory"].iloc[-1]
                results[(name, r)].append(res.assign(hour=hour.stem))

    full = {k: pd.concat(v, ignore_index=True) for k, v in results.items()}
    metrics = pd.DataFrame({k: calculate_metrics(v) for k, v in full.items()}).T

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # FIX: was missing
    metrics.to_parquet(output_dir / "metrics.parquet")

    figs = plot_data(full, metrics, resolutions=resolutions,
                     initial_cash=initial_cash, save_dir=save_dir, show=show)
    return full, metrics, figs


def main():
    run(HOURS, RESOLUTIONS)


if __name__ == '__main__':
    main()