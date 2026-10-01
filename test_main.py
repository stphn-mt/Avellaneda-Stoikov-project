import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: never pops a GUI window, never blocks

import main  # noqa: E402  (import after backend is set)
"""
Smoke test for main.py: runs the REAL reconstruct/analyse/backtest pipeline
end-to-end on just 1-2 hours of real data, so you catch integration bugs
(shape mismatches, empty LR frames at coarse resolutions, plotting errors)
before committing to a 24-hour run.

Run from your project root (same place you'd run `python main.py`):
    python test_main.py
    python test_main.py 1          # only the first 1 hour
    python test_main.py 3          # first 3 hours

Doesn't touch data/timeseries/ - writes to test_output/ instead.
"""

TEST_OUTPUT_DIR = Path("test_output")


def check(full, metrics):
    """Basic sanity checks that would otherwise only surface as confusing
    downstream errors or silently-wrong plots."""
    problems = []

    for k, df in full.items():
        if df.empty:
            problems.append(f"{k}: EMPTY results (0 rows)")
            continue
        n_nan = df.isna().sum().sum()
        if n_nan:
            problems.append(f"{k}: {n_nan} NaNs across {df.shape}")
        n_fills = int(df["bid_filled"].sum() + df["ask_filled"].sum())
        if n_fills == 0:
            problems.append(f"{k}: zero fills in {len(df)} quotes - "
                            f"check quote sizing / fill logic")

    if metrics.isna().any().any():
        bad_cols = metrics.columns[metrics.isna().any()].tolist()
        problems.append(f"metrics has NaNs in: {bad_cols}")

    return problems


def main_test():
    n_hours = int(sys.argv[1]) if len(sys.argv) > 1 else 2

    if not main.HOURS:
        print(f"No files matched {main.RAW_DIR}/BTCUSDT_orderbook_20260918_*.parquet "
             f"- check RAW_DIR and the glob pattern in main.py.")
        sys.exit(1)

    test_hours = main.HOURS[:n_hours]
    print(f"Testing with {len(test_hours)} hour(s): "
         f"{[h.name for h in test_hours]}\n")

    try:
        full, metrics, figs = main.run(
            test_hours,
            resolutions=main.RESOLUTIONS,
            output_dir=TEST_OUTPUT_DIR,
            save_dir=TEST_OUTPUT_DIR / "figures",
            show=False,
        )
    except Exception:
        print("\n--- Pipeline raised an exception ---")
        traceback.print_exc()
        print("\nThe progress lines above ('processing ...', '  <res>: N LR "
             "rows') show which hour/resolution it reached before failing.")
        sys.exit(1)

    print("\n--- result shapes ---")
    for k, df in sorted(full.items()):
        print(f"  {k}: {df.shape}")

    print("\n--- metrics ---")
    print(metrics)

    problems = check(full, metrics)
    print("\n--- checks ---")
    if problems:
        for p in problems:
            print(f"  ! {p}")
        print(f"\n{len(problems)} issue(s) found - see above.")
    else:
        print("  all checks passed.")

    print(f"\nFigures saved to {TEST_OUTPUT_DIR / 'figures'}/")
    print(f"Metrics saved to {TEST_OUTPUT_DIR / 'metrics.parquet'}")


if __name__ == "__main__":
    main_test()