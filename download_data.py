import cryptohftdata as chd
from secrets import API_KEY

chd.configure_client(api_key=API_KEY)

print("Downloading orderbook data...")

orderbook = chd.get_orderbook(
    symbol="BTCUSDT",
    exchange=chd.exchanges.BINANCE_FUTURES,
    start_date="2026-08-28",
    end_date="2026-08-28",
)

print(f"Downloaded {len(orderbook):,} rows")

# save locally 
filename = "data/BTCUSDT_orderbook_2026-08-28.parquet"

orderbook.to_parquet(filename, index=False)

print(f"Saved data to: {filename}")
print(orderbook.head())