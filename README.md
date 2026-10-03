# Avellaneda-Stoikov-project
## Objective
What am I investigating?
I am investigating trading models. 
This project will be quite limited scope, but I think that makes sense as a first project.
I will be comparing a naive model with the Avellaneda Stoikov model, and assessing their performance on real market data.
I will then modify the Avellaneda Stoikov model by taking into account order flow imbalance, and seeing how that affects performance.

## Introduction 
what is AS, what is the context it is used in, why did it come about, what is it solving?

The Avellaneda Stoikov (which I will refer to as AS) model came about as a way of accounting for inventory exposure when trading. Ideally when trading for a certain time period, by the end of that time period, the trader will make a profit, and will not have taken on much risk to do so. In practice, that means when trading, one's inventory is taken into account; if they have taken on too much risk, they will try to trade back out of that risk first, so that money is purely made on making a market, rather than hedging on a stock.

The original paper thus compares a naive model, which trades purely based on the mid-price, with a model that does take into account the inventory held at a particular moment. The paper manages to quantify this as a "reservation price", which the bid and ask price are centered on; the reservation price will have its distance (from the mid price) dependent on inventory exposure. Moreover, the model quantifies an optimal spread around the mid-price using a parameter for risk aversion with a rolling volatility.

The result of the paper was that accounting for inventory substantially reduced variance in P&L, but also generated "more modest profits".
What this project will aim to do, then, is to firstly investigate whether the results of the paper transfer over into results on real market LOB data; the original paper models the market as a random walk.


## Data
what data am I using, what units.




## Strategy
what will I do with it, why, and how?

- HR/LR data
Because I was using data that was not my own, I had to filter out a lower-resolution time series of the data; this gave me the idea of trying different time resolutions with my trading strategies. 
I would import the high-resolution data with millisecond precision (most of the data only needed ~100ms precision anyway), and then filter out lower-resolution dataframes for: 100ms, 1s, 5s, 1min. I could then compare how the trading strategies did on several time scales. It would also serve the purpose of protecting my results from being completely invalid: market impact was certainly a worry for me, so by keeping the order size that I quoted small, and having my strategy trade at a range of time frequencies, my market impact should be low.

There was also the problem of RAM: for a more accurate trading strategy, I would need lots of high-quality data. However, my computer couldn't even import a day's worth of BTCUSDT order book L2 data at once in a parquet file. 
My work-around for this was to import each hour of the day seperately, and then have my main.py file handle 1 hour at a time, and concatenate the data fo the results. By that point the data would be filtered by my lower-resolution dataframes. 

### Avellaneda-Stoikov model
- Reservation price
calculated by: mid_price - inventory*gamma*(volatility**2)*time_horizon
I used fixed gamma = 0.1
- Optimal spread
the optimal spread around the reservation price at time t was then equal to delta_t:
delta_t = (1/2)*(gamma*(volatility**2)*time_horizon+(2/gamma)*np.log(1+gamma/k))

### Implementation
- Quote frequency
- Order size
- Fill model
- Inventory/cash accounting
- Execution logic

## Methodology & Assumptions
what assumptions am I making to limit the scope of this project, and why?
- L2 vs L1 data
- 1 symbol for 1 day
- Fill assumption
- Volatility estimation window
- Naive benchmark uses volatility currently
- AS parameters fixed, no calibrated
- Fixed order size
- No fees, no latency, no partial fills
- Time horizon resets every hour


## Results
what did I find - did I achieve my original goal? To what extent (measurements)?