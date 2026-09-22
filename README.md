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
Getting the order book data ended up being more of a pain than I originally thought it would be.
I had to already make some design choices: do I want L1 or L2 data? In other words, how detailed do i want my data to be? If I want more detail, though, that will cost the length of time I can simulate. 
I decided that L2 data was more aligned with the spirit of this project: I want to test the AS model in a more realistic environment, so I should be open to the idea of simulating quote fills accounting for volume. 
There are more caveats though; some data sources provide high quality L2 data, but I couldn't import more than 1 hour at a time. 
Other sources include using a Binance API, and downloading orderbook data myself. 
For the first version of my project, I wanted to get straight to implementing the strategy, so I went with downloading the pre-recorded data on cryptohftdata.com. 



## Strategy
what will I do with it, why, and how?
### Avellaneda-Stoikov model
- Reservation price
- Optimal spread
- Inventory adjustment

### Implementation
- Quote frequency
- Order size
- Fill model
- Inventory/cash accounting
- Execution logic

## Methodology & Assumptions
what assumptions am I making to limit the scope of this project, and why?
## Results
what did I find - did I achieve my original goal? To what extent (measurements)?