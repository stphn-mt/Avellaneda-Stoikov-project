import numpy as np


def naive_strategy(mid_price, volatility, spread, inventory, time_horizon):
    alpha=1.0
    # Convert return volatility into approximate price volatility
    price_volatility = mid_price * volatility
    epsilon = max(alpha * price_volatility, spread/2) #we have the max spread because sometimes volatility=0
    # Quote symmetrically around the mid-price
    bid_quote = mid_price - epsilon
    ask_quote = mid_price + epsilon

    return bid_quote, ask_quote

def as_strategy(mid_price, volatility, spread, inventory, time_horizon):
    gamma=0.1 
    k=1.5
    reservation_price = mid_price - inventory*gamma*(volatility**2)*time_horizon

    delta_t = (1/2)*(gamma*(volatility**2)*time_horizon+(2/gamma)*np.log(1+gamma/k))

    bid_quote = reservation_price - delta_t
    ask_quote = reservation_price + delta_t
    return bid_quote, ask_quote
