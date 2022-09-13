import numpy as np
from scipy.stats import norm
import datetime as dt
import matplotlib.pyplot as plt






def d_one(price, strike, time, interest, volatility):
    return (np.log(price / strike) + (interest  + 0.5 * np.power(volatility,2)) * time) / (volatility * np.sqrt(time))


def d_two(price, strike, time, interest, volatility):
    return d_one(price, strike, time, interest, volatility) - (volatility * np.sqrt(time))


def call_option(price, strike, time, interest, volatility) -> float:
    '''
    model a call option based on generalized BSM model
    input price: current stock price of underlying
    input strike: strike price of option contract
    input time: time to expiry of contract, expressed in days
    input interest: interest rate over period
    input volatility: vol over time period
    return: model-implied price of option
    '''

    d1 = (np.log(price / strike) + (interest + 0.5 * volatility ** 2) * time) / (volatility * np.sqrt(time))
    d2 = (np.log(price / strike) + (interest - 0.5 * volatility ** 2) * time) / (volatility * np.sqrt(time))

    call = (price * norm.cdf(d1, 0.0, 1.0) - strike * np.exp(-interest * time) * norm.cdf(d2, 0.0, 1.0))
    return call



def implied_call_volatility(price, strike, time, interest, target, threshold= 0.0001):
    high = 5
    low = 0
    while high-low  > threshold:
        theoretical = call_option(price, strike, time, interest, (high+low)/2)
        if theoretical > target:
            high = (high+low) / 2

        else:
            low = (high+low) / 2

    return (high + low) / 2




def main():

    start_date = dt.date.today()
    end_date = dt.date(2022, 10, 7)
    time = (end_date - start_date).days / 365

    price = 270.21
    strike = 275
    interest = 0.0024
    target = 3.9
    threshold = 0.00001
    annual_vol = implied_call_volatility(price, strike, time, interest, target, threshold)
    daily_vol = annual_vol / np.sqrt(252)
    print(f'Annual Volatility: {annual_vol}')
    print(f'Daily Volatility: {daily_vol}')


if __name__ == '__main__':
    main()
