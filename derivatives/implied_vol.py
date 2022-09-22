import numpy as np
from scipy.stats import norm
import datetime as dt
import matplotlib.pyplot as plt




def d_one(price, strike, time, interest, volatility):
    '''
    conditional probability function looking at moneyness of option

    params
    ======
    price (float): current stock price of underlying
    strike (float): strike price of option contract
    time (datetime): time to expiry of contract, expressed in days
    interest (float): implied interest rate over contract period
    volatility (float): historical vol over contract period

    returns
    =======
    model-implied conditional probability of option moneyness

    '''
    print(price)
    print(strike)
    print(interest)
    print(volatility)
    print(time)
    return (np.log(price / strike) + (interest  + 0.5 * np.power(volatility,2))\
            * time) / (volatility * np.sqrt(time))


def d_two(price, strike, time, interest, volatility):
    '''
    probability of option expiring ITM

    params
    ======
    price (float): current stock price of underlying
    strike (float): strike price of option contract
    time (datetime): time to expiry of contract, expressed in days
    interest (float): implied interest rate over contract period
    volatility (float): historical vol over contract period

    returns
    =======
    model-implied probability of option being ITM

    '''

    return d_one(price, strike, time, interest, volatility) - (volatility * \
            np.sqrt(time))


def call_option(price, strike, time, interest, volatility) -> float:
    '''
    model a call option based on generalized BSM model

    params
    ======
    price (float): current stock price of underlying
    strike (float): strike price of option contract
    time (datetime): time to expiry of contract, expressed in days
    interest (float): implied interest rate over contract period
    volatility (float): historical vol over contract period

    returns
    =======
    model-implied price of option
    '''

    d1 = d_one(price, strike, time, interest, volatility)
    d2 = d_two(price, strike, time, interest, volatility)

    call = (price * norm.cdf(d1, 0.0, 1.0) - strike * np.exp(-interest * time) \
            * norm.cdf(d2, 0.0, 1.0))

    return call


def vega(price, strike, time, interest, volatility):
    '''
    vega estimation function

    params
    ======
    price (float): current stock price of underlying
    strike (float): strike price of option contract
    time (datetime): time to expiry of contract, expressed in days
    interest (float): implied interest rate over contract period
    volatility (float): historical vol over contract period

    returns
    =======
    estimated vega of option
    '''
    d1 = d_one(price, strike, time, interest, volatility)
    return price * np.sqrt(time) * norm.pdf(d1)

def implied_call_volatility(option_price, price, strike, time, interest,
                            volatility = 0.30, threshold = 0.0001,
                            max_iter = 10000):
    '''
    estimation of implied volatility from observable market data

    params
    ======
    option_price (float): current observable option price
    price (float): current stock price of underlying
    strike (float): strike price of option contract
    time (datetime): time to expiry of contract, expressed in days
    interest (float): implied interest rate over contract period
    volatility (float): initial volatility estimate to work from
    threshold (float): error threshold to break iteration on
    max_iter (float): max number of iterations to allow before breaking

    returns
    =======
    implied volatility of option given market data
    '''

    print(option_price)
    print(price)
    print(strike)
    print(time)
    print(interest)


    for i in range(0, max_iter):
        print(f'Init Vol: {volatility}')
        err = call_option(price, strike, time,
                         interest, volatility) - option_price
        print(err)

        if (abs(err) < threshold):
            return volatility
        volatility = volatility + err / vega(price, strike, time, interest,
                                             volatility)

    return volatility


def main():

    start_date = dt.date.today()
    end_date = dt.date(2023, 1, 20)
    time = (end_date - start_date).days / 365
    option_price = 276.87
    price = 300.2996
    strike = 3.33
    interest = 0.0350
    threshold = 0.00001
    annual_vol = implied_call_volatility(option_price, price, strike, time, interest)
    daily_vol = annual_vol / np.sqrt(252)
    print(f'Annual Volatility: {annual_vol}')
    print(f'Daily Volatility: {daily_vol}')


if __name__ == '__main__':
    main()
