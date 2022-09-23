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
    d1 = (np.log(price/strike) + (interest + 0.5*volatility**2)*time) / \
    (volatility*np.sqrt(time))

    return d1


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

    d_two = d_one(price, strike, time, interest, volatility) - volatility * \
    np.sqrt(time)
    return d_two


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
                            max_iter = 10000, verbose=False):
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

    if verbose:
        print('INITIAL PARAMETERS')
        print('==================')
        print(f'Option Price: {option_price}')
        print(f'Price: {price}')
        print(f'Strike: {strike}')
        print(f'Time: {time}')
        print(f'Interest Rate: {interest}')
        print(f'Initial Volatility: {volatility}')


    for i in range(0, max_iter):
        try:
            err = call_option(price, strike, time,
                             interest, volatility) - option_price

            if (abs(err) < threshold):
                return volatility

            v = vega(price, strike, time, interest, volatility)
            volatility = volatility - err / v



        except Exception as err:
            print(err)
            volatility = -1
            continue

    return volatility


def main():

    start_date = dt.date.today()
    end_date = dt.date(year=2022, month=10, day=21)
    time = (end_date - start_date).days
    option_price = 1.7
    price = 276.45
    strike = 348.33
    interest = 0.025
    threshold = 0.00001
    annual_vol = implied_call_volatility(option_price, price, strike, time, \
                                        interest)
    daily_vol = annual_vol / np.sqrt(252)
    print(f'Annual Volatility: {annual_vol}')
    print(f'Daily Volatility: {daily_vol}')


if __name__ == '__main__':
    main()
