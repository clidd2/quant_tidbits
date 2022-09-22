import pandas_datareader.data as web
import pandas as pd
import numpy as np
import os


def list_comp_handler(i):
    try:
        return i

    except Exception as err:
        print(err)
        return np.inf


def get_data(ticker: str):
    try:
        dat = web.YahooOptions(ticker)
        dat.headers = {'User-Agent': 'Chrome'}
        return dat

    except Exception as err:
        #TODO: test to see what errors might be caused here
        return -1


def get_expiries(dat):
    return [exp for exp in dat.expiry_dates]

def get_all_data(dat):
    return dat.get_all_data().reset_index()


def main():
    ticker = 'TSLA'
    dat = get_data(ticker)
    expiries = get_expiries(dat)
    print(expiries)
    print(get_all_data(dat))


if __name__ == '__main__':
    main()
