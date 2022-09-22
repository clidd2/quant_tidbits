import pandas_datareader.data as web
import pandas as pd
import numpy as np
import os
from implied_vol import implied_call_volatility

def pandas_options():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

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
    return [list_comp_handler(exp) for exp in dat.expiry_dates]

def get_all_data(dat):
    df = dat.get_all_data().reset_index()
    df['mid_price'] = (df['Bid'] + df['Ask']) / 2
    df['time_to_exp'] = (df['Expiry'] - df['Quote_Time']).dt.days / 365
    df = df[df['mid_price'].isna()  == False]
    df = df[df['mid_price'] > 0]
    return df


def sanitize_input(df, cols):
    df.dropna(inplace=True)
    for col in cols:
        df = df[df[col] != 0]
    return df

def call_iv(df, interest_rate, init_vol):
    input_cols = ['mid_price', 'Underlying_Price', 'Strike', 'time_to_exp']



    df = df[df['Type'] == 'call']

    df = sanitize_input(df, input_cols)
    df['interest_rate'] = interest_rate
    df['init_vol_est'] = init_vol

    df['iv_calc'] = df.apply(lambda x: implied_call_volatility(x['mid_price'],
     x['Underlying_Price'], x['Strike'], x['time_to_exp'], x['interest_rate'],
     x['init_vol_est']), axis=1)

    return df


def main():
    interest_rate = 0.03
    init_vol = 0.3
    #pandas_options()
    ticker = 'TSLA'
    dat = get_data(ticker)
    expiries = get_expiries(dat)

    df = get_all_data(dat)
    print(df)
    #call_df = call_iv(df, interest_rate, init_vol)
    #print(call_df)


if __name__ == '__main__':
    main()
