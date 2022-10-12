import pandas as pd
import requests
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
import pandas_datareader as web
import os
import time
import datetime as dt
import re

def get_tickers() -> list:
    '''
    gets all S&P 500 tickers from the url listed below.

    params
    ======
    None

    returns
    =======
    list object containing all current S&P 500 tickers
    '''

    tickers = list()

    #need to think of a way to make sure this is always kosher
    url = r'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    resp = requests.get(url) #most of these self-explanatory... get the url

    soup = BeautifulSoup(resp.text, features='lxml') #convert to soup obj

    table = soup.find('table') #find the index composition table

    rows = [row for row in table.find_all('tr')] #nice lil list comp of rows

    for row in rows:
        #in each table row search for an identifier
        found = re.search(r'CIK=([A-Z]*)&amp', str(row))

        #unfortunately need the try except-loop here for weird ticker issues
        try:
            #damn you BRK-A and BRK-B
            #ticker found, add to list
            tickers.append(found.group(1))

        except Exception as err:
            #TODO: could clean this up a bit and be more explicit catching errs
            continue

    return tickers


def concurrent_sp_data(tickers : list, start : dt.date, end=dt.date.today(),
                    increments=10) -> dict:

    '''
    concurrent calls to pandas_datareader to generate a list of csv dataframes.
    Should think about ways this could go wrong and catch them preemptively

    params
    ======
    tickers (list): list of ticker symbols to request data for
    start (datetime.date): requested start date of pricing data
    end (datetime.date): requested end date of pricing data
    increments (int): max batch size for concurrent calls

    returns
    =======
    dictionary of form {ticker : pd.DataFrame} for each successfully received
    piece of data
    '''
    res = dict()

    #timing for the sake of own edification. Can uncomment print function to see
    #full exectution time
    t0 = time.time()

    #context manage the executor
    with ThreadPoolExecutor(max_workers=increments) as executor:
        for ticker in tickers:
            try:
                res[ticker] = executor.submit(get_data, ticker, start, end)

            except Exception as err:
                print(err)

    t1 = time.time()
    completion_time = t1 - t0
    #print(f'Concurrent data collection took {completion_time} seconds to run')

    #big fan of list/dict comprehension in these cases. Cleaner code
    to_return = {ticker : future.result() for ticker, future in res.items()}
    return to_return



def merge_frame(data_dict : dict) -> pd.DataFrame:
    '''
    merge the dictionary of dataframes into a composite frame with all closing
    prices

    params
    ======
    data_dict (dict): dictionary of form {ticker : pd.DataFrame} with pricing
                      data

    returns
    =======
    composite pandas dataframe containing all requested pricing data
    '''

    series = list()
    keys = data_dict.keys()

    #unfortunately need to explicitly iterate for try-except block
    #util function to error catch?
    #https://stackoverflow.com/questions/1528237/how-to-handle-exceptions-in-a-list-comprehensions
    for df in data_dict.values():
        try:
            #date index to properly concat all data later. could do earlier?
            df.set_index('Date', inplace=True)
            #add date-index pricing data series
            series.append(df['price'])

        except Exception as err:
            #TODO: figure out a better exception catching framework here
            print(err)
            continue

    return pd.concat(series, axis=1, keys=keys)


def get_data(ticker : str, start_date : dt.date, end_date=dt.datetime.today(),
        colname='Adj Close') -> pd.DataFrame:

    '''
    fetch a single pricing data set over requested period for given ticker

    params
    ======
    ticker (str): ticker of company/asset of which data is being requested
    start_date (datetime.date): requested pricing data start date
    end_date (datetime.date): requested pricing data end date
    colname (str): column name for series representing adj. close. This will
                    vary with data provider, so adding for sake of flexibility

    returns
    =======
    pandas dataframe object with OHLP data for requested ticker over end - start
    trading days (weekends excluded)
    '''

    #attempt to make request for data over time period
    try:
        df = web.DataReader(name=ticker,data_source='yahoo',start=start_date,
                            end=end_date)
        df.index = pd.to_datetime(df.index)
        #rename column within the return statement for more lean code
        return df.rename(columns={colname:'price'}).reset_index()

    except Exception as err:
        print(err)
        #my de facto error code is same as cpp... -1 signifies something has
        #gone horribly wrong. see this a lot when over api call limit.
        return -1


def generate_fname() -> str:
    '''
    generic filename export function. nothing too crazy here, just make it
    easier to track DoD, WoW, YoY, etc. data changes

    params
    ======
    None

    returns
    =======
    string denoting filename in csv format
    '''
    today = dt.date.today()
    year = today.year
    year_str = str(year % 2000)
    month = today.month

    #always want a consistent MM/DD/YY format. Pad 0 if month less than 10
    month_str = str(month) if month > 9 else f'0{month}'
    day = today.day
    #always want a consistent MM/DD/YY format. Pad 0 if day less than 10
    day_str = str(day) if day > 9 else f'0{day}'

    #TODO: might want to add optional ftype parameter... not significant now
    return f'{month_str}{day_str}{year_str}_data.csv'


def export_price_data(dfs: dict, opath: str) -> None:
    '''
    exports a series of pandas dataframes to stated outpath location

    params
    ======
    dfs (dict): dictionary of form {ticker : pd.DataFrame} with pricing data
    opath (str): outpath destination for export of data

    returns
    =======
    No return here, this function exports data to file location and nothing else
    '''

    for ticker, df in dfs.items():
        try:
            #outfile name
            outfile = f'{ticker}.csv'

            #like keeping this in here to track progress
            print(f'Saving {ticker} dataframe to csv file...')
            df.to_csv(os.path.join(opath, outfile))

        except AttributeError as AErr:
            #generally getting error when ticker did not receive data (-1 code)
            print(f'Error when attempting to export data for {ticker}: {AErr}')
            continue


def main():
    tickers = get_tickers()
    end = dt.date.today()
    start = end - dt.timedelta(365*5)
    col = 'Adj Close'
    dfs = concurrent_sp_data(tickers, start, end, 25)
    opath = r'/Users/christianliddiard/Programming/Python/data'
    export_price_data(dfs, opath)



if __name__ == '__main__':
    main()
