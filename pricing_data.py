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

def get_tickers():

    tickers = list()
    url = r'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, features='lxml')
    table = soup.find('table')
    rows = [row for row in table.find_all('tr')]
    for row in rows:
        found = re.search(r'CIK=([A-Z]*)&amp', str(row))
        try:
            tickers.append(found.group(1))

        except Exception as err:
            continue

    return tickers




def collect_sp_tickers():
    tickers = list()
    url = r'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    table = soup.find('table')
    rows = table.find_all('tr')
    for row in rows:
        found = re.search(r'CIK=([A-Z]*)&amp', str(row))
        try:
            tickers.append(found.group(1))

        except Exception as err:
            print(err)
            continue

    return tickers

def concurrent_sp_data(tickers : list, start : dt.date, end=dt.date.today(),
                    increments=10) -> dict:
    res = dict()
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=increments) as executor:
        for ticker in tickers:
            res[ticker] = executor.submit(get_data, ticker, start, end)

    t1 = time.time()
    completion_time = t1 - t0

    to_return = {ticker : future.result() for ticker, future in res.items()}
    return to_return



def merge_frame(data_dict : dict) -> pd.DataFrame:
    series = list()
    keys = data_dict.keys()
    for df in data_dict.values():
        try:
            df.set_index('Date', inplace=True)
            series.append(df['price'])

        except Exception as err:
            print(err)
            continue

    return pd.concat(series, axis=1, keys=keys)


def get_data(ticker : str, start_date : dt.date, end_date=dt.datetime.today(),
        colname='Adj Close') -> pd.DataFrame:
    try:
        df = web.DataReader(name=ticker,data_source='yahoo',start=start_date,
                            end=end_date)
        df.index = pd.to_datetime(df.index)
        return df.rename(columns={colname:'price'}).reset_index()

    except Exception as err:
        print(err)
        return -1


def generate_fname() -> str:
    today = dt.date.today()
    year = today.year
    year_str = str(year % 2000)
    month = today.month
    month_str = str(month) if month > 9 else f'0{month}'
    day = today.day
    day_str = str(day) if day > 9 else f'0{day}'
    return f'{month_str}{day_str}{year_str}_data.csv'


def export_price_data(dfs, opath):
    for ticker, df in dfs.items():
        try:
            outfile = f'{ticker}.csv'
            print(f'Saving {ticker} dataframe to csv file...')
            df.to_csv(os.path.join(opath, outfile))

        except AttributeError as AErr:
            print(f'Error when attempting to export data for {ticker}: {AErr}')
            continue


def main():
    tickers = get_tickers()
    end = dt.date.today()
    start = end - dt.timedelta(365*5)
    col = 'Adj Close'
    dfs = concurrent_sp_data(tickers, start, end, 25)
    opath = os.getcwd()
    export_price_data(dfs, opath)



if __name__ == '__main__':
    main()
