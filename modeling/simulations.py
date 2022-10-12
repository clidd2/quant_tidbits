import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import os


def read_csv(fpath, fname):
    full_path = os.path.join(fpath, fname)
    return pd.read_csv(full_path, index_col=0, header=0)


class Brownian():

    def __init__(self, x0=0, df=None):
         self._x0 = x0
         self._df = df

    def get_x0(self):
        return self._x0

    def get_df(self):
        return self._df

    def set_x0(self, x0):
        self._x0 = x0


    def set_vol(self, vol):
        self._vol = vol


    def calc_vol(self):
        '''
        generates daily vol from dataframe assuming Yahoo Finance naming scheme
        Params
        ======
        No inputs, pulls from within object

        Return
        ======
        float value representing standard deviation at point in time
        '''

        return np.array(self.get_df()['Adj Close']).std()


    def random_distrib_walk(self, n_steps=100):
        '''
        Generate series of random scalars from 1 to -1 based on weiner process

        params
        ======
        n_steps (int) : default number of steps for random walk

        return
        ======
        numpy array containing scalar values from which to generate random walk
        path
        '''

        if n_steps < 30:
            #cannot reliably generate walk with less than 30 steps
            raise ValueError('Must have more than 30 steps.')

        #array created and populated with initial value
        w = np.ones(n_steps) * self.get_x0()

        for i in range(1, n_steps):
            yi = np.random.choice([1, -1])
            w[i] = w[i-1] + (yi / np.sqrt(n_steps))

        return w


    def normal_distrib_walk(self, n_steps=100):
        '''
        Generate series of normally distributed scalars based on weiner process

        params
        ======
        n_steps (int) : default number of steps for random walk

        return
        ======
        numpy array containing scalar values from which to generate random walk
        path
        '''

        if n_steps < 30:
            #cannot reliably generate walk with less than 30 steps
            raise ValueError('Must have more than 30 steps.')

        #array created and populated with initial value
        w = np.ones(n_steps) * self.get_x0()

        for i in range(1, n_steps):
            yi = np.random.normal()
            w[i] = w[i-1] + (yi / np.sqrt(n_steps))

        return w


    def asset_performance(self, pricing_func=None, s0=None, vol=None, drift=1,
                        time_period=252, time_slice=1, type='normal'):

        '''
        Run stochastic browninan motion simulation on asset and return result

        Inputs
        ======
        pricing_func (func): function to return drift through
                            (ie: bond pricing function)
        drift (float): drift term for weiner process
        time (int): time period over which simulation takes place
        time_slice (float): number of slices per 1 time unit
        type (str): choose between simulation types; default normal walk

        Return
        ======
        generate price series of asset performance under simulation
        '''

        if vol == None:
            vol = self.calc_vol()

        n_steps = int(time_period / time_slice)
        time_arr = np.linspace(0, time_period, num=n_steps)
        variance = (drift - (vol ** 2 / 2)) * time_arr

        if s0 == None:
            raise ValueError('Must provide an initial asset price.')


        self.set_x0(0)

        #select simulation type
        if type == 'normal':
            weiner = drift * self.normal_distrib_walk(n_steps=n_steps)

        elif type == 'random':
            weiner = drift * self.random_distrib_walk(n_steps=n_steps)

        else:
            #initialize array so that python doesn't get mad at me
            weiner = np.ones(n_steps)
            msg = f'Simulation type "{type}" has not yet been implemented.'
            raise ValueError(msg)


        if pricing_func:
            #TODO: looking at ways of passing result this through a pricing function
            pass

        else:
            return s0 * (np.exp(variance + weiner))



def main():
    bm = Brownian()
    initial_price = 0.05
    drift = 1
    time_period = 52
    slice = 0.5
    vol = 0.65

    sns.set()
    for i in range(1,6):
        plt.plot(bm.asset_performance(None, initial_price, vol, drift,
                                    time_period, slice, 'normal'))

    plt.show()

if __name__ == '__main__':
    main()
