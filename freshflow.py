#!/usr/bin/env python
# coding: utf-8

# # Freshflow task <<

# Imports
import os
import numpy as np
import pandas as pd

import argparse

from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, mean_absolute_percentage_error as mape
from sklearn.model_selection import TimeSeriesSplit

from statsmodels.tsa.arima.model import ARIMA


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", help="Complete path to input file", default='./data')
    parser.add_argument("--file_name", help="Input file name")
    parser.add_argument("--item_number", help="Enter Item Number to process")

    args = parser.parse_args()
    args = vars(args)

    data_path = args["data_path"]
    fn = args["file_name"]
    ITEM_NR = args["item_number"]

    #data_path = './data'
    #fn = 'data.csv'
    #ITEM_NR = 80028349

    df = pd.read_csv(os.path.join(data_path, fn))

    df['date'] = pd.to_datetime(df.day)
    dfd = df.set_index('date')
    dfd = dfd[['item_number', 'item_name', 'purchase_price', 
               'suggested_retail_price', 'orders_quantity', 
               'sales_quantity', 'revenue']]

    dfi = dfd.loc[dfd.item_number==ITEM_NR] 
    sr = dfi.sales_quantity.values

    trn_size = int(sr.shape[0] * 0.66)
    tst_size = sr.shape[0] - trn_size

    Xtrn, Xtst = sr[:trn_size], sr[trn_size:]

    # Walking forward evaluation of an ARIMA model
    ARIMA_PARAMS = (1, 0, 4) # AR of grade 1, no differencing, MA = 4

    preds, history, results = eval_arima(Xtrn, Xtst, ARIMA_PARAMS) # let's predict at least one year
    print(results)

    return

# ### Functions

def eval_arima(Xtrn, Xtst, arima_params=(1, 0, 0), verbose=False):
    '''
    '''
    metrics = {'mse': mse, 'mae': mae, 'mape': mape}
    
    history = Xtrn.copy()
    preds = np.array([])
    for k in range(Xtst.shape[0]):
        char = k if k%10 == 0 else '.'
        print(char, end='')
        model = ARIMA(history, order=arima_params)
        model_fit = model.fit()
        if verbose:
            print(f'Using model ARIMA({model.k_ar},{model.k_diff},{model.k_ma}). ', end='')
        y_hat = model_fit.forecast()[0]
        obs = Xtst[k]
        history = np.append(history, obs)
        preds = np.append(preds, y_hat)
        if verbose:
            print(f'predicted={round(y_hat, 2)}, expected={round(obs, 2)}.')
    
    results = {metric_name: metric(Xtst, preds) for metric_name, metric in metrics.items()} 
    return preds, history, results 
