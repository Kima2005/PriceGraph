#!/usr/bin/env python
# encoding: utf-8
import os
import sys
import math
import json
import pickle
import random
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
from multiprocessing import Pool


PWD = os.path.dirname(os.path.realpath(__file__))


def load_stock(s):
    df = pd.read_csv(os.path.join('../data', s), index_col=0)
    df.set_index(df.index.astype('str'), inplace=True)
    return df


def load_ci(f, xi='close'):
    with open(os.path.join('../CI', xi, '%s.json' % f[:-4])) as fp:
        return json.load(fp)


def load_embedding(f, xi='close', ti=None):
    with open(os.path.join('../Struc2vec', xi, '%s.json' % f[:-4])) as fp:
        j = json.load(fp)
    if ti is not None:
        return {d: j[d] for d in ti if d in j}
    return j


def z_score(df):
    return (df - df.mean()) / df.std()


def stock_sample(input_):
    s, d = input_
    T = 7
    df = global_df[s]
    if d not in df.index:
        return
    iloc = list(df.index).index(d) + 1
    if iloc < T:  # not enough history data
        return
    # Debug statement to print iloc and dataframe size
    print(f"Processing stock: {s}, date: {d}, iloc: {iloc}, df length: {len(df)}")

    if iloc + target - 1 >= len(df):
        print(f"Index out of bounds for stock: {s}, date: {d}, iloc: {iloc}")
        return
    
    xss = {}
    for xi in x_column:
        if xi not in df.columns:
            print(f"Column {xi} not found in stock: {s}")
            return
        # t
        t = 1 if df.iloc[iloc+target-1,:][xi] > df.loc[d, xi] else 0
        # y
        y = df.iloc[iloc-T:iloc][xi].copy()
        yz = np.array(z_score(y))
        if np.isnan(yz).any():
            return
        # ems
        ems = global_ems[s][xi]
        if d not in ems:
            return
        keys = ['%s' % i for i in range(T)]
        emd = np.array([ems[d][k] for k in keys])
        if len(emd) < T:
            return
        # ci
        cis = global_ci[s][xi]
        if d not in cis:
            return
        cid = cis[d]
        cid = [cid[str(i)] for i in range(T)]
        ciz = np.array(z_score(np.array(cid)))
        if np.isnan(ciz).any():
            ciz = np.array(cid)
        xss['%s_ems' % xi] = emd
        xss['%s_ys' % xi] = yz
        xss['%s_cis' % xi] = ciz
        xss['%s_t' % xi] = t
    return s, d, \
           xss['Close_t'], xss['Close_ems'], xss['Close_ys'], xss['Close_cis'], \
           xss['Open_t'], xss['Open_ems'], xss['Open_ys'], xss['Open_cis'], \
           xss['High_t'], xss['High_ems'], xss['High_ys'], xss['High_cis'], \
           xss['Low_t'], xss['Low_ems'], xss['Low_ys'], xss['Low_cis'], \
           xss['Volume_t'], xss['Volume_ems'], xss['Volume_ys'], xss['Volume_cis']



def sample_by_dates(dates):
    files = os.listdir('../data')
    fds = [(f, d) for d in dates for f in files]
    pool = Pool()
    samples = pool.map(stock_sample, fds)
    pool.close()
    pool.join()
        # Add debug statement to check the length of samples
    print(f"Number of samples generated: {len(samples)}")

    samples = list(filter(lambda s: s is not None, samples))
    
    # Add debug statement to check the length of valid samples
    print(f"Number of valid samples: {len(samples)}")

    if len(samples) == 0:
        print("No valid samples found.")
        return {}

    samples = filter(lambda s: s is not None, samples)
    stocks, days, \
    Close_t, Close_ems, Close_ys, Close_cis, \
    Open_t, Open_ems, Open_ys, Open_cis, \
    High_t, High_ems, High_ys, High_cis, \
    Low_t, Low_ems, Low_ys, Low_cis, \
    Volume_t, Volume_ems, Volume_ys, Volume_cis = zip(*samples)
    # amount_t, amount_ems, amount_ys, amount_cis = zip(*samples)
    return {'stock': np.array(stocks), 'day': np.array(days),
            'Close_t': np.array(Close_t), 'Close_ems': np.array(Close_ems), 'Close_ys': np.array(Close_ys), 'Close_cis': np.array(Close_cis),
            'Open_t': np.array(Open_t), 'Open_ems': np.array(Open_ems), 'Open_ys': np.array(Open_ys), 'Open_cis': np.array(Open_cis),
            'High_t': np.array(High_t), 'High_ems': np.array(High_ems), 'High_ys': np.array(High_ys), 'High_cis': np.array(High_cis),
            'Low_t': np.array(Low_t), 'Low_ems': np.array(Low_ems), 'Low_ys': np.array(Low_ys), 'Low_cis': np.array(Low_cis),
            'Volume_t': np.array(Volume_t), 'Volume_ems': np.array(Volume_ems), 'Volume_ys': np.array(Volume_ys), 'Volume_cis': np.array(Volume_cis)}


def generate_data_year(year):
    global global_ems
    start_date = datetime(year, 1, 1)
    days = [(start_date+timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
    days = [d for d in days if '%s0101' % year <= d <= '%s1231' % year]
    global_ems = {f: {xc: load_embedding(f, xc, days) for xc in x_column} for f in files}
    dataset = sample_by_dates(days)
    with open(os.path.join('../dataset', '%s.pickle' % year), 'wb') as fp:
        pickle.dump(dataset, fp)


def generate_data_season(year, season):
    global global_ems
    sm, em = str((season - 1) * 3 + 1).zfill(2), str(season * 3).zfill(2)
    print("sm is: ", sm)
    print("em is: ", em)
    sm = int(sm)
    em = int(em)
    start_date = datetime(year, sm, 1)
    days = [(start_date+timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
    days = [d for d in days if '%s%s01' % (year, str(sm).zfill(2)) <= d <= '%s%s31' % (year, str(em).zfill(2))]
    print("Filtered days:")  # Debugging statement
    print(days)  # Debugging statement
    global_ems = {f: {xc: load_embedding(f, xc, days) for xc in x_column} for f in files}
    
    dataset = sample_by_dates(days)
    with open(os.path.join('../dataset', '%s_S%s.pickle' % (year, season)), 'wb') as fp:
        pickle.dump(dataset, fp)



if __name__ == '__main__':
    files = os.listdir('../data')
    if not os.path.exists('../dataset'):
        os.makedirs('../dataset')
    x_column = ['Open','High','Low','Close','Volume']
    y_column = 'close'
    target = 1
    global_ems = None
    global_df = {f: load_stock(f) for f in files}
    global_ci = {f: {xc: load_ci(f, xc) for xc in x_column} for f in files}

    # for y in range(2021, 2012, -1):
    #     print(y)
    #     generate_data_year(y)
    for m in range(1, 5):
        print(m)
        generate_data_season(2022, m)

