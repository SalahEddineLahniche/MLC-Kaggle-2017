import os
import os.path
import csv
from collections import OrderedDict
from pprint import pprint
import numpy as np
import pandas as pd

RESULTS_TRAIN_DIR = 'results\\train'

def get_csv_file(filename):
    return os.path.join(RESULTS_TRAIN_DIR, filename)

def power_increase():
    df = pd.read_csv(get_csv_file('power_increase.csv'))
    df = df.describe()
    df.to_csv(get_csv_file('power_increase_summury.csv'))
    return {"pe {k}".format(k=k): v for k, v in zip(df.index, df.power_increase)}

def time():
    df = pd.read_csv(get_csv_file('time.csv'))
    df = df.describe()
    df.to_csv(get_csv_file('time_summury.csv'))
    return {"pe {k}".format(k=k): v for k, v in zip(df.index, df.time)}

def users():
    # reader = csv.DictReader(open(get_csv_file('users.csv')))
    # reader.fieldnames()
    rslts = OrderedDict()
    dfs = []
    dfs.append(pd.read_csv(get_csv_file('night.csv')))
    dfs.append(pd.read_csv(get_csv_file('power_increase.csv')))
    # dfs[1].describe().to_csv(get_csv_file('power_increase_summury.csv'))
    df = pd.concat(dfs, axis=1)
    rslts['number of unique patients'] = df.nunique()[0]
    df['total'] = np.ones(df.night.size)
    df = df.groupby('night')
    sdf = df.agg({'total': 'sum', 'power_increase': 'mean'}).sort_values(by='total')
    df.power_increase.describe().to_csv(get_csv_file('power_increase_by_night.csv'))
    rslts['contribution of 20% of patients(28)'] = int(sdf.iloc[-28:].total.sum())    
    return rslts

def print_rslts(d):
    for k, v in d.items():
        print('{name}: {value}'.format(name=k, value=str(v)))

if __name__ == '__main__':
    print_rslts(time())