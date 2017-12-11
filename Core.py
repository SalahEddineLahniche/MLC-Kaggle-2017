from collections import defaultdict
import functools
import re
import os
import os.path

import numpy as np
import pandas as pd
from sklearn import linear_model, ensemble, svm
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'
CSV_REGEXP = re.compile(r"(?:(?<=,)|(?<=^))(\"(?:[^\"]|\"\")*\"|[^,]*)")
COLUMNS_INDEXES = {
    'index': 0,
    'eeg': 1,
    'night': 9,
    'number_previous': 6,
    'power_increase': 10,
    'respiration_x': 2,
    'respiration_y': 3,
    'respiration_z': 4,
    'time': 7,
    'time_previous': 5,
    'user': 8
}

DEFAULT_PRE_MAPPING_FUNCTIONS = defaultdict(lambda: (lambda t: float(t[1].strip())))
DEFAULT_PRE_MAPPING_FUNCTIONS.update(
    eeg=lambda t: eval(t[1][1:-1]),
    respiration_x=lambda t: eval(t[1][1:-1]),
    respiration_y=lambda t: eval(t[1][1:-1]),
    respiration_z=lambda t: eval(t[1][1:-1])
)
DEFAULT_MAPPER = lambda t: t[1]

'''
this function reverse a dictionary given the condition that both the keys and the values are unique
'''
def reverse_dict(d):
    return {d[key]: key for key in d}

'''
this decorator time the execution of a given function
'''
def timed(f):
    import time
    @functools.wraps(f)
    def wrapped(*args):
        t=time.time() # get the current time
        f(*args)
        # print the current time - the recorded time (which is the elapsed time in seconds)
        print("Time of execution is: {t:.0f} s".format(t=(time.time()) - t))
    return wrapped

def formatted_now():
    import datetime as dt
    return dt.datetime.strftime(dt.datetime.today(), '%y%m%d-%H%M%S')

class Session:
    def __init__(self, debug=True):
        self.dir = formatted_now()
        os.mkdir('data/{d}'.format(d=self.dir))
        self.log = 'log.txt'
        self.rslts = 'rslts.txt'
        self.train = 'train.csv'
        self.test = 'test.csv'
        self.debug = debug
        self.log_format = '[{date}] {msg}'

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, type, value, traceback):
        if self.logf:
            self.logf.close()
        if self.rsltsf:
            self.rsltsf.close()
        if self.trainf:
            self.trainf.close()
        if self.testf:
            self.logf.close()
        

    def get_log_filename(self):
        return '{d}/{s}'.format(d=self.dir, s=self.log)

    def get_results_filename(self):
        return '{d}/{s}'.format(d=self.dir, s=self.rslts)

    def get_train_filename(self):
        return '{d}/{s}'.format(d=self.dir, s=self.train)

    def get_test_filename(self):
        return '{d}/{s}'.format(d=self.dir, s=self.test)

    def init(self):
        self.logf = open(self.get_log_filename(), 'w')
        self.rsltsf = open(self.get_results_filename(), 'w')

    def init_train(self):
        self.trainf = open(self.get_train_filename(), 'w')

    def init_test(self):
        self.testf = open(self.get_test_filename(), 'w')

    def log(self, msg, rslts=False):
        if not rslts:
            print(self.log_format.format(msg=msg, date=formatted_now()), file=self.logf)
        else:
            print(self.log_format.format(msg=msg, date=formatted_now()), file=self.rsltsf)
        if self.debug:
            print(self.log_format.format(msg=msg, date=formatted_now()))

class Processer:
    def __init__(self, mapper, pre_mapper, post_mapper, header_transformer, session):
        self.mapper = mapper
        self.pre_mapper = pre_mapper
        self.post_mapper = post_mapper
        self.htransform = header_transformer
        self.session = session
    '''
    return the different csv elements using the regular expression csv_regexp
    '''        
    def parse_line(self, line):
        return list(map(str, CSV_REGEXP.findall(line)))

    '''
    this generalized map function transform a python structure of one sample to fit our needs :

    the inner map apply a given function that depend on the index to each element of the parsedline
    and this dependance on the function is given by a generic mapping function `mapper`
    that applies a function in mapping_functions on the tuple index, value and return index(unchanged), value(mapped)

    the second map get rid of the indexes and return only the values
    '''
    def gmap(self, mapper, pobjects, iterator=False):
        if iterator:
            return map(mapper, enumerate(pobjects))
        return list(map(mapper, enumerate(pobjects)))


<<<<<<< HEAD
    def process(self, f, g, header=True, debug=False, length=1000):
=======
    def process(self, f, g, header=True, debug=False, length=None, offset=0):
>>>>>>> d0d1a973286358bc3cebd44b823b3d7da4332a55
        if header:
            head = next(f)
            g.write(self.htransform(head))
        index=0
        if debug:
            self.session.log('Processing...')
        for line in f:
            if index < offset:
                continue
            pline = self.parse_line(line)
            pre_pobjects = self.gmap(self.pre_mapper, pline, iterator=True)
            pobjects = self.gmap(self.mapper, pre_pobjects, iterator=True)
            post_pobjects = self.gmap(self.post_mapper, pobjects, iterator=True)
            g.write(','.join(post_pobjects) + "\n")
            index += 1
            if debug:
                self.session.log('Line {index} is processed'.format(index=index))
            if length:
                if index >= offset + length:
                    break
        if debug:
            self.session.log('Finished processing !')

def get_mapper(mapping_functions):
            return (lambda t: mapping_functions[reverse_dict(COLUMNS_INDEXES)[t[0]]](t))

def flatten(obj):
    if type(obj) == type([]):
        return str(obj)[1:-1]
    return str(obj)

def clean_element(el, d):
    el = el.strip()
    if el in d:
        if type(d[el]) == type(0):
            el = ",".join([(el + "_" + str(i)) for i in range(d[el])])
        elif type(d[el]) == type(''):
            el = d[el]
    return el

def transform_header(d):
    def transformer(header):
        lst = header.split(',')
        lst = list(map(lambda el: clean_element(el, d), lst))
        return ','.join(lst) + "\n"
    return transformer

DEFAULT_PROCESSER = Processer(DEFAULT_MAPPER, get_mapper(DEFAULT_PRE_MAPPING_FUNCTIONS),
                              lambda t: flatten(t[1]), transform_header({
                                          '': 'index',
                                          'eeg': 2000,
                                          'respiration_x': 400,
                                          'respiration_y': 400,
                                          'respiration_z': 400
                                      }))

class Regressor:
    def __init__(self, train_df, test_df=None, dcols=None, model=None, **kwargs):
        self.df = train_df
        self.tdf = test_df
        self.dcols = dcols if dcols else []
        if type('model') == type(''):
            if (model == "linear") or model == "l":
                self.model = linear_model.LinearRegression(**kwargs)
            elif (model == "random_forst") or model == "rf":
                self.model = ensemble.RandomForestRegressor(**kwargs)
            elif model == "svr":
                self.model = svm.SVR(**kwargs)
            elif model=="gb":
                self.model = ensemble.GradientBoostingRegressor(**kwargs)
            elif model=="ad":
                self.model = ensemble.AdaBoostRegressor(**kwargs)
            elif model=="ad":
                self.model = ensemble.AdaBoostRegressor(**kwargs)
                
        else:
            self.model = model

        
    def cross_validate(self, length=None, test_size=0.2):
<<<<<<< HEAD
        X = self.df.drop(labels=['power_increase','night','time','time_previous','user'], axis=1)[:length].as_matrix()
=======
        X = self.df.drop(labels=(['power_increase'] + dcols), axis=1)[:length].as_matrix()
>>>>>>> d0d1a973286358bc3cebd44b823b3d7da4332a55
        y = self.df['power_increase'][:length].as_matrix()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        return mean_squared_error(y_test, y_pred)

    
    def predict(self):
        X = self.df.drop(labels=(['power_increase'] + dcols), axis=1).as_matrix()
        y = self.df['power_increase'].as_matrix()
        rX = self.tdf.as_matrix()
        self.model.fit(X, y)
        y_pred = self.model.predict(rX)
        return y_pred

class model:
    def __init__(self, processer, session, offset=0, dcols=None, length=None, model=None, **kwargs):
        self.pr = processer
        self.session = session
        self.model = model
        self.offset = offset
        self.length = length
        self.m_args = kwargs

    def run(self, cross_validate=False, processed_train_data=None, processed_test_data=None):
        if not processed_train_data:
            self.session.log("--{file}--".format(file=TRAIN_PATH))
            tmp_train = self.session.get_train_filename
            self.session.log("--{file}--".format(file=tmp_train))
            with open(TRAIN_PATH) as f:
                with open(tmp_train, 'w') as g:
<<<<<<< HEAD
                    self.pr.process(f, g, debug=self.debug)
=======
                    self.pr.process(f, g, debug=self.session.debug, length=self.length, offset=self.offset)
>>>>>>> d0d1a973286358bc3cebd44b823b3d7da4332a55
            processed_train_data = tmp_train
        if not processed_test_data and not cross_validate:
            self.session.log("--{file}--".format(file=TEST_PATH))
            tmp_test = self.session.get_test_filename
            self.session.log("--{file}--".format(file=tmp_test))
            with open(TEST_PATH) as f:
                with open(tmp_test, 'w') as g:
<<<<<<< HEAD
                    self.pr.process(f, g, debug=self.debug)
=======
                    self.pr.process(f, g, debug=self.session.debug, length=self.length, offset=self.offset)
>>>>>>> d0d1a973286358bc3cebd44b823b3d7da4332a55
            processed_test_data = tmp_test
        self.df = pd.read_csv(processed_train_data)
        if not cross_validate:
            self.tdf = pd.read_csv(processed_test_data)
        else:
            self.tdf = None
<<<<<<< HEAD
        reg = Regressor(self.df, self.tdf, self.model, **self.m_args)
=======
        reg = Regressor(self.df, self.tdf, self.dcols, self.model, self.m_args)
>>>>>>> d0d1a973286358bc3cebd44b823b3d7da4332a55
        if cross_validate:
            return reg.cross_validate()
        else:
            return reg.predict()

        