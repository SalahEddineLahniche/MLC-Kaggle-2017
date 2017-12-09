from collections import defaultdict
import functools
import re

import numpy as np

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


class Processer:
    def __init__(self, mapper, pre_mapper, post_mapper, header_transformer):
        self.mapper = mapper
        self.pre_mapper = pre_mapper
        self.post_mapper = post_mapper
        self.htransform = header_transformer
    
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


    def process(self, f, g, header=True, debug=False, length=None):
        if header:
            head = next(f)
            g.write(self.htransform(head))
        if debug:
            index=0
            print('Processing...')
        for line in f:
            pline = self.parse_line(line)
            pre_pobjects = self.gmap(self.pre_mapper, pline, iterator=True)
            pobjects = self.gmap(self.mapper, pre_pobjects, iterator=True)
            post_pobjects = self.gmap(self.post_mapper, pobjects, iterator=True)
            g.write(','.join(post_pobjects) + "\n")
            if debug:
                index += 1
                print('Line {index} is processed'.format(index=index))
            if length:
                if index >= length:
                    break
        if debug:
            print('Finished !')

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
    
f = lambda t: t[1][-100:]

# the different mapping functions given the raw features, for the needs of model1:
# basically in this case evaluate the mean for 4 features
# with the default behaviour of doing nothing for the others
MAPPING_FUNCTIONS = defaultdict(lambda: (lambda t: t[1]))
MAPPING_FUNCTIONS.update(
    eeg=f,
    respiration_x=f,
    respiration_y=f,
    respiration_z=f,
)

pr = Processer(DEFAULT_MAPPER, get_mapper(DEFAULT_PRE_MAPPING_FUNCTIONS),
                              lambda t: flatten(t[1]), transform_header({
                                          '': 'index',
                                          'eeg': 99,
                                          'respiration_x': 99,
                                          'respiration_y': 99,
                                          'respiration_z': 99
                                      }))

@timed
def main():
    with open(TRAIN_PATH) as f:
        with open('data/yes22222.csv', 'w') as g:
            pr.process(f, g, debug=True, length=3)
            
main()