# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 23:51:05 2017

@author: Yassir²
"""

from collections import defaultdict
import pandas as pd
import numpy as np
import myParser as pr
from utils import *

# the mean mapping function takes tuple(index, value) as argument, return a tuple(index, value)
mean = lambda t: (t[0], [t[1][-100:]] )

# the different mapping functions given the raw features, for the needs of model1:
# basically in this case evaluate the mean for 4 features
# with the default behaviour of doing nothing for the others
MAPPING_FUNCTIONS = defaultdict(lambda: (lambda t: t))
MAPPING_FUNCTIONS.update(
    eeg=mean,
    respiration_x=mean,
    respiration_y=mean,
    respiration_z=mean,
)

# given a tuple of index, value apply the corresponding function to the value(passing the tuple as arg)
# and return the changed tuple
mapper = lambda t: MAPPING_FUNCTIONS[reverse_dict(pr.COLUMNS_INDEXES)[t[0]]](t)

'''
this generalized map function transform a python structure of one sample to fit our needs :

the inner map apply a given function that depend on the index to each element of the parsedline
and this dependance on the function is given by a generic mapping function `mapper`
that applies a function in mapping_functions on the tuple index, value and return index(unchanged), value(mapped)

the second map get rid of the indexes and return only the values
'''
def gmap(pobjects):
    return list(map(lambda t: t[1], map(mapper, enumerate(pobjects))))

'''
the function that transform the raw Dataset to a new suitable dataset for the needs of model1
'''
def pre_process(filename, newfilename, debug=False):
    index2=0
    if debug:
        index=0
        print("starting...")
    with open(filename) as f:
        with open(newfilename, "w") as g:
            
            
            g.write(next(f))
            for line in f:
                index2+=1
                if index2<15000:
                    continue
                pline = pr.parse_line(line)
                pobjects = pr.to_python_objects(pline)
                pobjects = gmap(pobjects)
                npline = map(str, pobjects)
                nline = ','.join(npline)
                g.write(nline + "\n")
                if debug:
                    index += 1
                    print("{index:5d} out of {tot:5d}".format(index=index2, tot=50000))
                
                    
def process(filename):
    df=pd.read_csv(filename)
    new_df=df.drop(labels=['index', 'user', 'night'], axis=1)
    new_df.to_csv(filename, index=False)
    
def model(dataset):
    pass
