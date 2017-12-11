"""Authors: Salah&Yassir"""
from collections import defaultdict
import pandas as pd
import Core as core
import numpy as np

    
f = lambda t: list(map(lambda x:x, t[1][-1001:]))
G = lambda t: list(map(lambda x:x , t[1][-101:]))
G1 = lambda t: list(map(lambda x:x, t[1][-11:]))
MAPPING_FUNCTIONS = defaultdict(lambda: (lambda t: t[1]))
MAPPING_FUNCTIONS.update(
    eeg=f,
    respiration_x=G,
    respiration_y=G1,
    respiration_z=G1,
)

pr = core.Processer(core.get_mapper(MAPPING_FUNCTIONS), core.get_mapper(core.DEFAULT_PRE_MAPPING_FUNCTIONS),
                              lambda t: core.flatten(t[1]), core.transform_header({
                                          '': 'index',
                                          'eeg': 1000,
                                           'respiration_x': 100,
                                          'respiration_y': 10,
                                          'respiration_z': 10
                                      }))

m = core.model(pr, 'l', debug=True)
s = 'data/train_171210-235840.csv'
@core.timed
def main():
    print(((m.run(cross_validate=True))))
    
    

main()
