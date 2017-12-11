"""Authors: Salah&Yassir"""
from collections import defaultdict
import pandas as pd
"""<<<<<<< HEAD
=======

>>>>>>> d0d1a973286358bc3cebd44b823b3d7da4332a55"""
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

"""<<<<<<< HEAD"""
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
"""======="""

with core.Session() as s: #Debug is true
    pr_train = 'some_bullshit_path'
    pr = core.Processer(core.get_mapper(MAPPING_FUNCTIONS), core.get_mapper(core.DEFAULT_PRE_MAPPING_FUNCTIONS),
                                lambda t: core.flatten(t[1]), core.transform_header({
                                            '': 'index',
                                            'eeg': 200,
                                            'respiration_x': 200,
                                            'respiration_y': 200,
                                            'respiration_z': 200
                                        }), s)
    dcols = ['time']
    m = core.model(pr, s, offset=0, length=None, dcols=dcols, model='linear')
    @core.timed
    def main():
        return m.run(cross_validate=True) # you can add the processed train set path here
    rslt = main()
    if type(rslt) == type(.0):
        s.log(rslt, rsls=True)
    pd.DataFrame(rslt).to_csv(s.rsltsf)
""">>>>>>> d0d1a973286358bc3cebd44b823b3d7da4332a55"""
