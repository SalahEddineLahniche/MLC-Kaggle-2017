"""Authors: Salah&Yassir"""
from collections import defaultdict
import pandas as pd

import Core as core

    
f = lambda t: t[1][-201:]
MAPPING_FUNCTIONS = defaultdict(lambda: (lambda t: t[1]))
MAPPING_FUNCTIONS.update(
    eeg=f,
    respiration_x=f,
    respiration_y=f,
    respiration_z=f,
)


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
