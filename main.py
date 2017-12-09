"""Authors: Salah&Yassir"""
from collections import defaultdict

import Core as core

    
f = lambda t: t[1][-201:]
MAPPING_FUNCTIONS = defaultdict(lambda: (lambda t: t[1]))
MAPPING_FUNCTIONS.update(
    eeg=f,
    respiration_x=f,
    respiration_y=f,
    respiration_z=f,
)

pr = core.Processer(core.get_mapper(MAPPING_FUNCTIONS), core.get_mapper(core.DEFAULT_PRE_MAPPING_FUNCTIONS),
                              lambda t: core.flatten(t[1]), core.transform_header({
                                          '': 'index',
                                          'eeg': 200,
                                          'respiration_x': 200,
                                          'respiration_y': 200,
                                          'respiration_z': 200
                                      }))

m = core.model(pr, 'linear', True)

@core.timed
def main():
    m.run(cross_validate=True)

main()
