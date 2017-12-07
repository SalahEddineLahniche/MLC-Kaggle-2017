from collections import defaultdict
import myParser as pr
from utils import *

# the mean mapping function takes tuple(index, value) as argument, return a tuple(index, value)
mean = lambda t: (t[0], sum(t[1]) / len(t[1]))

# the different mapping functions given the raw features, for the needs of model1:
# basically in this case evaluate the mean for 4 features
# with the default behaviour of doing nothing for the others
MAPPING_FUNCTIONS = defaultdict(lambda: (lambda t: t))
MAPPING_FUNCTIONS.update(
    eeg=mean,
    respiration_x=mean,
    respiration_y=mean,
    respiration_z=mean
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
def cleaner(filename, newfilename, debug=False):
    if debug:
        index=0
        print("starting...")
    with open(filename) as f:
        with open(newfilename, "w") as g:
            g.write(next(f))
            for line in f:
                pline = pr.parse_line(line)
                pobjects = pr.to_python_objects(pline)
                pobjects = gmap(pobjects)
                npline = map(str, pobjects)
                nline = ','.join(npline)
                g.write(nline + "\n")
                if debug:
                    index += 1
                    print("{index:5d} out of {tot:5d}".format(index=index, tot=50000))