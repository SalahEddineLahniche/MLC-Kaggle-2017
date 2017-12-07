from collections import defaultdict
import myParser as pr
from utils import *

mean = lambda t: (t[0], sum(t[1]) / len(t[1]))

MAPPING_FUNCTIONS = defaultdict(lambda: (lambda t: t))
MAPPING_FUNCTIONS.update(
    eeg=mean,
    respiration_x=mean,
    respiration_y=mean,
    respiration_z=mean
)

mapper = lambda t: MAPPING_FUNCTIONS[reverse_dict(pr.COLUMNS_INDEXES)[t[0]]](t)

def gmap(pobjects):
    return list(map(lambda t: t[1], map(mapper, enumerate(pobjects))))

def cleaner(filename, newfilename):
    index=0
    print("starting...")
    with open(filename) as f:
        with open(newfilename, "w") as g:
            g.write(next(f))
            for line in f:
                index += 1
                pline = pr.parse_line(line)
                pobjects = pr.to_python_objects(pline)
                pobjects = gmap(pobjects)
                npline = map(str, pobjects)
                nline = ','.join(npline)
                g.write(nline + "\n")
                print("{index:5d} out of {tot:5d}".format(index=index, tot=50000))