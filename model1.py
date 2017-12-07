from collections import defaultdict
import myParser as pr
from utils import *

mean = lambda t: sum(t) / len(t)

MAPPING_FUNCTIONS = defaultdict(lambda t: t)
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
    with open(filename) as f:
        with open(newfilename, "w") as g:
            for line in f:
                pline = pr.parse_line(line)
                pobjects = pr.to_python_objects(pline)
                pobjects = gmap(pobjects)
                pline = map(str, pobjects)
                line = ','.join(pline)
                g.write(line + "\n")