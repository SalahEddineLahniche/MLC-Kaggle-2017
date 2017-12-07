import re
from utils import *
from collections import defaultdict

# the features structure as specified the the csv files
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

# the different mapping functions given the feature: basically in this case evaluate the list for 4 features
# with the default behaviour of a float value
MAPPING_FUNCTIONS = defaultdict(lambda: (lambda t: (t[0], float(t[1].strip()))))
MAPPING_FUNCTIONS.update(
    eeg=lambda t: (t[0], eval(t[1][1:-1])),
    respiration_x=lambda t: (t[0], eval(t[1][1:-1])),
    respiration_y=lambda t: (t[0], eval(t[1][1:-1])),
    respiration_z=lambda t: (t[0], eval(t[1][1:-1]))
)

# given a tuple of index, value apply the corresponding function to the value(passing the tuple as arg)
# and return the changed tuple
mapper = lambda t: MAPPING_FUNCTIONS[reverse_dict(COLUMNS_INDEXES)[t[0]]](t)

# the regular expression to parse a csv line even if it contains expressions between double quotes
CSV_REGEXP = re.compile(r"(?:(?<=,)|(?<=^))(\"(?:[^\"]|\"\")*\"|[^,]*)")

'''
return the different csv elements using the regular expression csv_regexp
'''
def parse_line(line):
    return list(map(str, CSV_REGEXP.findall(line)))

'''
this function returns a parsed line to the corresponding python objects:

the inner map apply a given function that depend on the index to each element of the parsedline
and this dependance on the function is given by a generic mapping function `mapper`
that applies a function in mapping_functions on the tuple index, value and return index(unchanged), value(mapped)

the second map get rid of the indexes and return only the values
'''
def to_python_objects(parsed_line):
    return list(map(lambda t: t[1], map(mapper, enumerate(parsed_line))))


