import re
from utils import *
from collections import defaultdict

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

MAPPING_FUNCTIONS = defaultdict(lambda: (lambda t: (t[0], float(t[1].strip()))))
MAPPING_FUNCTIONS.update(
    eeg=lambda t: (t[0], eval(t[1][1:-1])),
    respiration_x=lambda t: (t[0], eval(t[1][1:-1])),
    respiration_y=lambda t: (t[0], eval(t[1][1:-1])),
    respiration_z=lambda t: (t[0], eval(t[1][1:-1]))
)


mapper = lambda t: MAPPING_FUNCTIONS[reverse_dict(COLUMNS_INDEXES)[t[0]]](t)

def parse_line(line):
    return list(map(str, CSV_REGEXP.findall(line)))

def to_python_objects(parsed_line):
    return list(map(lambda t: t[1], map(mapper, enumerate(parsed_line))))


CSV_REGEXP = re.compile(r"(?:(?<=,)|(?<=^))(\"(?:[^\"]|\"\")*\"|[^,]*)")
