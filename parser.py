from utils import *

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

MAPPING_FUNCTIONS = {
    'eeg': lambda tuple: eval(tuple[1][1:-1]),
    'respiration_x': lambda tuple: eval(tuple[1][1:-1]),
    'respiration_y': lambda tuple: eval(tuple[1][1:-1]),
    'respiration_z': lambda tuple: eval(tuple[1][1:-1])
}


mapper = lambda index, value: MAPPING_FUNCTIONS[reverse_dict(COLUMNS_INDEXES)[index]]((index, value))

def parse_line(line):
    return list(map(str, CSV_REGEXP.finditer(line)))

def to_python_objects(parsed_line):
    return list(map(lambda tuple: tuple[1], map(mapper, enumerate(parsed_line))))


CSV_REGEXP = re.compile(r"(?:(?<=,)|(?<=^))(\"(?:[^\"]|\"\")*\"|[^,]*)")
