# useful libraries
import os
import os.path
import re
import csv
# Let's define the big contants :
TRAIN_PATH = 'results/train/momen.csv'
TEST_PATH = 'data/test.csv'
RESULTS_DIR = 'results'
RESULTS_TRAIN_DIR = os.path.join(RESULTS_DIR, 'train')
EMBEDED_LIST_COLUMN_RE = re.compile(r'(?P<base>[a-zA-Z_\d]+)(?!_\d)(_(?P<index>\d+))?')
# First we need to explore the train data that we're given
# To do that first we need to tweak the data a little bit so that we can easily read it and visualise it
with open(TRAIN_PATH, 'r') as train_file:
    # read the head
    head =  next(train_file)
    files = {}
    columns = list(map(str.strip, head.split(',')))
    # Create the appropriate files and retain the handles:
    for column in columns:
        match = EMBEDED_LIST_COLUMN_RE.match(column)
        if match:
            if match.group('base') not in files:
                files[match.group('base')] = {}
                files[match.group('base')]['columns'] = [column]
                relative_column_path = os.path.join(RESULTS_TRAIN_DIR, match.group('base') + '.csv')
                files[match.group('base')]['handle'] = open(relative_column_path, 'w', newline='')
            else:
                files[match.group('base')]['columns'] += [column]
    for value in files.values():
        value['handle'] = csv.DictWriter(value['handle'], fieldnames=value['columns'], extrasaction="ignore")
        value['handle'].writeheader()
    reader = csv.reader(train_file)
    for index, row in enumerate(reader):
            objs = dict(zip(columns, map(lambda x: x, row)))
            for file in files.values():
                file['handle'].writerow(objs)
            print('line {index}'.format(index=index), end='\r')
    print('Finished !')
        
            
            

