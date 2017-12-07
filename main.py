"""Authors: Salah&Yassir"""
import numpy as np
import pandas as pd

TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'
COLUMNS_INDEXES = {
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

def read_n_lines(n: int, new_file: str):
    with open(TRAIN_PATH) as f:
        with open(new_file, "w") as g:
            for i in range(n):
                g.write(next(f))

f=open('data/truncated_training_dataset.csv')
columns=next (f).split(',')[1:]
line1=next(f).split(',')

df = pd.DataFrame(line1, index=columns)
print(df.head())

if __name__ == '__main__':
    read_n_lines(100, 'data/truncated_training_dataset.csv')

