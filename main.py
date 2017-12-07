"""Authors: Salah&Yassir"""
import re
import numpy as np
import pandas as pd

from utils import *
import parser as pr

TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'


def read_n_lines(n: int, new_file: str):
    with open(TRAIN_PATH) as f:
        with open(new_file, "w") as g:
            for i in range(n):
                g.write(next(f))

with open('data/truncated_training_dataset.csv') as f:
    columns=next (f).split(',') #skipping the first line
    line1=next(f)
    pline = pr.parse_line(line1)
    pobjects = pr.to_python_objects(pline)
    print(pobjects)
    # df = pd.DataFrame(line2, index=columns)
    # print(df.head())

if __name__ == '__main__':
    read_n_lines(100, 'data/truncated_training_dataset.csv')

