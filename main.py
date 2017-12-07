"""Authors: Salah&Yassir"""
import re

TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'

CSV_REGEXP = re.compile(r"(?<=^|,)(\"(?:[^\"]|\"\")*\"|[^,]*)")

def read_n_lines(n: int, new_file: str):
    with open(TRAIN_PATH) as f:
        with open(new_file, "w") as g:
            for i in range(n):
                g.write(next(f))



if __name__ == '__main__':
    read_n_lines(100, 'data/truncated_training_dataset.csv')
