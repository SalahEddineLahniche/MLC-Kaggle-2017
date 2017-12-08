"""Authors: Salah&Yassir"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



from utils import *
import myParser as pr
import model1 as m1

def read_n_lines(n: int, new_file: str):
    with open(TRAIN_PATH) as f:
        with open(new_file, "w") as g:
            if n>0:
                for i in range(n):
                    g.write(next(f))
            else:
                for line in f:
                    g.write(line)
                    

@timed
def main():
    m1.process('data/model1DS_real.csv')
    # read_n_lines(-1, 'data/truncated_training_dataset.csv')

#with open('data/truncated_training_dataset.csv') as f:
#    columns=next (f).split(',') #skipping the first line
#    line1=next(f)
#    pline = pr.parse_line(line1)
#    pobjects = pr.to_python_objects(pline)
#  
#    plt.plot(pobjects[pr.COLUMNS_INDEXES['respiration_x']])
#    res_x=np.mean(pobjects[pr.COLUMNS_INDEXES['eeg']])
#    print(res_x)
#    plt.plot(pobjects[pr.COLUMNS_INDEXES['respiration_z']])
#    plt.plot(pobjects[pr.COLUMNS_INDEXES['respiration_y']])
#    #print(pobjects)
#    # df = pd.DataFrame(line2, index=columns)
#    # print(df.head())
#df=pd.read_csv("data/model1DS.csv")
#plt.plot(df.eeg)
#
#
#if __name__ == '__main__':
#    pass
#    #main()
