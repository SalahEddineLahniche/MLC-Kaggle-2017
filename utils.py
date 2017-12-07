import functools

TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'

def reverse_dict(d):
    return {d[key]: key for key in d}

def timed(f):
    import time
    def g():
        t=time.time()
        f()
        print("le temps d'excution de f est {t}".format(t=(time.time())-t))
    return g
    