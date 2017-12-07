import functools

TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'

'''
this function reverse a dictionary given the condition that both the keys and the values are unique
'''
def reverse_dict(d):
    return {d[key]: key for key in d}

'''
this decorator time the execution of a given function
'''
def timed(f):
    import time
    def g():
        t=time.time() # get the current time
        f()
        # print the current time - the recorded time (which is the elapsed time in seconds)
        print("le temps d'excution de f est {t:.0f}".format(t=(time.time()) - t))
    return g
    