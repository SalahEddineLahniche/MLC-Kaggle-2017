import functools
import numpy as np

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

def some_bullshit(y):
    import matplotlib.pyplot as plt
    import scipy.fftpack

    # Number of samplepoints
    N = len(y)
    # sample spacing
    T = 750.0
    x = np.linspace(0.0, N*T, N)
    yf = scipy.fftpack.fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

    fig, ax = plt.subplots()
    ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
    plt.show()
    