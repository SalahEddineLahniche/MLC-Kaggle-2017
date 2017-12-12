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
    T = 1/250
    x = np.linspace(0.0, N*T, N)
    yf = scipy.fftpack.fft(y)
    xf = scipy.fftpack.fftfreq(N,T)
    xs = scipy.fftpack.fftshift(xf)
    yshift = scipy.fftpack.fftshift(yf)
    print(yf)
    fig, ax = plt.subplots()
    plt.xlim(-8, 8)
    a=np.array(yshift)
    a=np.abs(yshift)
    
    b=np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True]
    b=a[b]
    b.sort()
    
    ax.plot(xs,1.0/N*np.abs(yshift))
    print(b)
    plt.show()
    