"""Authors: Salah&Yassir"""
import functools
import numpy as np
import pandas as pd
import ABONO as abono

xs_eeg = [['eeg_{i}'.format(i=i) for i in range((j * 250), (j + 1) * 250)] for j in set([range(1, 8)])]
xs_rep = ['respiration_{{}}_{i}'.format(i=i) for i in range(0, 400)]

# Mean
# Sum |A|
# Sum A²
# Std
# 3rd moment
# Skewness coeff
# 4th moment
# kurtosis coef
# Sum(f_hat(0-8Hz)) F
# Sum(f_hat(4-8Hz)) beta
# Sum(f_hat(2-4Hz)) alpha
# Sum(f_hat(1-2Hz)) theta
# Sum(f_hat(0-1Hz)) delta
# beta%
# alpha%
# theta%
# delta%
# mean(F)
# mean (beta)
# mean (alpha)
# mean (theta)
# mean (delta)
# sum(|f_hat|)
# sum(f_hat²)
# Std(f_hat)



def fundamuntale_eeg(objs):
    # xs = ['eeg_{i}'.format(i=i) for i in range(0, 2000)]
    y = [objs[x] for x in xs]
    Y = np.fft.fft(y)/1000 # fft computing and normalization
    m = -1
    j = -1
    for i in range(1000):
        m = max(Y[i], m)
        if m == Y[i]:
            j = i
    return j * (1 / 250)

def fundamuntale_eeg2(objs):
    # xs = ['eeg_{i}'.format(i=i) for i in range(0, 2000)]
    y = [objs[x] for x in xs2]
    Y = np.fft.fft(y)/1000 # fft computing and normalization
    m = -1
    j = -1
    for i in range(500):
        m = max(Y[i], m)
        if m == Y[i]:
            j = i
    return j * (1 / 250)

def f_(xx):
    def fundamuntale_resp_g(objs):
        # xs = ['eeg_{i}'.format(i=i) for i in range(0, 2000)]
        y = [objs[x.format(xx)] for x in xs3]
        Y = np.fft.fft(y)/400 # fft computing and normalization
        m = -1
        j = -1
        for i in range(200):
            m = max(Y[i], m)
            if m == Y[i]:
                j = i
        return j * (1 / 50)
    return fundamuntale_resp_g

def max_eeg(objs):
    y = [abs(objs[x]) for x in xs]
    return max(y)

def max_eeg2(objs):
    y = [abs(objs[x]) for x in xs2]
    return max(y)

def g_(xx):
    def max_resp_g(objs):
        y = [abs(objs[x.format(xx)]) for x in xs]
        return max(y)
    return max_resp_g

mapper = {
    'f_eeg': fundamuntale_eeg,
    'f_eeg2': fundamuntale_eeg2,
    'f_respx': f_('x'),
    'f_respy': f_('y'),
    'f_respz': f_('z'),
    'max_eeg': max_eeg,
    'max_eeg2': max_eeg2,
    'max_respx': g_('x'),
    'max_respz': g_('z'),
    'max_respy': g_('y'),
}

newcols = list(mapper.keys())

dropcols = ['eeg_{i}'.format(i=i) for i in range(0, 2000)] + \
            ['respiration_x_{i}'.format(i=i) for i in range(0, 400)] + \
            ['respiration_y_{i}'.format(i=i) for i in range(0, 400)] + \
            ['respiration_z_{i}'.format(i=i) for i in range(0, 400)] + \
            ['user', 'night']


with abono.Session() as s: #Debug is true
    prr = 'data/171211-225354/train.csv'
    prr2 = 'data/171211-225354/test.csv'
    s.init_train()
    s.init_model()
    s.init_test()
    pr = abono.Processer(s, newcols, mapper, dropcols)
    m = abono.model(pr, s, offset=0, length=None, model='l')
    @abono.timed(s)
    def main():
        return m.run(cross_validate=False)#, processed_train_data=prr, processed_test_data=prr2) # you can add the processed train set path here
    rslt = main()
    if type(rslt) == type(.0):
        s.log('MSE: {mse}'.format(mse=rslt), rslts=True)
    pd.DataFrame(rslt).to_csv(s.rsltsf)

# with open('bull.txt') as f:
#     l = eval(f.read())

# import matplotlib.pyplot as plt
# import numpy as np
# # Learn about API authentication here: https://plot.ly/python/getting-started
# # Find your api_key here: https://plot.ly/settings/api

# Fs = 250.0;  # sampling rate
# Ts = 1.0/Fs; # sampling interval
# t = np.arange(0,8,Ts) # time vector

# y = l

# n = len(y) # length of the signal
# k = np.arange(n)
# T = n/Fs
# frq = k/T # two sides frequency range
# frq = frq[range(int(n/2))] # one side frequency range

# Y = np.fft.fft(y)/n # fft computing and normalization
# Y = Y[range(int(n/2))]

# fig, ax = plt.subplots(2, 1)
# ax[0].plot(t,y)
# ax[0].set_xlabel('Time')
# ax[0].set_ylabel('Amplitude')
# ax[1].scatter(x=frq,y=abs(Y),color='r') # plotting the spectrum
# ax[1].set_xlabel('Freq (Hz)')
# ax[1].set_ylabel('|Y(freq)|')

# # print(len(Y))
# F = abs(Y)
# n = 5
# L = [F[:1 - n]] + [F[i:1 - n + i] for i in range(1, n - 1)] + [F[n:]]

# # print(len(L))

# def d(els):
#     half = int(len(els) / 2)
#     for i in range(half):
#         if els[i] > els[i + 1]:
#             return False
#     for j in range(half + 1, len(els)):
#         if els[j - 1] < els[j]:
#             return False
#     return True

# for els in zip(*L):
#     if d(els):
#         print(els[int(n / 2)])

# ax[1].set_xlim([0, 10])
# plt.show()
