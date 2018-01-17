"""Authors: Salah&Yassir"""
import functools
import numpy as np
import pandas as pd
import ABONO as abono
from scipy.stats import kurtosis, skew, moment

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

def fourrier_related_features(objs):
    y = [objs[x] for x in xs_eeg]
    loop = [[i for i in range((j * 250), (j + 1) * 250)] for j in set([range(1, 4)])]
    Y = np.fft.fft(y)/n # fft computing and normalization
    m = [-1] * 8 # maximum
    gm = -1 # g maximum
    s_abs_window = [0] * 4 # sum of absolute value
    s_abs = 0 # g sum of absolute value
    s_sq_window = [0] * 4 # sum of squares
    s_sq = 0 # g sum of squares
    j = [-1] * 4
    g_j = -1
    for index, window in enumerate(loop):
        for i in window:
            gm = max(Y[i], m)
            m[index] = max(Y[i], m)
            s_abs_window[index] += Y[i]
            s_sq_window[index] += Y[i] ** 2
            if m[index] == Y[i]:
                j[index] = i
            if gm == Y[i]:
                g_j = i
    s_abs = sum(s_abs_window)
    s_sq = sum(s_sq_window)

    d = {}
    d['delta'] = s_abs_window[0]
    d['theta'] = s_abs_window[1]
    d['alpha'] = s_abs_window[2]
    d['beta'] = s_abs_window[3]
    d['sum_f_hat'] = s_abs
    d['sum_f_hat_sq'] = s_sq
    d['f_hat_std'] = Y.std()
    d['fonda'] = g_ * (1 / 250)

    return d

def time_series_related_features(objs, xs, f):
    y = [objs[x] for x in xs_eeg]
    y = np.array(y)

    d = {}
    d['kurtosis'] = kurtosis(y)
    d['skew'] = skew(y)
    d['std'] = y.std()
    d['mean'] = y.mean()
    d['sum_abs'] = sum(map(abs, y))
    d['sum_sq'] = sum(map(lambda x: x ** 2, y))
    d['moment3'] = moment(y, moment=3)
    d['moment4'] = moment(y, moment=3)

    return d

mapper = {}
convoluted_mapper = [fourrier_related_features, time_series_related_features]

newcols = list(mapper.keys())

dropcols = ['eeg_{i}'.format(i=i) for i in range(0, 2000)] + \
            ['respiration_x_{i}'.format(i=i) for i in range(0, 400)] + \
            ['respiration_y_{i}'.format(i=i) for i in range(0, 400)] + \
            ['respiration_z_{i}'.format(i=i) for i in range(0, 400)] + \
            ['user', 'night']


with abono.Session() as s: #Debug is true
    prr = 'data/171212-161438/train.csv'
    prr2 = 'data/171212-170858/test.csv'
    mm = 'data/171212-161438/model.dat'
    s.init_train()
    s.init_model()
    s.init_test()
    pr = abono.Processer(s, newcols, mapper, dropcols, convoluted_mappers)
    with open(mm, 'rb') as ff:
        model = pk.load(ff)
    m = abono.model(pr, s, offset=0, length=None, model='gb')#, model=model)
    @abono.timed(s)
    def main():
        return m.run(cross_validate=True, processed_train_data=prr)#, processed_test_data=prr2) # you can add the processed train set path here
    rslt = main()
    if type(rslt) == np.float64:
        s.log('MSE: {mse}'.format(mse=rslt), rslts=True)
    else:
        s.log(rslt[1])#**0.5)
        pd.DataFrame(rslt[0]).to_csv(s.rsltsf)

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
