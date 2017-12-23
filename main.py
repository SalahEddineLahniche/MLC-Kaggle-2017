"""Authors: Salah&Yassir"""
import functools
import numpy as np
import pandas as pd
import pickle as pk
import ABONO as abono

# dir xs list dial les colonnes li bghit tapliqui 3lihom
xs = ['eeg_{i}'.format(i=i) for i in range(0, 2000)]
# defini fonction:
def f(objs):
    s = 0
    for x in xs:
        v = objs[x]
        # hna dir traitement 3la list
        s += v
    return s



p8 = lambda x: (lambda objs: objs[x] ** 8)

mapper = {
    'eeg_sum': f #dir smyat lcolonne jdida : fonction    
}

for x in xs:
    mapper[x] = p8(x)


newcols = list(mapper.keys())

dropcols = ['eeg_{i}'.format(i=i) for i in range(0, 1900)] + \
            ['respiration_x_{i}'.format(i=i) for i in range(0, 395)] + \
            ['respiration_y_{i}'.format(i=i) for i in range(0, 395)] + \
            ['respiration_z_{i}'.format(i=i) for i in range(0, 395)] + \
            ['user', 'night']


with abono.Session() as s: #Debug is true
    prr = 'data/171212-161438/train.csv'
    prrr = 'data/train.csv'
    prr2 = 'data/171212-170858/test.csv'
    mm = 'data/171212-161438/model.dat'
    s.init_train()
    s.init_model()
    s.init_test()
    pr = abono.Processer(s, newcols, mapper, dropcols)
    # with open(mm, 'rb') as ff:
    #     model = pk.load(ff)
    m = abono.model(pr, s, offset=0, length=None, model='en')#, model=model)
    @abono.timed(s)
    def main():
        return m.run(cross_validate=True, processed_train_data=prrr)#, processed_test_data=prr2) # you can add the processed train set path here
    rslt = main()
    if type(rslt) == np.float64:
        s.log('MSE: {mse}'.format(mse=rslt**0.5), rslts=True)
    else:
        s.log(rslt[1]**0.5)
        pd.DataFrame(rslt[0]).to_csv(s.rsltsf)

# with open('bull.txt') as f:
#     l = eval(f.read())

# import matplotlib.pyplot as plt
# import numpy as np
# # Learn about API authentication here: https://plot.ly/python/getting-started
# # Find your api_k ey here: https://plot.ly/settings/api

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
