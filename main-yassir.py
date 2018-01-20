import functools
import numpy as np
import pandas as pd
import ABONO as abono
import pickle as pk

TRAIN_PATH = 'results/train/'
TEST_PATH = 'results/train/'

COLS = [
	'user', 'night', 'time_previous', 'number_previous', 'time'
	'delta', 'theta', 'alpha', 'beta', 'sum_f_hat', 'sum_f_hat_sq', 'f_hat_std', 'fonda',
	'delta_x', 'theta_x', 'alpha_x', 'beta_x', 'sum_f_hat_x', 'sum_f_hat_sq_x', 'f_hat_std_x', 'fonda_x',
	'delta_y', 'theta_y', 'alpha_y', 'beta_y', 'sum_f_hat_y', 'sum_f_hat_sq_y', 'f_hat_std_y', 'fonda_y',
	'delta_z', 'theta_z', 'alpha_z', 'beta_z', 'sum_f_hat_z', 'sum_f_hat_sq_z', 'f_hat_std_z', 'fonda_z',
	'kurtosis', 'skew', 'std', 'mean', 'sum_abs', 'sum_sq', 'moment3', 'moment4',
	'kurtosis_x', 'skew_x', 'std_x', 'mean_x', 'sum_abs_x', 'sum_sq_x', 'moment3_x', 'moment4_x',
	'kurtosis_y', 'skew_y', 'std_y', 'mean_y', 'sum_abs_y', 'sum_sq_y', 'moment3_y', 'moment4_y',
	'kurtosis_z', 'skew_z', 'std_z', 'mean_z', 'sum_abs_z', 'sum_sq_z', 'moment3_z', 'moment4_z',
	('eeg', range(1900, 2000)), ('respiration_x', [0, 2, 3]), ('respiration_y', range(50)), ('respiration_z', range(50, 100))
]

MODEL = 'gb'
PARAMS = {}

with abono.Session() as s:
	s.init_train()
	s.init_model()
	s.init_test()
	dfs = {}
	tdfs = {}
	for el in COLS:
		if type(el) == tuple:
			dfs[el] = pd.read_csv(TRAIN_PATH + el[0] + '.csv')
			dfs[el] = d[el][list(map(lambda x: el[0] + '_' + x, el[1]))]
			tdfs[el] = pd.read_csv(TEST_PATH + el[0] + '.csv')
			tdfs[el] = d[el][list(map(lambda x: el[0] + '_' + x, el[1]))]
		dfs[el] = pd.read_csv(TRAIN_PATH + el + '.csv')
		if el != 'power_increase':
			tdfs[el] = pd.read_csv(TEST_PATH + el + '.csv')

	df = pd.concat(list(dfs.values()), axis=1)
	tdf = pd.concat(list(tdfs.values()), axis=1)
	reg = abono.Regressor(s, df, tdf, model=MODEL, **PARAMS)
	mse = reg.cross_validate()
	s.log("RMSE: {mse:.02f}".format(mse ** 0.5), rslts=True)
	rslts = reg.predict()
    pd.DataFrame(rslt[0]).to_csv(s.rsltsf)
    s.log("Finished !")