import numpy as np
import pandas as pd
from sklearn import linear_model, ensemble, svm, neural_network, neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
import xgboost 


class Regressor:
    def __init__(self, session, train_df, test_df=None, dcols=None, model=None, **kwargs):
        self.df = train_df
        self.session = session
        self.tdf = test_df
        self.dcols = dcols if dcols else []
        if type(model) == type('') or not model:
            if (model == "linear") or model == "l":
                self.model = linear_model.LinearRegression(**kwargs)
            elif (model == "random_forst") or model == "rf":
                self.model = ensemble.RandomForestRegressor(**kwargs)
            elif model == "svr":
                self.model = svm.SVR(**kwargs)
            elif model == "nn":
                self.model = neural_network.MLPRegressor(**kwargs)
            elif model == "knn":
                self.model = neighbors.KNeighborsRegressor(**kwargs)
            elif model == "lasso":
                self.model = linear_model.Lasso(**kwargs)
            elif model == "en":
                self.model = linear_model.ElasticNet(**kwargs)
            elif model == "xgb":
                self.model = xgboost.XGBRegressor(**kwargs)
            elif model == 'gb':
                params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
                          'learning_rate': 0.01, 'loss': 'ls'}
                self.model = ensemble.GradientBoostingRegressor(**params)
            self.fit = True
        else:
            self.fit = False
            self.model = model

        
    def cross_validate(self, length=None, test_size=0.2, fit=True):
        X = self.df.drop(labels=(['power_increase'] + self.dcols), axis=1)[:length].as_matrix()
        y = self.df['power_increase'][:length].as_matrix()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
        if fit and self.fit:
            self.model.fit(X_train, y_train)
            pickle.dump(self.model, self.session.modelf)
            self.fit = False
        y_pred = self.model.predict(X_test)
        return mean_squared_error(y_test, y_pred)

    
    def predict(self):
        X = self.df.drop(labels=(['power_increase'] + self.dcols), axis=1).as_matrix()
        y = self.df['power_increase'].as_matrix()
        rX = self.tdf.drop(self.dcols, axis=1).as_matrix()
        if self.fit:
            self.model.fit(X, y)
            pickle.dump(self.model, self.session.modelf)
        y_pred = self.model.predict(rX)
        return y_pred
