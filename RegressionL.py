# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 19:36:05 2017

@author: Yassir²
"""

import numpy as np
import pandas as pd
import myParser as pr
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import ensemble,svm



df=pd.read_csv('data/new4.csv')
#tdf=pd.read_csv('data/new2.csv')

X=df.drop(labels=['power_increase'],axis=1).as_matrix()
Y=df['power_increase'].as_matrix()

#rX = tdf.as_matrix()

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1000)


#model=linear_model.LinearRegression()
model2=ensemble.RandomForestRegressor(n_estimators=15,max_features='sqrt')
#model3=svm.SVR()

#model.fit(X,Y)
model2.fit(X_train,Y_train)
#model3.fit(X_train,Y_train)

#Y_pred=model.predict(X_test)
#save=pd.DataFrame(Y_pred)
#save.to_csv('sm2.csv')


Y_pred3=model2.predict(X_test)
print(mean_squared_error(Y_test,Y_pred3) )








