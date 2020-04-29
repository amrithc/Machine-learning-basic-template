# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 10:50:25 2020

@author: Amrith
"""

# LOAD THE REQUIRED LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#LOAD THE DATA FROM THE CSV FILE
dataset=pd.read_csv('.csv')
#SELECT THE REQUIRED FEATURES 
X=dataset.iloc[:,:].values
#y IS THE OUT CATEGORICAL CLASS (USUALLY LAST COL.)
y=dataset.iloc[:,-1].values

#DATA PREPROCESSING
#SCALING THE FEATURES (DEPENDING ON YOUR MODEL AND DATA)
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X= sc_X.fit_transform(X)
sc_y=StandardScaler()
y=sc_y.fit_transform(y)

#SPLITING THE DATA INTO TEST AND TRAIN(DEPENDING ON YOUR DATA)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.25, random_state=0)

#ENCODING CATEGORICAL DATA (DEPENDING ON YOUR DATA)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X= np.array(ct.fit_transform(X))

#FITTING THE REQUIRED MODEL

#SIMPLE LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
model= LinearRegression()
model.fit(X_train, y_train)

#MULTIPLE LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
model= LinearRegression()
model.fit(X_train, y_train)

#POLYNOMIAL REGRESSION
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
model = LinearRegression()
model.fit(X_poly, y)

#SUPPORT VECTORE MACHINE
from sklearn.svm import SVR
model=SVR(kernel='rbf')
model.fit(X,y)

#DECISION TREE
from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor(random_state= 0)
model.fit(X,y)

#RANDOM FOREST ALGORITHM
from sklearn.ensemble import RandomForestRegressor
model= RandomForestRegressor(n_estimators=300, random_state=0)
model.fit(X,y)

#PREDICTION
pred=model.predict([['YOUR VALUE']])
#FOR POLYNOMIAL REGRESSTION
pred=model.predict(poly_reg.fit_transform([['YOUR VALUE']]))
#PREDICTION IF YOU SCALED THE DATA
pred=sc_y.inverse_transform(model.predict(sc_X.transform(np.array([['YOUR VALUE']]))))

#FOR 2 DIMENTIONAL DATA ONLY
#VISUALISTAION OF TRAINING SET RESULTS
plt.scatter(X_train,y_train,c='red')
plt.plot(X_train,model.predict(X_train),color='blue')
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.show()

#VISUALISATION OF TEST SET RESULTS
plt.scatter(X_test,y_test,c='red')
plt.plot(X_test,model.predict(X_test),c='blue')
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.show()

#VISUALISATION OF RESULTS FOR HIGHER RESOLUTION AND SMOOTH CURVE
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, model.predict(X_grid), color = 'blue')
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.show()
