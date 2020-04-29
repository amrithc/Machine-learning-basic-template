# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 17:51:30 2020

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
#SPLITING THE DATA INTO TEST AND TRAIN
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.25, random_state=0)

#SCALING THE FEATURES 
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_test=sc_X.fit_transform(X_test)
X_train=sc_X.fit_transform(X_train)

#FITTING THE REQUIRED MODEL

#K- NEAREST NEIGHBORS ALGORITHM
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=5,metric='minkowski', p=2)
model.fit(X_train,y_train)

#SUPPORT VECTOR MACHINE ALGORITHM
from sklearn.svm import SVC
model= SVC(kernel='linear',random_state=0)
model.fit(X_train,y_train)

#NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
model= GaussianNB()
model.fit(X_train,y_train)

#DECISION TREE ALGORITHM
from sklearn.tree import DecisionTreeClassifier
model= DecisionTreeClassifier(criterion='entropy', random_state=0)
model.fit(X_train, y_train)

#RANDOM FOREST ALGORITHM
from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier(n_estimators=100, criterion='entropy',random_state=0)
model.fit(X_train,y_train)

#PREDICTION
pred=model.predict(X_test)

#EVALUATING THE MODEL
#MAKING THE CONFUSION MATRIX
from sklearn import metrics
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, pred) 

#FOR 2 DIMENTIONAL DATA ONLY
#VISUALISTAION OF TRAINING SET RESULTS
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop=X_set[:,0].max() + 1, step = 0.01),
                     np.arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max() + 1, step = 0.01))
plt.contourf(X1,X2,model.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
                c=ListedColormap(('red','green'))(i),label=j)
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.show()

#VISUALISATION OF TEST SET RESULTS
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop=X_set[:,0].max() + 1, step = 0.01),
                     np.arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max() + 1, step = 0.01))
plt.contourf(X1,X2,model.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
                c=ListedColormap(('red','green'))(i),label=j)
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.show()