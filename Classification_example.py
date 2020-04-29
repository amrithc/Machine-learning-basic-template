# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 18:11:02 2020

@author: Amrith
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#LOADING THE DATA
from sklearn.datasets import load_digits
digits = load_digits()

#FLATTENING THE IMAGE FEATURES 
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y= digits.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.25, random_state=0)

#FITTING THE MODEL. FEEL FREE TO CHOOSE YOUR MODEL
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=5,metric='minkowski', p=2)
model.fit(X_train,y_train)

#PREDICTION
pred=model.predict(X_test)

#EVALUATION
from sklearn import metrics
print("Classification report for classifier %s:\n%s\n"
      % (model, metrics.classification_report(y_test, pred)))
disp = metrics.plot_confusion_matrix(model, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)

