# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 12:41:07 2019

@author: Ebran
"""

from sklearn import svm
from sklearn import datasets

iris = datasets.load_iris()
X, y = iris.data, iris.target

classifier = svm.SVC(
    C=1.0,
    probability=True, 
    random_state=1)
classifier.fit(X, y)

print('Training Accuracy: ', classifier.score(X, y))

print('Prediction results: ', classifier.predict_proba([[5.2,  3.5,  2.4,  1.2]]))