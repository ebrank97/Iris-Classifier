# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 12:41:07 2019

@author: Ebran
"""

from sklearn import svm
from sklearn import datasets

iris = datasets.load_iris()


#Modeling Different Kernel Svm classifier using Iris Sepal features
X = iris.data[:, :2]
y = iris.target
C = 1.0

# SVC with linear kernel
svc = svm.SVC(kernel='linear', C=C).fit(X, y)
# LinearSVC (linear kernel)
lin_svc = svm.LinearSVC(C=C).fit(X, y)
# SVC with RBF kernel
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
# SVC with polynomial (degree 3) kernel
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)

print('Training Accuracy LinearKernel: ', svc.score(X, y))
print('Training Accuracy LinearSVC: ', lin_svc.score(X, y))
print('Training Accuracy RBF: ', rbf_svc.score(X, y))
print('Training Accuracy PolynomialSVC: ', poly_svc.score(X, y))

print('Output LinearKernel: ', svc.predict([[5.2,  3.5]]))
print('Output LinearSVC: ', lin_svc.predict([[5.2,  3.5]]))
print('Output RBF: ', rbf_svc.predict([[5.2,  3.5]]))
print('Output SVC: ', poly_svc.predict([[5.2,  3.5]]))

#Modeling Svm classifier using Iris Sepal and Petal features
'''
X, y = iris.data, iris.target

classifier = svm.SVC(C=1.0,probability=True, random_state=1)
classifier.fit(X, y)

print('Training Accuracy: ', classifier.score(X, y))
print('Prediction results: ', classifier.predict([[5.2,  3.5,  2.4,  1.2]]))
'''