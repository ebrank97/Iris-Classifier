# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 12:41:07 2019

@author: Ebran
"""

from sklearn import svm
from sklearn import datasets

iris = datasets.load_iris()
# Modeling Svm classifier using Iris Sepal and Petal features
X, y = iris.data, iris.target

def train_model(C, probability, random_state, kernel):
    # Creating the classifier
    if kernel == 'linear':
        classifier = svm.SVC(kernel='linear', C=C, probability=probability, random_state=random_state)
    else:
        classifier = svm.SVC(C=C, probability=probability, random_state=random_state)

    # Fitting the dataset to the model
    classifier.fit(X, y)

    return classifier

def predict_result(C=1.0, kernel='rbf'):
    model = train_model(C, True, 1, kernel)
    probabilities = model.predict_proba([[5.2,  3.5,  2.4,  1.2]])
    output = [{'name': 'Iris-Setosa', 'value': round(probabilities[0, 0] * 100, 2)}, {'name': 'Iris-Versicolour', 'value': round(probabilities[0, 1] * 100, 2)}, {'name': 'Iris-Virginica', 'value': round(probabilities[0, 2] * 100, 2)}]
    #print('Training Accuracy: ', model.score(X, y))
    #print('Prediction results: ', model.predict([[5.2,  3.5,  2.4,  1.2]]))
    #print('Prediction results: ', classifier.predict([['SepalLength(Cm)',  'SepalWidth(Cm)',  'PetalLength(Cm)',  'PetalWidth(Cm)']]))
    return round(model.score(X, y) * 100, 2), output

print(predict_result(1.0, 'linear'))






# Modeling Different Kernel Svm classifier using Iris Sepal features
'''
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

print('Prediction results LinearKernel: ', svc.predict([[5.2,  3.5]]))
print('Prediction results LinearSVC: ', lin_svc.predict([[5.2,  3.5]]))
print('Prediction results RBF: ', rbf_svc.predict([[5.2,  3.5]]))
print('Prediction results SVC: ', poly_svc.predict([[5.2,  3.5]]))
'''

# Modeling Different Kernel Svm classifier using Iris Petal features
'''
X = iris.data[:, 2:]
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

print('Prediction results LinearKernel: ', svc.predict([[2.4,  1.2]]))
print('Prediction results LinearSVC: ', lin_svc.predict([[2.4,  1.2]]))
print('Prediction results RBF: ', rbf_svc.predict([[2.4,  1.2]]))
print('Prediction results SVC: ', poly_svc.predict([[2.4,  1.2]]))
'''