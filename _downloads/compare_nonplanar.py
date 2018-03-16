"""
============================================
Comparing classifiers on three real datasets
============================================

An example plot of :class:`skltemplate.template.TemplateClassifier`
"""
import numpy as np
import datetime
from numpy.testing import assert_almost_equal
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.datasets import load_breast_cancer, load_iris

from exposing import EE
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

datasets = [load_breast_cancer(return_X_y=True),
            load_iris(return_X_y=True)
            ]

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "AdaBoost",
         "Naive Bayes", "QDA", "EEequal", "EEtheta"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    EE(fuser='equal'),
    EE(fuser='theta')]

for ds_idx, ds in enumerate(datasets):
    scores = {}
    times = {}

    # preprocess dataset, split into training and test part
    X, y = ds
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    print 'Dataset %i' % ds_idx
    print X.shape

    for name, clf in zip(names, classifiers):
        start = datetime.datetime.now()
        clf.fit(X_train, y_train)
        med = datetime.datetime.now()
        score = clf.score(X_test, y_test)
        end = datetime.datetime.now()
        scores.update({name: score})
        times.update({
            name: ((med-start).microseconds,
                   (end-med).microseconds)
            })

    results = sorted(scores, key=scores.get, reverse=True)
    for idx, name in enumerate(results):
        print '%i - %.1f%% @ %s - %s' % (
            idx+1, 100 * scores[name], times[name], name
        )
