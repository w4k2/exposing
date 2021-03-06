"""
=================================================
Comparing classifiers on three synthetic datasets
=================================================

An example plot of :class:`skltemplate.template.TemplateClassifier`
"""
import numpy as np
import datetime
from numpy.testing import assert_almost_equal
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification

from exposing import Exposer
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

n_samples = 1000
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1, n_samples = n_samples)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=0, n_samples = n_samples),
            make_circles(noise=0.2, factor=0.5, random_state=1, n_samples = n_samples),
            linearly_separable
            ]

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "AdaBoost",
         "Naive Bayes", "QDA", "Exposer"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    Exposer()]

for ds_idx, ds in enumerate(datasets):
    scores = {}
    times = {}

    # preprocess dataset, split into training and test part
    X, y = ds
    #X = StandardScaler().fit_transform(X)
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
        times.update({name: ((med-start).microseconds, (end-med).microseconds)})

    results = sorted(scores, key=scores.get, reverse=True)
    for idx, name in enumerate(results):
        print '%i - %.1f%% @ %s - %s' % (idx+1, 100 * scores[name], times[name], name)
