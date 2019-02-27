import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.datasets import make_circles, make_moons, load_breast_cancer
from sklearn.model_selection import train_test_split
import warnings
from exposing import Exposer, EE
from sklearn import base
from sklearn.neighbors import KNeighborsClassifier
import timeit
from sklearn import tree, svm, naive_bayes


def dataset():
    n_samples = 1000
    ds = make_moons(noise=0.3, random_state=0, n_samples=n_samples)
    X, y = ds
    return train_test_split(X, y, test_size=0.4, random_state=42)


def breast_dataset():
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.4, random_state=42)


def test_given_subspace():
    X_train, X_test, y_train, y_test = dataset()
    estimator = Exposer(given_subspace=(0, 1))
    estimator.fit(X_train, y_train)
    score = estimator.score(X_test, y_test)
    print("\nExposer %.3f" % score)


def test_locations_and_reverse():
    X_train, X_test, y_train, y_test = dataset()
    estimator = Exposer(given_subspace=(0, 1))
    estimator.fit(X_train, y_train)

    x = X_train[:2]
    locations = estimator.locations(x)
    inverse_locations = estimator.inverse_locations(locations)


def test_rgb():
    X_train, X_test, y_train, y_test = dataset()
    estimator = Exposer()
    estimator.fit(X_train, y_train)
    rgb = estimator.rgb()


def test_ee():
    X_train, X_test, y_train, y_test = breast_dataset()
    for fuser in ("equal", "theta"):
        estimator = EE(fuser=fuser)
        estimator.fit(X_train, y_train)
        score = estimator.score(X_test, y_test)
        print("\nEE(%s) %.3f" % (fuser, score))


def test_approaches():
    X_train, X_test, y_train, y_test = breast_dataset()
    for approach in ("random", "brute", "purified"):
        estimator = EE(approach=approach)
        estimator.fit(X_train, y_train)
        score = estimator.score(X_test, y_test)
