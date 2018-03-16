import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
import warnings

from exposing import Exposer, ECE


def dataset():
    n_samples = 1000
    ds = make_moons(noise=0.3, random_state=0, n_samples=n_samples)
    X, y = ds
    return train_test_split(X, y, test_size=.4, random_state=42)


def test_given_subspace():
    X_train, X_test, y_train, y_test = dataset()
    estimator = Exposer(given_subspace=(0, 1))
    estimator.fit(X_train, y_train)
    score = estimator.score(X_test, y_test)
    warnings.warn("Accuracy = %.3f" % score)


def test_rgb():
    X_train, X_test, y_train, y_test = dataset()
    estimator = Exposer()
    estimator.fit(X_train, y_train)
    rgb = estimator.rgb()


def test_ece():
    X_train, X_test, y_train, y_test = dataset()
    estimator = ECE()
    estimator.fit(X_train, y_train)