import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

from exposing import Exposer

def test_demo():
    n_samples = 1000
    ds = make_circles(noise=0.2, factor=0.5, random_state=1, n_samples = n_samples)
    X, y = ds
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    estimator = Exposer()
    estimator.fit(X_train, y_train)
    score = estimator.score(X_test, y_test)
    print score
    #assert False
