from sklearn.utils.estimator_checks import check_estimator
from exposing import (Exposer, EE)


def test_exposer():
    return check_estimator(Exposer)


def test_EE():
    return check_estimator(EE)
