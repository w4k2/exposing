from sklearn.utils.estimator_checks import check_estimator
from exposing import (Exposer)

def test_classifier():
    return check_estimator(Exposer)
