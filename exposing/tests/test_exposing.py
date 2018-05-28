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
    return train_test_split(X, y, test_size=.4, random_state=42)


def breast_dataset():
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=.4, random_state=42)


def test_given_subspace():
    X_train, X_test, y_train, y_test = dataset()
    estimator = Exposer(given_subspace=(0, 1))
    estimator.fit(X_train, y_train)
    score = estimator.score(X_test, y_test)

def test_locations_and_reverse():
    X_train, X_test, y_train, y_test = dataset()
    estimator = Exposer(given_subspace=(0, 1))
    estimator.fit(X_train, y_train)

    x = X_train[:2]
    locations = estimator.locations(x)
    inverse_locations = estimator.inverse_locations(locations)

    # assert(np.max(x - inverse_locations) < .1)

def test_rgb():
    X_train, X_test, y_train, y_test = dataset()
    estimator = Exposer()
    estimator.fit(X_train, y_train)
    rgb = estimator.rgb()


def test_ece():
    X_train, X_test, y_train, y_test = breast_dataset()
    for fuser in ('equal', 'theta'):
        estimator = EE(fuser=fuser)
        estimator.fit(X_train, y_train)
        score = estimator.score(X_test, y_test)

def test_approaches():
    X_train, X_test, y_train, y_test = breast_dataset()
    print(X_train.shape, X_test.shape)
    for approach in ('random', 'brute', 'purified'):
        start = timeit.default_timer()
        print(approach)
        estimator = EE(approach=approach)
        estimator.fit(X_train, y_train)
        score = estimator.score(X_test, y_test)
        print(len(estimator.ensemble_))
        print(score)
        stop = timeit.default_timer()
        print(stop - start)

def test_generation():
    X_train, X_test, y_train, y_test = breast_dataset()
    print(X_train.shape, X_test.shape)
    start = timeit.default_timer()

    estimator = EE(approach='random', n_base=50)
    estimator.fit(X_train, y_train)
    a = estimator.make_classification(100)

    stop = timeit.default_timer()
    print(stop - start)

def test_regeneration_power():
    X_train, X_test, y_train, y_test = breast_dataset()

    print(np.sum(y_test==0))
    print(np.sum(y_test==1))


    ee = EE(approach='random', n_base=50)
    ee.fit(X_train, y_train)
    X_reg, y_reg = ee.make_classification(10)

    ee_score = ee.score(X_test, y_test)
    print("EE score = %.3f" % ee_score)

    clfs = {
        'DT': tree.DecisionTreeClassifier,
        'kNN': KNeighborsClassifier,
        'SVC': svm.SVC,
        'NB': naive_bayes.GaussianNB,
        'EE': EE
    }
    for name in clfs:
        print("\n- %s" % name)

        clf_full = clfs[name]()
        clf_reg = clfs[name]()

        clf_full.fit(X_train, y_train)
        clf_reg.fit(X_reg, y_reg)

        score_f = clf_full.score(X_test, y_test)
        score_r = clf_reg.score(X_test, y_test)

        print("Full: %.3f" % score_f)
        print("Reg.: %.3f" % score_r)

def test_generation_preparing():
    X_train, X_test, y_train, y_test = breast_dataset()
    print(X_train.shape, X_test.shape)
    start = timeit.default_timer()

    estimator = EE(approach='random', n_base=50)
    estimator.fit(X_train, y_train)
    print(estimator._prepare_generator())
