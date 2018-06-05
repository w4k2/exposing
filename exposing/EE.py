"""Exposing Classification"""
# Author: Pawel Ksieniewicz <pawel.ksieniewicz@pwr.edu.pl>

import itertools
import numpy as np
from builtins import range
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import _check_partial_fit_first_call
from sklearn.utils.random import sample_without_replacement as swr
from .Exposer import Exposer

APPROACHES = ('brute', 'random', 'purified')
FUSERS = ('equal', 'theta')

class EE(BaseEstimator, ClassifierMixin):
    def __init__(self, grain=16, a_steps=5, n_base=16, n_seek=32,
                 approach='random', fuser='theta', random_state=0, focus = 2):
        self.grain = grain
        self.a_steps = a_steps
        self.n_base = n_base
        self.n_seek = n_seek
        self.approach = approach
        self.fuser = fuser
        self.focus = focus
        self.random_state = random_state

    def partial_fit(self, X, y, classes=None):
        if _check_partial_fit_first_call(self, classes):
            self.fit(X, y)
        else:
            for e in self.ensemble_:
                e.partial_fit(X, y)


    def fit(self, X, y):
        X, y = check_X_y(X, y)
        random_state = check_random_state(self.random_state)
        self.X_ = X
        self.y_ = y
        self.n_features_ = X.shape[1]

        # Store the classes seen during fit
        self.classes_, _ = np.unique(y, return_inverse=True)

        # Establish set of subspaces
        self.subspaces_ = None
        if self.approach == 'brute' or self.approach == 'purified':
            self.subspaces_ = list(itertools.combinations(range(self.n_features_), 2))
        elif self.approach == 'random':
            self.subspaces_ = [swr(self.n_features_, 2,
                                   random_state=random_state)
                               for i in range(self.n_base)]

        # Compose ensemble
        self.ensemble_ = [Exposer(grain=self.grain,
                                  a_steps=self.a_steps,
                                  focus = self.focus,
                                  given_subspace=subspace)
                          for subspace in self.subspaces_]

        # Fit ensemble
        [clf.fit(X, y) for clf in self.ensemble_]

        # Cut purified ensemble
        if self.approach == 'purified':
            self.ensemble_.sort(key=lambda x: x.theta_, reverse=True)
            self.ensemble_ = self.ensemble_[:self.n_base]

        # Gather thetas
        self.thetas_ = np.array([clf.theta_ for clf in self.ensemble_])

        # Return the classifier
        return self

    def _prepare_generator(self):
        psize = self.grain * self.grain
        established = []
        pairs = []
        units = []
        for e in self.ensemble_:
            subspace = e.given_subspace
            count = 0
            known = None
            unknown = None
            for feature in subspace:
                if feature not in established:
                    count += 1
                    unknown = feature
                    established.append(feature)
                else:
                    known = feature
            if count == 0:
                pass
            elif count == 1:
                units.append((known, unknown, e))
            else:
                pairs.append((subspace[0], subspace[1], e))
            if len(established) == self.n_features_:
                break
        if len(established) == self.n_features_:
            return pairs, units
        else:
            return None

    def make_classification(self, n_samples = 1000, p = None):
        # Get generation scheme
        pairs, units = self._prepare_generator()

        # Prepare labels
        y = np.random.choice(self.classes_, n_samples, p = p)

        # Prepare storage
        X = np.zeros((n_samples, self.n_features_))
        X_ = np.zeros((n_samples, self.n_features_),dtype=np.int16)
        noise_scale = np.zeros(self.n_features_)

        # Generate both
        for setup in pairs:
            a, b, e = setup
            sublocation, subsample = e.get_samples(y)
            X_[:,[a,b]] = sublocation
            X[:,[a,b]] = subsample
            noise_scale[[a,b]] = e.scaler_.data_range_

        # Generate unit
        for setup in units:
            a, b, e = setup
            partial_location = (X_[:,a], e.given_subspace[1] == a)
            sublocation, subsample = e.get_samples(y, partial_location)
            if e.given_subspace[1] == a:
                X_[:,[b,a]] = sublocation
                X[:,[b,a]] = subsample
                noise_scale[[b,a]] = e.scaler_.data_range_
            else:
                X_[:,[a,b]] = sublocation
                X[:,[a,b]] = subsample
                noise_scale[[a,b]] = e.scaler_.data_range_

        noise_scale /= self.grain
        noise = np.random.rand(X.shape[0], X.shape[1]) * noise_scale - (noise_scale/2)
        X += noise

        return X, y

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_', 'ensemble_'])

        # Input validation
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError('number of features does not match')

        # Establish signatures
        signatures = [clf.signatures(X) for clf in self.ensemble_]

        # Acquire supports
        supports = None
        if self.fuser == 'equal':
            supports = np.sum(signatures, axis=0)
        elif self.fuser == 'theta':
            weighted_signatures = signatures * self.thetas_[:,
                                                            np.newaxis,
                                                            np.newaxis]
            supports = np.sum(weighted_signatures, axis=0)

        # Predict
        prediction = np.argmax(supports, axis=1)

        return self.classes_[prediction]
