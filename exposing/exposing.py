"""Exposing Classification"""
# Author: Pawel Ksieniewicz <pawel.ksieniewicz@pwr.edu.pl>

from builtins import range
import numpy as np
np.set_printoptions(suppress=True,precision=2)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.random import sample_without_replacement as swr
from sklearn.preprocessing import MinMaxScaler
from medpy.filter.smoothing import anisotropic_diffusion
import matplotlib.colors as colors
import itertools

APPROACHES = ('brute', 'random', 'mst')
FUSERS = ('equal', 'theta')

class EE(BaseEstimator, ClassifierMixin):
    def __init__(self, grain=16, a_steps=5, n_base=15, n_seek=30,
                 approach='random', fuser='theta', random_state=0):
        self.grain = grain
        self.a_steps = a_steps
        self.n_base = n_base
        self.n_seek = n_seek
        self.approach = approach
        self.fuser = fuser
        self.random_state = random_state

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
        # print("Method is %s" % self.approach)
        if self.approach == 'brute':
            self.subspaces_ = list(itertools.combinations(range(self.n_features_), 2))
            pass
        elif self.approach == 'random':
            self.subspaces_ = [swr(self.n_features_, 2,
                                   random_state=random_state)
                               for i in range(self.n_base)]
        elif self.approach == 'mst':
            pass
        # print("Choosen subspaces: %s" % self.subspaces_)

        # Compose ensemble
        self.ensemble_ = [Exposer(grain=self.grain,
                                  a_steps=self.a_steps,
                                  given_subspace=subspace)
                          for subspace in self.subspaces_]

        # Fit ensemble
        [clf.fit(X, y) for clf in self.ensemble_]

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


class Exposer(BaseEstimator, ClassifierMixin):
    """A using basic, planar exposer usable for classification of datasets limited to two features.

    Notes
    -----
    Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod
    tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim
    veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea
    commodo consequat. Duis aute irure dolor in reprehenderit in voluptate
    velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat
    cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id
    est laborum.

    Parameters
    ----------
    given_subspace : tuple, optional, shape = [2]
        Indices of dataset subspace used to calculate exposer. (the default is
        usage of last two features).
    grain : int, optional
        Number of bins dividing every dimension (the default is ``16``).
    a_steps : int, optional
        Number of steps of anisotropic diffusion (the default is ``5``).

    Attributes
    ----------
    X_ : array, shape = [n_samples, n_features]
        The input passed during :meth:`fit`
    y_ : array, shape = [n_samples]
        The labels passed during :meth:`fit`
    theta_ : float
        Mean saturation of informative pixels of exposer.

    References
    ----------
    .. [1] Ksieniewicz, P., Grana, M., Wozniak, M. (2017). Paired feature
       multilayer ensemble - concept and evaluation of a classifier. Journal of
       Intelligent and Fuzzy Systems,  32(2), 1427-1436.
    """

    def __init__(self, given_subspace=None, grain=16, a_steps=5):
        self.given_subspace = given_subspace
        self.grain = grain
        self.a_steps = a_steps

    def model(self):
        """Returning a model of fitted exposer.

        Returns
        -------
        array_like : array of float of shape = [grain, grain, n_classes]
            A complete, fitted model of exposer.
        """
        check_is_fitted(self, ['X_', 'y_', 'model_'])
        return self.model_

    def hsv(self):
        """Returning a HSV representation of fitted exposer.

        Returns
        -------
        array_like : array of float of shape = [grain, grain, 3]
            The label for each sample is the label of the closest sample seen
             during fit.
        """
        check_is_fitted(self, ['X_', 'y_', 'model_'])
        return self._hsv

    def rgb(self):
        """Returning a HSV to RGB visualization of fitted exposer.

        Returns
        -------
        array_like : array of float of shape = [grain, grain, 3]
            The label for each sample is the label of the closest sample seen
             during fit.
        """
        check_is_fitted(self, ['X_', 'y_', 'model_'])
        rgb = colors.hsv_to_rgb(self._hsv)
        return rgb

    def fit(self, X, y):
        """A fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        self.X_ = X
        self.y_ = y
        self.n_features_ = X.shape[1]

        # Last two in subspace when none provided
        if self.given_subspace is None:
            if self.n_features_ == 1:
                self.subspace_ = np.array((0, 0))
            elif self.n_features_ == 2:
                self.subspace_ = np.array((0, 1))
            else:
                self.subspace_ = np.array((-1, -2))
        else:
            self.subspace_ = np.array(self.given_subspace)

        # Acquire subspaced X
        subspaced_X = X[:, self.subspace_].astype('float64')

        # Store the classes seen during fit
        self.classes_, y = np.unique(y, return_inverse=True)

        # Scaler
        self.scaler_ = MinMaxScaler()
        self.scaler_.fit(subspaced_X)

        # Empty model
        self.model_ = np.zeros((
            self.grain, self.grain,
            len(self.classes_))).astype('float_')

        # Exposing
        X_locations = self.locations(subspaced_X)
        unique, counts = np.unique(np.array(
            [X_locations[:, 0], X_locations[:, 1], y]).T,
                                   return_counts=True, axis=0)
        self.model_[unique[:, 0], unique[:, 1], unique[:, 2]] += counts

        # Blurring and normalization
        for layer in range(len(self.classes_)):
            plane = self.model_[:, :, layer]
            plane = anisotropic_diffusion(plane, niter=self.a_steps)
            plane /= np.sum(plane)
            self.model_[:, :, layer] = plane
        self.model_ /= np.max(self.model_)

        # Calculate measures

        # HSV
        self._hue = np.argmax(self.model_, axis=2) / float(len(self.classes_))
        self._saturation = np.max(self.model_, axis=2) - \
            np.min(self.model_, axis=2)
        self._value = np.max(self.model_, axis=2)
        self._hsv = np.dstack((self._hue, self._saturation, self._value))

        self._calculate_measures()

        # Prepare linear model
        self.linear_model_ = self.model_.reshape((-1,
                                                  len(self.classes_)))
        self.linear_model_ = np.divide(self.linear_model_,
                                       np.sum(self.linear_model_,
                                              axis = 0))

        # Return the classifier
        return self

    def _calculate_measures(self):
        treshold = .7
        self.theta_ = np.mean(self._saturation[self._value > treshold])

    def locations(self, subsamples):
        """Returning indices of exposer corresponding to given subsamples.

        Parameters
        ----------
        subsamples : array-like, shape = [n_samples, n_features]
            The training input samples.

        Returns
        -------
        locations : array-like
            Indices for given samples
        """
        transformed_subsamples = self.scaler_.transform(subsamples)
        locations = np.clip(
            np.rint(transformed_subsamples * self.grain).astype('int64'),
            0, self.grain - 1)
        return locations

    def inverse_locations(self, locations):
        """Returning indices of exposer corresponding to given subsamples.

        Parameters
        ----------
        location : array-like, shape = [n_samples, n_features]
            The training input samples.

        Returns
        -------
        subsamples : array-like
            Indices for given samples
        """
        grained_locations = np.array(locations) / self.grain
        subsamples = self.scaler_.inverse_transform(grained_locations)
        return subsamples

    def get_samples(self, y, partial_location = None):
        # MUSIMY KONIECZNIE DODAC SZUM DO PROBEK NA BAZIE PRZESKA
        # LOWANEGO ZIARNA. INACZEJ WSZYSTKIE WYGENEROWANE PROBKI
        # ROZLOZA SIE NA SIATCE ZIARNA EKSPOZERA.

        # Prepare storage for results
        X_ = np.zeros((len(y), 2), dtype=np.int16)

        if partial_location == None:
            # Prepare location space
            psize = self.grain * self.grain
            a = np.linspace(0,psize-1, psize, dtype=np.int16)

            # Iterate through classes
            for label in self.classes_:
                # Establish mask on given labels
                label_mask = y == label

                # Get points from distribution
                _ = np.random.choice(a, np.sum(y == label),
                                     p=self.linear_model_[:,label])
                X_[label_mask] = np.stack((_ // self.grain,
                                           _ % self.grain)).T

        else:
            locs, is_column = partial_location

            # Prepare location space
            a = np.linspace(0,self.grain-1, self.grain, dtype=np.int16)

            # Iterate through samples
            for i, label in enumerate(y):
                loc = locs[i]
                if is_column:
                    p = self.model_[:,loc,label]
                else:
                    p = self.model_[loc,:,label]

                p = np.divide(p, np.sum(p))
                _ = np.random.choice(a, p=p)
                X_[i] = [_, loc] if is_column else [loc, _]

        # Calculate subsamples from locations in distribution
        X = self.inverse_locations(X_)

        return X_, X

    def signatures(self, X):
        """Returning signatures corresponding to given samples from exposed model

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.

        """
        subspaced_X = X[:, self.subspace_]
        locations = self.locations(subspaced_X)
        supports = self.model_[locations[:, 0], locations[:, 1], :]
        return supports

    def predict_proba(self, X):
        """A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of int of shape = [n_samples]
            The label for each sample is the label of the closest sample
            seen udring fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_', 'model_'])

        # Input validation
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError('number of features does not match')

        # Signatures to support vectors
        signatures = self.signatures(X)
        sums = np.sum(signatures, axis=1)
        sums[sums == 0] = 1
        signatures /= sums[:, np.newaxis]

        return signatures

    def predict(self, X):
        """A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of int of shape = [n_samples]
            The label for each sample is the label of the closest sample
            seen udring fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_', 'model_'])

        # Input validation
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError('number of features does not match')

        signatures = self.signatures(X)
        prediction = np.argmax(signatures, axis=1)

        return self.classes_[prediction]
