"""
This is exposing module, which contains classes of planar exposer and ensemble
of them
"""
from builtins import range
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from medpy.filter.smoothing import anisotropic_diffusion
import matplotlib.colors as colors


class Exposer(BaseEstimator, ClassifierMixin):
    """A classifier using basic, planar exposer.

    Parameters
    ----------
    given_subspace : str, optional
        A parameter used for demonstation of how to pass and store paramters.
    grain : str, optional
        A parameter used for demonstation of how to pass and store paramters.
    a_steps : str, optional
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    X_ : array, shape = [n_samples, n_features]
        The input passed during :meth:`fit`
    y_ : array, shape = [n_samples]
        The labels passed during :meth:`fit`
    model_ : array, shape = [grain, grain, n_classes]
        The labels passed during :meth:`fit`
    hue_ : array, shape = [grain, grain]
        The labels passed during :meth:`fit`
    value_ : array, shape = [grain, grain]
        The labels passed during :meth:`fit`
    saturation_ : array, shape = [grain, grain]
        The labels passed during :meth:`fit`
    """

    def __init__(self, given_subspace=None, grain=16, a_steps=5):
        self.given_subspace = given_subspace
        self.grain = grain
        self.a_steps = a_steps

    def rgb(self):
        """Returning a HSV to RGB visualization of fitted exposer.

        Returns
        -------
        y : array of int of shape = [n_samples]
            The label for each sample is the label of the closest sample seen
             during fit.
        """
        check_is_fitted(self, ['X_', 'y_', 'model_'])
        hsv = np.dstack((self._hue, self._saturation, self._value))
        rgb = colors.hsv_to_rgb(hsv)
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
        self.le_ = LabelEncoder()
        self.le_.fit(y)
        self.classes_ = unique_labels(y)

        # Scaler
        self.scaler_ = MinMaxScaler()
        self.scaler_.fit(subspaced_X)

        # Exposing versions of X and y
        exposing_y = self.le_.transform(y)

        # Empty model
        self.model_ = np.zeros((
            self.grain, self.grain,
            len(self.classes_))).astype('float_')

        # Exposing
        X_locations = self.locations(subspaced_X)
        unique, counts = np.unique(np.array(
            [X_locations[:, 0], X_locations[:, 1], exposing_y]).T,
                                   return_counts=True, axis=0)
        self.model_[unique[:, 0], unique[:, 1], unique[:, 2]] += counts

        # Blurring and normalization
        for layer in range(len(self.classes_)):
            plane = self.model_[:, :, layer]
            plane = anisotropic_diffusion(plane, niter=self.a_steps)
            plane /= np.sum(plane)
            self.model_[:, :, layer] = plane
        self.model_ /= np.max(self.model_)

        # HSV
        self._hue = np.argmax(self.model_, axis=2) / float(len(self.classes_))
        self._saturation = np.max(self.model_, axis=2) - \
            np.min(self.model_, axis=2)
        self._value = np.max(self.model_, axis=2)

        # Return the classifier
        return self

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
        decoded_prediction = self.le_.inverse_transform(prediction)

        return decoded_prediction