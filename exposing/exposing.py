"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from medpy.filter.smoothing import anisotropic_diffusion


class Exposer(BaseEstimator, ClassifierMixin):
    """ An example classifier which implements a 1-NN algorithm.

    Parameters
    ----------
    demo_param : str, optional
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    X_ : array, shape = [n_samples, n_features]
        The input passed during :meth:`fit`
    y_ : array, shape = [n_samples]
        The labels passed during :meth:`fit`
    """

    def __init__(self, given_subspace=None, grain=16, a_steps=5):
        self.given_subspace = given_subspace
        self.grain = grain
        self.a_steps = a_steps

    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.

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
                self.subspace_ = np.array((0,0))
            elif self.n_features_ == 2:
                self.subspace_ = np.array((0,1))
            else:
                self.subspace_ = np.array((-1,-2))
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
        exposing_X = self.scaler_.transform(subspaced_X)
        exposing_y = self.le_.transform(y)

        # Empty model
        self.model_ = np.zeros((
            self.grain, self.grain,
            len(self.classes_))).astype('float_')

        # Exposing
        X_locations = np.clip(
            np.rint(exposing_X * self.grain).astype('int64'), 0, self.grain - 1)
        unique, counts = np.unique(np.array(
            [X_locations[:, 0], X_locations[:, 1], exposing_y]).T, return_counts=True, axis=0)
        self.model_[unique[:, 0], unique[:, 1], unique[:, 2]] += counts

        # Blurring and normalization
        for layer in xrange(len(self.classes_)):
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

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.

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

        # Input conversion
        subspaced_X = X[:, self.subspace_]
        exposing_X = self.scaler_.transform(subspaced_X)
        locations = np.clip(
            np.rint(exposing_X * self.grain).astype('int64'), 0, self.grain - 1)

        supports = self.model_[locations[:, 0], locations[:, 1], :]
        prediction = np.argmax(supports, axis=1)
        decoded_prediction = self.le_.inverse_transform(prediction)

        return decoded_prediction
