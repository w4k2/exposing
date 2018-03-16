"""
====================================
Showing Exposition of planar Exposer
====================================

An example plot of :class:`skltemplate.template.TemplateClassifier`
"""
import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import exposing
reload(exposing)

n_samples = 10000
ds = make_moons(noise=0.3, random_state=0, n_samples=n_samples)
X, y = ds
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.4, random_state=42)

estimator = exposing.Exposer(a_steps=5, grain=32)
estimator.fit(X_train, y_train)

plt.imshow(estimator.rgb())

plt.show()
