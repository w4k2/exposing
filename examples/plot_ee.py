"""
========================
Showing Exposition of EE
========================

An example plot of :class:`skltemplate.template.TemplateClassifier`
"""
import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer, load_iris

import exposing

ds = load_breast_cancer(return_X_y=True)
X, y = ds
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.4, random_state=42)

estimator = exposing.EE(a_steps=5, grain=32, random_state=1, n_base=25)
estimator.fit(X_train, y_train)

fig, ax = plt.subplots(5, 5, figsize=(10, 10))
for i in xrange(5):
    for j in xrange(5):
        index = i*j+j
        ax[i, j].imshow(estimator.ensemble_[index].rgb())
        ax[i, j].set_title("%i - %s (%.2f)" % (
            index,
            estimator.subspaces_[index],
            estimator.thetas_[index]))
        ax[i, j].set_axis_off()

plt.show()
