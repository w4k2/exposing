"""
============================
Influence of hyperparameters
============================

An example plot of :class:`skltemplate.template.TemplateClassifier`
"""
import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from exposing import Exposer

n_samples = 10000
ds = make_moons(noise=0.3, random_state=0, n_samples=n_samples)
X, y = ds
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.4, random_state=42)

grains = [2, 4, 8, 16, 32, 64]
a_stepss = [0, 1, 2, 3, 4, 5]

fig, ax = plt.subplots(len(grains), len(a_stepss), figsize=(10, 10))
for i, grain in enumerate(grains):
    for j, a_steps in enumerate(a_stepss):
        clf = Exposer(a_steps=a_steps, grain=grain)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        ax[i, j].imshow(clf.rgb())
        ax[i, j].set_axis_off()
        ax[i, j].set_title("acc = %.3f" % score)

plt.axis('off')
plt.show()
