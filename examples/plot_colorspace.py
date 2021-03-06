"""
===============================================
Showing colorspace parameters of planar Exposer
===============================================

An example plot of :class:`skltemplate.template.TemplateClassifier`
"""
import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from exposing import Exposer

n_samples = 10000
ds = make_blobs(n_samples=n_samples, n_features=2, centers=2,
                random_state=0, cluster_std=3.0)
X, y = ds
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.4, random_state=42)

clf = Exposer()
clf.fit(X_train, y_train)

fig, axs = plt.subplots(2, 3, figsize=(9, 6))


model = clf.model()
axs[0, 0].imshow(model[:, :, 0], cmap='gray')
axs[0, 0].set_title("Negative class")
axs[0, 1].imshow(model[:, :, 1], cmap='gray')
axs[0, 1].set_title("Positive class")

rgb = clf.rgb()
axs[0, 2].imshow(rgb)
axs[0, 2].set_title('RGB visualization')

hsv = clf.hsv()
axs[1, 0].imshow(hsv[:, :, 0])
axs[1, 0].set_title('Hue')
axs[1, 1].imshow(hsv[:, :, 1], cmap='gray')
axs[1, 1].set_title('Saturation')
axs[1, 2].imshow(hsv[:, :, 2], cmap='gray')
axs[1, 2].set_title('Value')

for ax in axs:
    for axx in ax:
        axx.axis('off')
plt.show()
