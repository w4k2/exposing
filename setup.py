from __future__ import print_function
import sys, os
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]


# get __version__ from _version.py
ver_file = os.path.join('exposing', '_version.py')
with open(ver_file) as f:
    exec(f.read())

try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)

try:
    import scipy
except ImportError:
    print('scipy is required during installation')
    sys.exit(1)

setup(name='exposing',
      version=__version__,
      description='Exposing classification algorithms compatible with scikit-learn API.',
      author='Pawel Ksieniewicz',
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      author_email='pawel.ksieniewicz@pwr.edu.pl',
      )
