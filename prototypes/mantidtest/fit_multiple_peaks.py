# The following line helps with future compatibility with Python 3
# print must now be used as a function, e.g print('Hello','World')
from __future__ import (absolute_import, division, print_function, unicode_literals)

# import mantid algorithms, numpy and matplotlib
import h5py
from mantid.simpleapi import *

import matplotlib.pyplot as plt

import numpy as np

import os

root_path = os.path.join(os.path.expanduser('~'), 'Projects/PyRS')
h5_name = os.path.join(root_path, 'tests/testdata/HZB_Raw_Project.hdf')
print(h5_name)
print(os.path.exists(h5_name))

hydra_h5 = h5py.File(h5_name, 'r')
