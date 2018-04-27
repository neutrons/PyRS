#!/usr/bin/python
from pyrs.core import scandataio
import os

# default testing directory is ..../PyRS/
print (os.getcwd())

# therefore it is not too hard to locate testing data
test_data = 'tests/testdata/BD_Data_Log.hdf5'
print (os.path.exists(test_data))
