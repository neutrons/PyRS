import h5py
import pylab as plt
import sys
import numpy as np
import matplotlib.image as mping
from matplotlib import pyplot as plt

import glob

PLOT = False

for name in glob.glob('*_Rotated.tif'):
    print ('Working on {}'.format(name))

    # read data
    Data = mping.imread(name)
    x_size = Data.shape[0]/2
    y_size = Data.shape[1]/2
    Data=Data.reshape(Data.shape[0]/2,2, Data.shape[1]/2, 2).sum(1).sum(-1)
    print (Data.shape)
    print (np.max(Data))
    Data[0, 0] = 60000
    Data[0, 1] = 50000
    Data[0, 2] = 40000
    if PLOT:
        plt.imshow(Data, cmap='nipy_spectral')
        plt.show()

    # insert
    Data = Data.reshape((x_size * y_size, ))
    print (Data.shape)
    Data = np.insert(Data, 0, y_size)
    print (Data.shape)
    Data = np.insert(Data, 0, x_size)
    print (Data.shape)

    # write
    file_name = name.replace('tif','bin')
    Data.astype('uint32').tofile(file_name)
    

    # ImageDataSet = np.array( Data )
    # ImageDataSet = ImageDataSet.reshape(Data.shape[0]*Data.shape[1])

    # print (type(Data))
    # print (Data.shape)
    # print (ImageDataSet.shape)
    # array = np.concatenate((np.arange(len(ImageDataSet)), ImageDataSet)).reshape(2,len(ImageDataSet)).T
    # print (array.shape)
    # print (array[10000])
    # print (ImageDataSet[10000])
    # # I need pure counts: np.save(name.replace('tif','.dat'), array)
    # np.save(name.replace('tif','dat'), ImageDataSet)
