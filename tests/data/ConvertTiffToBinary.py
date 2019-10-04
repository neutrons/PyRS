#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 16:40:26 2018

@author: hcf
"""

from skimage import io, exposure, img_as_uint, img_as_float
from PIL import Image
import numpy as np
import pylab as plt


tiff_name = 'LaB6_10kev_0deg-00000.tif'
tiff_name = 'LaB6_10kev_35deg-00004.tif'
tiff_name = 'LaB6_10kev_35deg-00004_Rotated.tif'

ImageData = Image.open(tiff_name)
#im = img_as_uint(np.array(ImageData))
io.use_plugin('freeimage')
Data = np.array(ImageData, dtype=np.int16)
print(Data.shape, type(Data))
Data.astype(np.uint32)

# merge
DataR = Data[::2, ::2] + Data[::2, 1::2] + Data[1::2, ::2] + Data[1::2, 1::2]
print(DataR.shape, type(DataR))

DataR.tofile('test.bin')

plt.imshow(DataR, cmap='Greys',  interpolation='nearest')
plt.show()

# Data[np.where((Data<0) == True)] = 0
# Data[np.where((Data>65536) == True)] = 0
# io.imsave('LaB6_10kev_0deg-00000_Rotated.tif', np.array(Data.T, dtype=np.uint16))
