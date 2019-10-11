#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 16:40:26 2018

@author: hcf
"""

from skimage import io
from PIL import Image
import numpy as np

ImageData = Image.open('./LaB6_10kev_0deg-00000.tif')
#im = img_as_uint(np.array(ImageData))
io.use_plugin('freeimage')
Data = np.array(ImageData, dtype=np.int16)
Data[np.where((Data < 0) == True)] = 0
Data[np.where((Data > 65536) == True)] = 0
io.imsave('LaB6_10kev_0deg-00000_Rotated.tif', np.array(Data.T, dtype=np.uint16))
