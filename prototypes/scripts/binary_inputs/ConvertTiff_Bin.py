import h5py
import pylab as plt
import sys
import numpy as np
import matplotlib.image as mping
from matplotlib import pyplot as plt

import glob

PLOT = False


def convert_to_spice_binary(option='1k'):
    """
    convert to SPICE binary file
    :param option:
    :return:
    """
    for name in glob.glob('*_Rotated.tif'):
        # read data
        image_data = mping.imread(name)
        print ('Working on {} of shape {} with value in range ({}, {})'.format(name, image_data.shape,
                                                                               np.min(image_data), np.max(image_data)))

        # optionally convert the data 1K from 2K
        if option.count('1k'):
            x_size = image_data.shape[0]/2
            y_size = image_data.shape[1]/2
            image_data = image_data.reshape(image_data.shape[0]/2, 2, image_data.shape[1]/2, 2).sum(1).sum(-1)
        elif option.count('raw') or option.count('2k'):
            x_size = image_data.shape[0]
            y_size = image_data.shape[1]
        else:
            raise RuntimeError('Option {} is not supported.'.format(option))

        # mask
        if option.count('mask'):
            image_data[0, 0] = 60000
            image_data[0, 1] = 50000
            image_data[0, 2] = 40000

        if PLOT:
            plt.imshow(image_data, cmap='nipy_spectral')
            plt.show()

        # insert data size for
        image_data = image_data.reshape((x_size * y_size, ))
        image_data = np.insert(image_data, 0, y_size)
        image_data = np.insert(image_data, 0, x_size)

        # write to binary
        file_name = name.replace('.tif',  '_{}.bin'.format(option))
        image_data.astype('uint32').tofile(file_name)
    # END-FOR

    return


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print ('{} option\noption: (1) 1k  (2) raw  (3) mask_1k  (4) mask_raw'.format(sys.argv[0]))
    else:
        convert_to_spice_binary(sys.argv[1])