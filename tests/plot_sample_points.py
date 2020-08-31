from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os
from pyrs.dataobjects.fields import StrainField

test_data_dir = '/home/jbq/repositories/pyrs/pyrs1/tests/data'


def plot_sample_points(*files):

    strain22 = StrainField(filename=os.path.join(test_data_dir, files[1]),
                           peak_tag='peak0')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for file in files:
        strain = StrainField(filename=os.path.join(test_data_dir, file), peak_tag='peak0')
        ax.scatter(strain.x, strain.y, strain.z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.legend(files)
    plt.show()

plot_sample_points('HB2B_1327.h5', 'HB2B_1328.h5', 'HB2B_1331.h5', 'HB2B_1332.h5')
