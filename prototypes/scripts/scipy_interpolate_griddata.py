import scipy
import scipy.interpolate
import numpy as np

from scipy.interpolate import griddata

# Test for the 2D system
system2d = np.ndarray(shape=(4, 2), dtype='float')
system2d[0, 0] = 1.0
system2d[0, 1] = 1.0
system2d[1, 0] = 2.0
system2d[1, 1] = 2.0
system2d[2, 0] = 1.0
system2d[2, 1] = 2.0
system2d[3, 0] = 2.0
system2d[3, 1] = 1.0

value_vec = np.array([5., 6., 7., 8])

grid_x, grid_y = 1.51, 2.01
v_int = griddata(system2d, value_vec, (grid_x, grid_y), method='nearest')
print v_int


# Test for the 3D system
system3d = np.ndarray(shape=(4, 3), dtype='float')
system3d[0, 0] = 1.0
system3d[0, 1] = 1.0
system3d[0, 2] = 0.0

system3d[1, 0] = 2.0
system3d[1, 1] = 2.0
system3d[1, 2] = 0.0

system3d[2, 0] = 1.0
system3d[2, 1] = 2.0
system3d[2, 2] = 0.0

system3d[3, 0] = 2.0
system3d[3, 1] = 1.0
system3d[3, 2] = 0.0

grid_x, grid_y, grid_z = 1.3000, 2.20000, 0.00000
v_int = griddata(system3d, value_vec, (grid_x, grid_y, grid_z), method='nearest')
print v_int
