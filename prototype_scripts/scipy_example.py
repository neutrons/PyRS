# Evaluate a simple example function on the points of a 3D grid:
from scipy.interpolate import RegularGridInterpolator
import numpy as np

# define an arbitrary function
def f(x,y,z):
    return 2 * x**3 + 3 * y**2 - z

# define a 3D grid
x = np.linspace(1, 4, 11)
y = np.linspace(4, 7, 22)
z = np.linspace(7, 9, 33)

# calculate data
data = f(*np.meshgrid(x, y, z, indexing='ij', sparse=True))
print (data.shape)

# data is now a 3D array with data[i,j,k] = f(x[i], y[j], z[k]). 

# Next, define an interpolating function from this data:
my_interpolating_function = RegularGridInterpolator((x, y, z), data)

# Evaluate the interpolating function at the two points (x,y,z) = (2.1, 6.2, 8.3) and (3.3, 5.2, 7.1):

pts = np.array([[2.1, 6.2, 8.3], [3.3, 5.2, 7.1]])
# which is indeed a close approximation to [f(2.1, 6.2, 8.3), f(3.3, 5.2, 7.1)].
r1 = my_interpolating_function(pts)
print (r1)
