import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import h5py


def import_h5_array(file_name):
    """
    """
    h5file = h5py.File(file_name, 'r')
    for entry in h5file:
        print entry
        if entry.startswith('Slice'):
            data_entry = h5file[entry]
            data_set = data_entry.value
            x = data_set[:, 0]
            y = data_set[:, 1]
            z = data_set[:, 2]

    h5file.close()

    return x, y, z

if True:
    x, y, z = import_h5_array('/tmp/pyrs_test_ss/test.hdf5')
    print x.min(), x.max()
    print y.min(), y.max()
else:
    np.random.seed(19680801)
    npts = 200
    x = np.random.uniform(-2, 2, npts)
    y = np.random.uniform(-2, 2, npts)
    z = x * np.exp(-x**2 - y**2)

fig, (ax1, ax2) = plt.subplots(nrows=2)

# -----------------------
# Interpolation on a grid
# -----------------------
# A contour plot of irregularly spaced data coordinates
# via interpolation on a grid.

# Create grid values first.
ngridx = 1000
ngridy = 2000
xi = np.linspace(-200, 200, ngridx)
yi = np.linspace(0, 70, ngridy)

# Perform linear interpolation of the data (x,y)
# on a grid defined by (xi,yi)
triang = tri.Triangulation(x, y)
interpolator = tri.LinearTriInterpolator(triang, z)
Xi, Yi = np.meshgrid(xi, yi)
zi = interpolator(Xi, Yi)

# Note that scipy.interpolate provides means to interpolate data on a grid
# as well. The following would be an alternative to the four lines above:
#from scipy.interpolate import griddata
#zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='linear')


ax1.contour(xi, yi, zi, 14, linewidths=0.5, colors='k')
cntr1 = ax1.contourf(xi, yi, zi, 14, cmap="RdBu_r")

fig.colorbar(cntr1, ax=ax1)
ax1.plot(x, y, 'ko', ms=3)
ax1.axis((-200, 200, 0, 80))
# ax1.set_title('grid and contour (%d points, %d grid points)' %
#               (npts, ngridx * ngridy))


# ----------
# Tricontour
# ----------
# Directly supply the unordered, irregularly spaced coordinates
# to tricontour.

ax2.tricontour(x, y, z, 14, linewidths=0.5, colors='k')
cntr2 = ax2.tricontourf(x, y, z, 14, cmap="RdBu_r")

fig.colorbar(cntr2, ax=ax2)
ax2.plot(x, y, 'ko', ms=3)
# ax2.axis((-2, 2, -2, 2))
ax2.axis((-200, 200, 0, 80))
# ax2.set_title('tricontour (%d points)' % npts)

plt.subplots_adjust(hspace=0.5)
plt.show()
