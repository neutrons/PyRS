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
    print z.min(), z.max(), z.mean()

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

# TESTME TODO ..... VZ: this is useless because color bar reflects the contour plot but cannot determine the contour plot!
# set color bar with a fixed range
if False:
    min_z = 0
    max_z = 0
    v = np.linspace(-.1, 2.0, 15, endpoint=True)  # (min_z, max_z, num_levels)
    x = plt.colorbar(ticks=v)
    print x
else:
    v = np.linspace(-.1, 2.0, 15, endpoint=True)  # (min_z, max_z, num_levels)
# END-TODO


# contour
level = 14
levels = [-1.5, -1, -0.5, 0, 0.5, 1]   # Level cannot solve the problem at all!
# FIXME - try this: https://stackoverflow.com/questions/21952100/setting-the-limits-on-a-colorbar-in-matplotlib !
ax1.contour(xi, yi, zi, levels, linewidths=0.5, colors='k')
cntr1 = ax1.contourf(xi, yi, zi, 14, cmap="RdBu_r")
# color bar
fig.colorbar(cntr1, ax=ax1)  #, ticks=v)
# scatterings
ax1.plot(x, y, 'ko', ms=3)
ax1.axis((-200, 200, 0, 80))
# ax1.set_title('grid and contour (%d points, %d grid points)' %
#               (npts, ngridx * ngridy))


# ----------
# Tricontour
# ----------
# Directly supply the unordered, irregularly spaced coordinates
# to tricontour.

level2 = 128  # 14 standard
ax2.tricontour(x, y, z, level2, linewidths=0.5, colors='k')
cntr2 = ax2.tricontourf(x, y, z, 14, cmap="RdBu_r")

fig.colorbar(cntr2, ax=ax2)
ax2.plot(x, y, 'ko', ms=3)
# ax2.axis((-2, 2, -2, 2))
ax2.axis((-200, 200, 0, 80))
# ax2.set_title('tricontour (%d points)' % npts)

plt.subplots_adjust(hspace=0.5)
plt.show()
