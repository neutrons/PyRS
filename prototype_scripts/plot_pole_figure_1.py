import numpy as np

import mpl_toolkits.axisartist.floating_axes as floating_axes
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist.grid_finder import FixedLocator, MaxNLocator, \
             DictFormatter
import matplotlib.pyplot as plt

tr = PolarAxes.PolarTransform()

degree_ticks = lambda d: (d*np.pi/180, "%d$^\\circ$"%(360-d))
degree0 = 0
angle_ticks = map(degree_ticks, np.linspace(degree0, 360, 5))
grid_locator1 = FixedLocator([v for v, s in angle_ticks])
tick_formatter1 = DictFormatter(dict(angle_ticks))
tick_formatter2 = DictFormatter(dict(zip(np.linspace(1000, 6000, 6),
                                         map(str, np.linspace(0, 5000, 6)))))

grid_locator2 = MaxNLocator(5)

gh = floating_axes.GridHelperCurveLinear(tr,
                                         extremes=(2*np.pi, np.pi, 1000, 6000),
                                         grid_locator1=grid_locator1,
                                         grid_locator2=grid_locator2,
                                         tick_formatter1=tick_formatter1,
                                         tick_formatter2=tick_formatter2)

fig = plt.figure()
ax = floating_axes.FloatingSubplot(fig, 111, grid_helper=gh)
fig.add_subplot(ax)

degree0 = 0.
azimuths = np.radians(np.linspace(degree0, 360, 90)) # added 180 degrees
zeniths = np.arange(1050, 6050, 50) # added 1000

r, theta = np.meshgrid(zeniths, azimuths)
print type(theta)
print theta
values = 90.0+5.0*np.random.random((len(azimuths), len(zeniths)))

aux_ax = ax.get_aux_axes(tr)
aux_ax.patch = ax.patch
ax.patch.zorder = 0.9

aux_ax.contourf(theta, r, values) # use aux_ax instead of ax

plt.show()
