import numpy as np
import matplotlib.pyplot as plt

#-- Generate Data -----------------------------------------
# Using linspace so that the endpoint of 360 is included...
azimuths = np.radians(np.linspace(0, 360, 20))
zeniths = np.arange(0, 70, 10)

r, theta = np.meshgrid(zeniths, azimuths)
values = np.random.random((azimuths.size, zeniths.size))

#-- Plot... ------------------------------------------------
fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
ax.contourf(theta, r, values)

plt.show()
