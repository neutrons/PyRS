# scipy_prototype.py
# reference: https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.interpolate.griddata.html#scipy.interpolate.griddata

----> 1 points3

NameError: name 'points3' is not defined

In [5]: points[0]
Out[5]: array([ 0.5727698 ,  0.04809841])

In [6]: clear


In [7]: def func(x, y):
   ...:     return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2
   ...: 

In [8]: values = func(points[:,0], points[:,1])

In [9]: from scipy.interpolate import griddata

In [10]: grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

In [11]: grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')

In [12]: def func3(x, y, z):
   ....:     return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2 + z
   ....: 

In [13]: points3 = np.random.rand(1000, 3)

In [14]: grid_x, grid_y, grid_z = np.mgrid[0:1:100j, 0:1:200j, 0:1:120j]

In [15]: grid_v0 = griddata(poi
points   points3  

In [15]: values3 = func3(points3[:, 0], point
points   points3  

In [15]: values3 = func3(points3[:, 0], points3[:, 1], points3[:, 2])

In [16]: grid_v0 = griddata(point3, values3, (grid_x, grid_y, grid_z), method='nearest')
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-16-0007364815a3> in <module>()
----> 1 grid_v0 = griddata(point3, values3, (grid_x, grid_y, grid_z), method='nearest')

NameError: name 'point3' is not defined

In [17]: grid_v0 = griddata(points3, values3, (grid_x, grid_y, grid_z), method='nearest')

