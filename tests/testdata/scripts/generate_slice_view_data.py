# Generate a list of 3D data for testing slice view
import numpy as np
import random

num_row = 5
num_col = 4
num_layer = 10

X0 = 2
Xf = 8
Y0 = 0
Yf = 10
Z0 = 0
Zf = 30

dx = (Xf - X0)/float(num_col)
dy = (Yf - Y0)/float(num_row)
dz = (Zf - Z0)/float(num_layer)

print (dx, dy, dz)

data_set = list()
for k in range(num_layer):
    z_k = Z0 + dz * float(k)
    for i in range(num_row):
        y_i = Y0 + dy * float(i)
        for j in range(num_col):
            x_j = X0 + dx * float(j)

            coord_i = np.array([x_j, y_i, z_k])
            data_set.append(coord_i)
# END-FOR

data_set = np.array(data_set)
print data_set
print data_set.shape

def get_height(x, y, z):
    """

    :param x:
    :param y:
    :param z:
    :return:
    """
    h = np.sin(x/float(Xf-X0) * 2 * np.pi / 180) + 1 / (y + 1)**2. - 2.* np.exp(-(z-15)**2/15)

    return h

def get_width(x, y, z):
    """

    :param x:
    :param y:
    :param z:
    :return:
    """
    vec_w = np.ndarray(shape=x.shape, dtype='float')
    for iw in range(len(vec_w)):
        vec_w[iw] = 3.2 * (1 + random.random())

    return vec_w

vec_height = get_height(data_set[:, 0], data_set[:, 1], data_set[:, 2])
print (vec_height, '\n', vec_height.shape)

vec_width = get_width(data_set[:, 0], data_set[:, 1], data_set[:, 2])
print vec_width


wbuf = '# {0:12}{1:12}{2:12}{3:12}{4:12}\n'.format('X', 'Y', 'Z', 'Height', 'Width')
for index in range(len(data_set)):
    wbuf += '{0:-12}{1:-12}{2:-12}  {3:12}{4:12}\n'.format(data_set[index, 0], data_set[index, 1], data_set[index, 2],
                                                      '{0:.5f}'.format(vec_height[index]),
                                                      '{0:.5f}'.format(vec_width[index]))

print wbuf

# write to file
ofile = open('slice_test.dat', 'w')
ofile.write(wbuf)
ofile.close()
