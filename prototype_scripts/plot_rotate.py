import numpy
from matplotlib import pyplot, transforms

data = numpy.random.randn(100)

# first of all, the base transformation of the data points is needed
base = pyplot.gca().transData
rot = transforms.Affine2D().rotate_deg(90)

# define transformed line
line = pyplot.plot(data, 'r--', transform= rot + base)
# or alternatively, use:
# line.set_transform(rot + base)

pyplot.show()
