import numpy
from matplotlib import pyplot, transforms

data = numpy.random.randn(100)

vec_x = numpy.arange(0, -10, -0.1)
vec_y = numpy.sin(vec_x*-1)  # * numpy.pi / 180.)

print data
print vec_x
print vec_y

# first of all, the base transformation of the data points is needed
base = pyplot.gca().transData
print(type(base))
rot = transforms.Affine2D().rotate_deg(270)
print(type(rot))

# define transformed line
# line = pyplot.plot(data, 'r--', transform= rot + base)
# line = pyplot.plot(vec_x, vec_y)
line = pyplot.plot(vec_x, vec_y, 'r--', transform=rot + base)
# or alternatively, use:
# line.set_transform(rot + base)

pyplot.show()
