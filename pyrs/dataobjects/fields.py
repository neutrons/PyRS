import enum
import uncertainties
from uncertainties import unumpy

from pyrs.dataobjects.sample_logs import PointList

# two points in real space separated by less than this amount (in mili meters) are considered the same point
DEFAULT_POINT_RESOLUTION = 0.01
SCALAR_FIELD_NAMES = ('lattice', 'strain', 'stress')  # standard names for most used fields


class ScalarFieldSample:
    r"""
    Evaluation of a scalar field on a discrete set of points in real space

    Examples of fields are lattice constants for a particular family of planes, or the strain along a particular
    direction

    Parameters
    ----------
    name: str
        Name of the field. Standard field names are defined in SCALAR_FIELD_NAMES
    values: list
        List of real values corresponding to the evaluation of the scalar field at the sample points
    errors: list
         List of real values corresponding to the undeterminacies in the evaluation of the scalar field at the sample
         points
    x: list
        List of coordinates along some X-axis for the set of sample points.
    y: list
        List of coordinates along some Y-axis for the set of sample points.
    z: list
        List of coordinates along some Z-axis for the set of sample points.
    """

    def __init__(self, name, values, errors, x, y, z):
        self._sample = unumpy.uarray(values, errors)
        self._point_list = PointList([x, y, z])
        self._name = name

    @property
    def name(self):
        r"""The identifying name of the scalar field"""
        return self._name

    @property
    def values(self):
        return unumpy.nominal_values(self._sample).tolist()

    @property
    def errors(self):
        return unumpy.std_devs(self._sample).tolist()

    @property
    def coordinates(self):
        return self._point_list.coordinates

    @property
    def point_list(self):
        return self._point_list

    def aggregate(self, other):
        r"""
        Bring in another scalar field sample. Overlaps can occur if a point is present in both samples

        Parameters
        ----------
        other

        Returns
        -------

        """
        if self._name != other.name:
            raise TypeError('Aggregation not allowed for ScalarFieldSample objects of different names')
        points = self._point_list.aggregate(other.point_list)
        values = self.values + other.values
        errors = self.errors + other.errors
        return ScalarFieldSample(self.name, values, errors, points.vx, points.vy, points.vz)

    def intersection(self, other, resolution=DEFAULT_POINT_RESOLUTION):
        r"""
        Find the scalar field sample common to both scalar field samples.

        Two points are common if they are apart from each other a distance smaller than the resolution distance

        Parameters
        ----------
        other: ~pyrs.dataobjects.fields.ScalarFieldSample
        resolution: float

        Returns
        -------
        ~pyrs.dataobjects.fields.ScalarFieldSample
        """
        aggregated_points = self.aggregate(other)


    def fuse(self, other, resolution=DEFAULT_POINT_RESOLUTION, selector='min_error'):
        r"""
        Bring in another scalar field sample and resolve the overlaps according to a selection criteria
        """
        aggregate = self.aggregate(other)
