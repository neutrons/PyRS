import numpy as np
from uncertainties import unumpy

from pyrs.dataobjects.sample_logs import PointList

# two points in real space separated by less than this amount (in mili meters) are considered the same point
from .constants import DEFAULT_POINT_RESOLUTION
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
        all_lengths = [len(values), len(errors), len(x), len(y), len(z)]
        assert len(set(all_lengths)) == 1, 'input lists must all have the same lengths'
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
    def point_list(self):
        return self._point_list

    @property
    def coordinates(self):
        return self._point_list.coordinates

    @property
    def x(self):
        return self._point_list.vx

    @property
    def y(self):
        return self._point_list.vy

    @property
    def z(self):
        return self._point_list.vz

    def extract(self, target_indexes):
        r"""
        Create a scalar field sample with a subset of the sampled points.

        Parameters
        ----------
        target_indexes: list

        Returns
        -------

        """

        subset = {}
        for member in ('values', 'errors', 'x', 'y', 'z'):
            member_value = getattr(self, member)
            subset[member] = [member_value[i] for i in target_indexes]
        return ScalarFieldSample(self.name, subset['values'], subset['errors'],
                                 subset['x'], subset['y'], subset['z'])

    def aggregate(self, other):
        r"""
        Bring in another scalar field sample. Overlaps can occur if a sample point is present in both samples.

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
        Find the scalar field sample common to two scalar field samples.

        Two samples are common if their corresponding sample points are apart from each other a distance
        smaller than the resolution distance.

        Parameters
        ----------
        other: ~pyrs.dataobjects.fields.ScalarFieldSample
        resolution: float
            Two points are considered the same if they are separated by a distance smaller than this quantity

        Returns
        -------
        ~pyrs.dataobjects.fields.ScalarFieldSample
        """
        # find the intersection of the aggregated point list
        target_indexes = self.point_list.intersection_aggregated_indexes(other.point_list, resolution=resolution)
        # extract those samples corresponding to the intersection of the aggregated point list
        return self.aggregate(other).extract(target_indexes)

    def fuse(self, other, resolution=DEFAULT_POINT_RESOLUTION, criterion='min_error'):
        r"""
        Bring in another scalar field sample and resolve the overlaps according to a selection criteria.

        Two samples are common if their corresponding sample points are apart from each other a distance
        smaller than the resolution distance.

        Parameters
        ----------
        other: ~pyrs.dataobjects.fields.ScalarFieldSample
        resolution: float
            Two points are considered the same if they are separated by a distance smaller than this quantity
        criterion: str
            Criterion by which to resolve which out of two (or more) samples is selected, while the rest is
            discarded. Possible values are:
            'min_error': the sample with the minimal uncertainty is selected
        Returns
        -------
        ~pyrs.dataobjects.fields.ScalarFieldSample
        """
        aggregate_sample = self.aggregate(other)  # combine both scalar field samples

        def min_error(indexes):
            r"""Find index of sample point with minimum error of the scalar field"""
            error_min_index = np.argmin(aggregate_sample.errors[indexes])
            return indexes[error_min_index]
        criterion_functions = {'min_error': min_error}
        assert criterion in criterion_functions, f'The criterion must be one of {criterion_functions.keys()}'

        # cluster sample points by their mutual distance. Points within a cluster are closer than `resolution`
        clusters = aggregate_sample.point_list.cluster(resolution=resolution)

        # find the indexes of the aggregated point list that we want to keep, and discard the rest
        target_indexes = list()
        # Iterate over the cluster, selecting only one index (one sample point) from each cluster
        # the clusters are sorted by size
        for cluster_index, point_indexes in enumerate(clusters):
            if len(point_indexes) == 1:
                break  # remaining clusters have all only one index, thus the selection process is irrelevant
            target_indexes.append(criterion_functions[criterion](point_indexes))
        # append point indexes from all clusters having only one member
        target_indexes.extend([point_indexes[0] for point_indexes in clusters[cluster_index, :]])

        # create a ScalarFieldSample with the sample points corresponding to the target indexes
        return aggregate_sample.extract(sorted(target_indexes))
