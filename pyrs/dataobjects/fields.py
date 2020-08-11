r"""
Module providing objects to represent strains and stress components, as well as any scalar field
defined over a set of sample points.

Classes defined in this module:

class ScalarFieldObject

  Definition: Scalar field (values and errors) defined over a set of sample points (a triad of
  x, y, and z values).

  Functionality:
    - Scalar fields can be combined in two fundamental ways, "along the same direction"
      and "accross different directions".
    - Combining ("fusing") two (or more) fields along the same direction results in one scalar
      field. The sample points of each field are "fused" together, resulting in a single set
      of sample points. The resulting field is defined over this new set of sample points. Any
      point common to two input fields is "fused" into a single point in the combined
      field. Operator '+' can be used to fuse scalar fields.
    - Combining ("stacking") two (or more) fields across different directions results in as many
      output fields as input fields. The sample points of each field are "fused" together,
      resulting in a single set of sample points. All the output fields are defined over
      this new set of sample points. Operator '*' can be used to stack scalar fields

class StrainField

  Definition: a scalar field of strain values and errors defined over a set of sample points (a triad of
  x, y, and z values).

  The set of sample points may originate from one or more experimental runs. Each experimental run has
  associated one PeakCollection instance, so we'll say that a strain field is associated to one or
  more PeakCollection instances.

  Functionality:
  - Strain fields can be fused (combine strains along the same direction) or stacked (combine
    strains across directions)
  - Strain fields resulting from the fusion of two or more strains are associated to a list of
    PeakCollection instances, which can be retrieved from property StrainField.peak_collections
  - Strain fields store the set of sample points over which they are defined. If a strain is
    associated to only one PeakCollection, then the sample points are those of the
    PeakCollection. If a strain is associated to, say, two PeakCollection objects, then some of the
    sample points will originate in the first PeakCollection object, and the remaining points will
    originate in the second PeakCollection object. The StrainField object holds a cross-reference
    index table resolving the provenance of each sample point to one PeakCollection and one sample
    point within the PeakCollection.
  - StrainField objects don't store strain values and errors, rather they are calculated every time
    the are requested using the cross-reference index table and function PeakCollection.get_strain()

class StressField

  Definition: a container of three stress and three strains scalar fields, a pair for each of the
  three mutually perperdicular directions.

  StressField objects are generated using three StrainField objects, one for each mutually perperdicular
  direction. Usually, these strains are defined over slightly different sets of sample points, so
  it is necessary to stack them. After the stacking operation, the three output StrainField objects are
  defined over the same set of sample points and calculation of the stress components can proceed. The
  StressField object stores the three stacked StrainField objects, it does not store the three original
  StrainField objects.

  The three stress components are stored as ScalarFieldSample objects.

  Selected Functionality:
  - Stacked strains are accessible with properties StressField.strain11 (.strain22, .strain33)
  - Stress components can be accessed with the bracket operator (stress['11'], stress['22'], stress['33'])
  - Iterating over a StressField objects returns an iterator over the stress
    components (for component in stress: ...)
  - StressField objects hold a "currently accessible direction" which can be updated with the
    StressField.select() method.
  - Properties StressField.values and StressField.errors returns the values and errors of the stress
    component along the currectly accessble direction
"""
from collections import namedtuple
from enum import Enum
from enum import unique as unique_enum
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import sys
from typing import TYPE_CHECKING, Any, cast, Iterator, List, Optional, Tuple, Union
from uncertainties import unumpy
from mantid.simpleapi import mtd, CreateMDWorkspace, BinMD
from mantid.api import IMDHistoWorkspace

from pyrs.core.workspaces import HidraWorkspace
from pyrs.dataobjects.sample_logs import PointList, aggregate_point_lists
from pyrs.peaks import PeakCollection, PeakCollectionLite  # type: ignore
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore
from .constants import DEFAULT_POINT_RESOLUTION, NOT_MEASURED_NUMPY
if sys.version_info < (3, 8):
    from typing_extensions import final
else:
    from typing import final


# two points in real space separated by less than this amount (in mili meters) are considered the same point
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

    def __init__(self, name: str,
                 values: Union[List[float], np.ndarray], errors: Union[List[float], np.ndarray],
                 x: List[float], y: List[float], z: List[float]) -> None:
        all_lengths = [len(values), len(errors), len(x), len(y), len(z)]
        assert len(set(all_lengths)) == 1, 'input lists must all have the same lengths'
        self._sample = unumpy.uarray(values, errors)
        self._point_list = PointList([x, y, z])
        self._name = name

    def __len__(self) -> int:
        return len(self.values)

    def __add__(self, other_field: 'ScalarFieldSample') -> 'ScalarFieldSample':
        r"""
        Fuse the current strain with another strain using the default resolution distance and overlap criterion

        resolution = ~pyrs.dataobjects.constants.DEFAULT_POINT_RESOLUTION
        criterion = 'min_error'

        Parameters
        ----------
        other_strain:  ~pyrs.dataobjects.fields.ScalarFieldSample
            Right-hand side of operation addition

        Returns
        -------
        ~pyrs.dataobjects.fields.ScalarFieldSample
        """
        return self.fuse_with(other_field)

    def __mul__(self, other: 'ScalarFieldSample') -> List['ScalarFieldSample']:
        r"""
        Stack this field with another field, or with a list of field

        Stacking two fields
            field1_stacked, field2_stacked = field1 * field2
        Stacking three fields proceeds by first stacking the two leftmost fields:
           field1 * field2 * field3 --> [field1_stacked, field2_stacked] * field3
        which is why we need to implement '*' between a list and field object

        Parameters
        ----------
        other: ~pyrs.dataobjects.fields.ScalarFieldSample, list
            If a list, each item is a ~pyrs.dataobjects.fields.ScalarFieldSample object

        Returns
        -------
        list
            list of stacked ~pyrs.dataobjects.fields.ScalarFieldSample objects.
        """
        stack_kwargs = dict(resolution=DEFAULT_POINT_RESOLUTION, stack_mode='union')
        if isinstance(other, ScalarFieldSample):
            return stack_scalar_field_samples(self, other, **stack_kwargs)  # type: ignore
        elif isinstance(other, (list, tuple)):
            for field in other:
                if isinstance(field, ScalarFieldSample) is False:
                    raise TypeError(f'{field} is not a {str(self.__class__)} object')
            return stack_scalar_field_samples(self, *other, **stack_kwargs)

    def __rmul__(self, other: 'ScalarFieldSample') -> List['ScalarFieldSample']:
        r"""
        Stack a list of fields along with this field.

        Example: [field1, field2] * field3 --> field1_stacked, field3_stacked, field3_stacked

        Parameters
        ----------
        other: list
            Each item is a ~pyrs.dataobjects.fields.ScalarFieldSample object.

        Return
        ------
        list
            List of stacked strains. Each item is a ~pyrs.dataobjects.fields.ScalarFieldSample object.
        """
        stack_kwargs = dict(resolution=DEFAULT_POINT_RESOLUTION, stack_mode='complete')
        if isinstance(other, (list, tuple)):
            for strain in other:
                if isinstance(strain, ScalarFieldSample) is False:
                    raise TypeError(f'{strain} is not a {str(self.__class__)} object')
            return stack_scalar_field_samples(*other, self, **stack_kwargs)
        else:
            error_message = f'Unable to multiply objects of type {str(other.__class__)} and ScalarFieldSample'
            raise NotImplementedError(error_message)

    @property
    def name(self) -> str:
        r"""The identifying name of the scalar field"""
        return self._name

    @property
    def values(self) -> np.ndarray:
        return unumpy.nominal_values(self._sample)

    @property
    def errors(self) -> np.ndarray:
        return unumpy.std_devs(self._sample)

    @property
    def sample(self) -> np.ndarray:
        r"""
        Uncertainties arrays containing both values and errors.

        Returns
        -------
        ~unumpy.array
        """
        return self._sample

    @property
    def point_list(self) -> PointList:
        return self._point_list

    @property
    def coordinates(self) -> np.ndarray:
        return self._point_list.coordinates

    @property
    def x(self) -> List[float]:
        return self._point_list.vx

    @property
    def y(self) -> List[float]:
        return self._point_list.vy

    @property
    def z(self) -> List[float]:
        return self._point_list.vz

    @property
    def isfinite(self) -> 'ScalarFieldSample':
        r"""
        Filter out scalar field values with non-finite values, such as :math:`nan`.

        Returns
        -------
        ~pyrs.dataobjects.fields.ScalarFieldSample
        """
        indexes_finite = np.where(np.isfinite(self.values))[0]
        return self.extract(indexes_finite)

    def interpolated_sample(self, method: str = 'linear', fill_value: float = float('nan'), keep_nan: bool = True,
                            resolution: float = DEFAULT_POINT_RESOLUTION,
                            criterion: str = 'min_error') -> 'ScalarFieldSample':
        r"""
        Interpolate the scalar field sample of a regular grid given by the extents of the sample points.

        Interpolation for linear scans is done along the scan direction, and interpolation for surface scans
        is done on the 2D surface.

        :math:`nan` values are disregarded when doing the interpolation, but they can be incorporated into
        the interpolated values by finding the point in the regular grid lying closest to a sample point
        containing a :math:`nan` value.

        When the scalar field contains two sample points within `resolution` distance, only one can
        be chosen as data point for the interpolation.

        Parameters
        ----------
        method: str
            Method of interpolation. Allowed values are 'nearest' and 'linear'
        fill_value: float
            Value used to fill in for requested points outside the input points.
        keep_nan: bool
            Incorporate :math:`nan` found in the sample points into the interpolated field sample.
        resolution: float
            Two points are considered the same if they are separated by a distance smaller than this quantity.
        criterion: str
            Criterion by which to resolve which sample points out of two (or more) ends up being selected,
            while the rest of the sample points are discarded is discarded. Possible values are:
            'min_error': the sample with the minimal uncertainty is selected.

        Returns
        -------
        ~pyrs.dataobjects.fields.ScalarFieldSample
        """
        # Corner case: the points fill a regular grid. Thus, there's no need to interpolate.
        if self.point_list.is_a_grid(resolution=resolution):
            return self.coalesce(resolution=resolution, criterion=criterion)  # get rid of duplicate points
        field_finite = self.isfinite  # We need to remove non-finite values prior to interpolation
        field_finite = field_finite.coalesce(resolution=resolution, criterion=criterion)  # for neighbor sample points
        # Regular grid using the `extents` of the sample points but reduced in dimension for linear or surface scans
        # Reduction dimension is required for griddata() to work `method == 'linear'`.
        irreducible = False if method == 'nearest' else True
        grid_irreducible = self.point_list.mgrid(resolution=resolution, irreducible=irreducible)
        coordinates_irreducible = field_finite.point_list.coordinates_irreducible(resolution=resolution)
        # Corner case: if scalar field sample `self` has `nan` values at the extrema of its extents,
        # then removing these non-finite values will result in scalar field sample `field_finite`
        # having `extents` smaller than those of`self`. Later when interpolating, some of the grid
        # points (determined using `extents` of `self`) will fall outside the `extents` of `field_finite`
        # and their values will have to be filled with the `fill_value` being passed on to function `griddata`.
        values = griddata(coordinates_irreducible, field_finite.values, tuple(grid_irreducible),
                          method=method, fill_value=fill_value).ravel()
        coordinates = self.point_list.grid_point_list(resolution=resolution).coordinates  # coords spanned by the grid
        if keep_nan is True:
            # For each sample point of `self` that has a `nan` field value, find the closest grid point and assign
            # a `nan` value to this point
            point_indexes_with_nan = np.where(np.isnan(self.values))[0]  # points of `self` with field values of `nan`
            if len(point_indexes_with_nan) > 0:
                coordinates_with_nan = self.coordinates[point_indexes_with_nan]
                _, grid_indexes = cKDTree(coordinates).query(coordinates_with_nan, k=1)
                values[grid_indexes] = float('nan')
        errors = griddata(coordinates_irreducible, field_finite.errors, tuple(grid_irreducible),
                          method=method, fill_value=fill_value).ravel()
        return ScalarFieldSample(self.name, values, errors, *list(coordinates.T))

    def extract(self, target_indexes: List[int]) -> 'ScalarFieldSample':
        r"""
        Create a scalar field sample with a subset of the sampled points.

        Parameters
        ----------
        target_indexes: list
            List of sample point indexes to extract field values from.

        Returns
        -------
        ~pyrs.dataobjects.fields.ScalarFieldSample
        """

        subset = {}
        for member in ('values', 'errors', 'x', 'y', 'z'):
            member_value = getattr(self, member)
            subset[member] = [member_value[i] for i in target_indexes]
        return ScalarFieldSample(self.name, subset['values'], subset['errors'],
                                 subset['x'], subset['y'], subset['z'])

    def aggregate(self, other: 'ScalarFieldSample') -> 'ScalarFieldSample':
        r"""
        Bring in another scalar field sample. Overlaps can occur if a sample point is present in both samples.

        Parameters
        ----------
        other: ~pyrs.dataobjects.fields.ScalarFieldSample

        Returns
        -------
        ~pyrs.dataobjects.fields.ScalarFieldSample
        """
        if self._name != other.name:
            raise TypeError('Aggregation not allowed for ScalarFieldSample objects of different names')
        points = self._point_list.aggregate(other.point_list)
        values = np.concatenate((self.values, other.values))
        errors = np.concatenate((self.errors, other.errors))
        return ScalarFieldSample(self.name, values, errors, points.vx, points.vy, points.vz)

    def intersection(self, other: 'ScalarFieldSample',
                     resolution: float = DEFAULT_POINT_RESOLUTION) -> 'ScalarFieldSample':
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

    def coalesce(self, resolution: float = DEFAULT_POINT_RESOLUTION,
                 criterion: str = 'min_error') -> 'ScalarFieldSample':
        r"""
        Merge sampled points separated by less than certain distance into one.

        Parameters
        ----------
        resolution: float
            Two points are considered the same if they are separated by a distance smaller than this quantity
        criterion: str
            Criterion by which to resolve which sample points out of two (or more) ends up being selected,
            while the rest of the sample points are discarded is discarded. Possible values are:
            'min_error': the sample with the minimal uncertainty is selected.
        Returns
        -------
        ~pyrs.dataobjects.fields.ScalarFieldSample
        """
        def min_error(indexes: List[int]) -> List[int]:
            r"""Find index of sample point with minimum error of the scalar field"""
            error_values = np.array(self.errors)[indexes]
            error_min_index = np.nanargmin(error_values)  # ignore 'nan' values
            return indexes[error_min_index]
        criterion_functions = {'min_error': min_error}
        assert criterion in criterion_functions, f'The criterion must be one of {criterion_functions.keys()}'

        # cluster sample points by their mutual distance. Points within a cluster are closer than `resolution`
        clusters = self.point_list.cluster(resolution=resolution)

        # find the indexes of the aggregated point list that we want to keep, and discard the rest
        target_indexes = list()
        # Iterate over the cluster, selecting only one index (one sample point) from each cluster
        # the clusters are sorted by size
        for cluster_index, point_indexes in enumerate(clusters):
            if len(point_indexes) == 1:
                break  # remaining clusters have all only one index, thus the selection process is irrelevant
            target_indexes.append(criterion_functions[criterion](point_indexes))
        # append point indexes from all clusters having only one member
        if 1 + cluster_index < len(clusters):  # the previous for-loop did not exhaust all the clusters
            target_indexes.extend([point_indexes[0] for point_indexes in clusters[cluster_index:]])

        # create a ScalarFieldSample with the sample points corresponding to the target indexes
        return self.extract(sorted(target_indexes))  # type: ignore

    def fuse_with(self, other: 'ScalarFieldSample',
                  resolution: float = DEFAULT_POINT_RESOLUTION, criterion: str = 'min_error') -> 'ScalarFieldSample':
        r"""
        Bring in another scalar field sample and resolve the overlaps according to a selection criterum.

        Two samples are common if their corresponding sample points are apart from each other a distance
        smaller than the resolution distance.

        Parameters
        ----------
        other: ~pyrs.dataobjects.fields.ScalarFieldSample
        resolution: float
            Two points are considered the same if they are separated by a distance smaller than this quantity.
        criterion: str
            Criterion by which to resolve which sample points out of two (or more) ends up being selected,
            while the rest of the sample points are discarded is discarded. Possible values are:
            'min_error': the sample with the minimal uncertainty is selected.
        Returns
        -------
        ~pyrs.dataobjects.fields.ScalarFieldSample
        """
        aggregate_sample = self.aggregate(other)  # combine both scalar field samples
        return aggregate_sample.coalesce(resolution=resolution, criterion=criterion)

    def to_md_histo_workspace(self, name: str = '',
                              interpolate: bool = True,
                              method: str = 'linear', fill_value: float = float('nan'), keep_nan: bool = True,
                              resolution: float = DEFAULT_POINT_RESOLUTION, criterion: str = 'min_error'
                              ) -> IMDHistoWorkspace:
        r"""
        Save the scalar field into a MDHistoWorkspace. Interpolation of the sample points is carried out
        by default.

        Parameters `method`, `fill_value`, `keep_nan`, `resolution` , and `criterion` are  used only if
        `interpolate` is `True`.

        Saved units for the sample points are in milimeters

        Parameters
        ----------
        name: str
            Name of the output workspace.
        interpolate: bool
            Interpolate the scalar field sample of a regular 3D grid given by the extents of the sample points.
        method: str
            Method of interpolation. Allowed values are 'nearest' and 'linear'
        fill_value: float
            Value used to fill in for requested points outside the input points.
        keep_nan: bool
            Incorporate :math:`nan` found in the sample points into the interpolated field sample.
        resolution: float
            Two points are considered the same if they are separated by a distance smaller than this quantity.
        criterion: str
            Criterion by which to resolve which sample points out of two (or more) ends up being selected,
            while the rest of the sample points are discarded is discarded. Possible values are:
            'min_error': the sample with the minimal uncertainty is selected.
        Returns
        -------
        MDHistoWorkspace
        """

        # use ScalarFieldSample name is one isn't defined
        if not name:
            name = self.name

        if interpolate is True:
            sample = self.interpolated_sample(method=method, fill_value=fill_value, keep_nan=keep_nan,
                                              resolution=resolution, criterion=criterion)
        else:
            sample = self

        extents = sample.point_list.extents(resolution=resolution)  # triad of DirectionExtents objects
        for extent in extents:
            assert extent[0] <= extent[1], f'min value of {extent} is greater than max value'
        # Units of sample points in PointList are 'mm', which we keep when exporting
        extents_str = ','.join([extent.to_createmd(input_units='mm', output_units='mm') for extent in extents])

        # create an empty event workspace of the correct dimensions
        axis_labels = ('x', 'y', 'z')
        CreateMDWorkspace(OutputWorkspace='__tmp', Dimensions=3, Extents=extents_str,
                          Names=','.join(axis_labels), Units='mm,mm,mm', EnableLogging=False)
        # set the bins for the workspace correctly
        aligned_dimensions = list()
        for label, extent in zip(axis_labels, extents):  # type: ignore
            extent_str = extent.to_binmd(input_units='mm', output_units='mm')
            aligned_dimensions.append(f'{label},{extent_str}')
        aligned_kwargs = {f'AlignedDim{i}': aligned_dimensions[i] for i in range(len(aligned_dimensions))}
        BinMD(InputWorkspace='__tmp', OutputWorkspace=name, EnableLogging=False, **aligned_kwargs)

        # remove original workspace, so sliceviewer doesn't try to use it
        mtd.remove('__tmp')

        # get a handle to the workspace
        wksp = mtd[name]
        # set the signal and errors
        dims = [extent.number_of_bins for extent in extents]
        wksp.setSignalArray(sample.values.reshape(dims))
        wksp.setErrorSquaredArray(np.square(sample.errors.reshape(dims)))

        return wksp

    def to_csv(self, file: str) -> None:
        raise NotImplementedError('This functionality has yet to be implemented')

    def export(self, *args: Any, form: str = 'MDHistoWokspace', **kwargs: Any) -> Any:
        r"""
        Export the scalar field to a particular format. Each format has additional arguments

        Allowed formats, along with additional arguments and return object:
        - 'MDHistoWorkspace' calls function `to_md_histo_workspace`
            name: str, name of the workspace
            interpolate (`True`): bool, interpolate values to a regular coordinate grid
            method: ('linear'): str, method of interpolation. Allowed values are 'nearest' and 'linear'
            fill_value: (float('nan'): float, value used to fill in for requested points outside the input points.
            keep_nan (`True`): bool, transfer `nan` values to the interpolated sample
            Returns: MDHistoWorkspace, handle to the workspace
        - 'CSV' calls function `to_csv`
            file: str, name of the output file
            Returns: str, the file as a string

        Parameters
        ----------
        form: str
        """
        exporters = dict(MDHistoWorkspace=self.to_md_histo_workspace,
                         CSV=self.to_csv)
        exporters_arguments = dict(MDHistoWorkspace=('name',), CSV=('file',))
        # Call the exporter
        exporter_arguments = {arg: kwargs[arg] for arg in exporters_arguments[form]}
        return exporters[form](*args, **exporter_arguments)  # type: ignore


class _StrainField:

    POINT_MISSING_INDEX = -1  # indicates one single-scan index is not present in one of the overlapping clusters

    r"""Base class for common implementation details of StrainFields"""
    def __init__(self):
        pass  # this stores nothing

    def __len__(self):
        if self.point_list:
            return len(self.point_list)
        else:
            return 0

    def __eq__(self, other) -> bool:
        r"""
        Assert if two strains are the same by comparing their list of scan-strains

        Parameters
        ----------
        other_strain: ~pyrs.dataobjects.fields._StrainField

        Returns
        -------
        bool
        """
        try:
            return set([id(s) for s in self.strains]) == set([id(s) for s in other.strains])
        except AttributeError:
            return False

    @property
    def field(self):
        raise NotImplementedError()

    @property
    def values(self) -> np.ndarray:
        '''Default implementation which is likely slower than it needs to be'''
        return self.field.values

    @property
    def errors(self) -> np.ndarray:
        '''Default implementation which is likely slower than it needs to be'''
        return self.field.errors

    @property
    def sample(self) -> unumpy.uarray:
        '''Uncertainties arrays containing both values and errors.

        Default implementation which is likely slower than it needs to be'''
        return self.field.sample

    @property
    def point_list(self):
        raise NotImplementedError()

    @property
    def peak_collections(self):
        raise NotImplementedError()

    @property
    def strains(self) -> List['StrainFieldSingle']:
        raise NotImplementedError()

    @final
    @property
    def coordinates(self) -> np.ndarray:
        return self.point_list.coordinates

    @final
    @property
    def x(self):
        return self.point_list.vx

    @final
    @property
    def y(self):
        return self.point_list.vy

    @final
    @property
    def z(self):
        return self.point_list.vz

    @final
    def to_md_histo_workspace(self, name: str = '',
                              interpolate: bool = True,
                              method: str = 'linear', fill_value: float = float('nan'), keep_nan: bool = True,
                              resolution: float = DEFAULT_POINT_RESOLUTION,
                              criterion: str = 'min_error'
                              ) -> IMDHistoWorkspace:
        r"""
        Save the strain field into a MDHistoWorkspace. Interpolation of the sample points is carried out
        by default.

        Parameters `method`, `fill_value`, `keep_nan`, `resolution` , and `criterion` are  used only if
        `interpolate` is `True`.

        Saved units for the sample points are in milimeters

        Parameters
        ----------
        name: str
            Name of the output workspace.
        interpolate: bool
            Interpolate the scalar field sample of a regular 3D grid given by the extents of the sample points.
        method: str
            Method of interpolation. Allowed values are 'nearest' and 'linear'
        fill_value: float
            Value used to fill in for requested points outside the input points.
        keep_nan: bool
            Incorporate :math:`nan` found in the sample points into the interpolated field sample.
        resolution: float
            Two points are considered the same if they are separated by a distance smaller than this quantity.
        criterion: str
            Criterion by which to resolve which sample points out of two (or more) ends up being selected,
            while the rest of the sample points are discarded is discarded. Possible values are:
            'min_error': the sample with the minimal uncertainty is selected.
        Returns
        -------
        IMDHistoWorkspace
        """
        export_kwags = dict(interpolate=interpolate, method=method, fill_value=fill_value,
                            keep_nan=keep_nan, resolution=resolution, criterion=criterion)
        return self.field.to_md_histo_workspace(name, **export_kwags)  # type: ignore

    def _calculate_pointlist_map(self,
                                 point_lists: List['PointList'],
                                 resolution: float) -> Tuple[PointList, List[np.ndarray]]:
        r"""
        Calculate the aggregated point list, and the sample points for each collection associated to each of
        the points in the aggregated point list

        Caveat: each individual point list cannot contain any two points within `resolution` distance

        Parameters
        ----------
        point_lists: list
            list of ~pyrs.dataobjects.sample_logs.PointList instances
        resolution: float
            Two points are considered the same if they are separated by a distance smaller than this quantity

        Returns
        -------
        tuple
            A two-item tuple whose items are, in order:
            - ~pyrs.dataobjects.sample_logs.PointList
            -
        """
        # create a list of all points
        full_point_list = PointList([point_lists[0].vx, point_lists[0].vy, point_lists[0].vz])
        for points in point_lists[1:]:
            full_point_list = full_point_list.aggregate(points)  # this concatenates the lists

        # calculate the clustering of all points
        # variable `cluster_indices` is a list with as many elements as clusters. Each list element represents one
        # cluster, and is made up of a list of point-list indexes, specifying the sample points belonging to the
        # cluster.
        # TODO why call `sorted`? Seems irrelevant in this algorithm
        cluster_indices = sorted(full_point_list.cluster(resolution=resolution))

        # create a list of which aggregate index starts each contributing PointList
        point_list_starts = list(np.cumsum([len(point_list) for point_list in point_lists]))
        point_list_starts.insert(0, 0)  # pointlist start at index=0

        merged_positions = []  # average position of the sample points contained in each cluster
        # variable `full_clusters`:
        #     - is a list, with an item per cluster (or per point of the aggregated point list)
        #     - each item is a list of length `num_point_lists`
        #     - each value in the item list is an index of an individual input point list.
        # Example: I have two input point lists of length 3 and 2.
        # Individual point lists indexes are [0, 1, 2] and [0, 1] for the first and second point list.
        #     aggregate indexes then run from zero to four: [0, 1, 2, 3, 4]
        #     the correspondence between aggregate and individual indexes: [0, 1, 2, 3 ,4] --> [0, 1, 2, 0, 1]
        # cluster_indices = [[0, 3], [1], [2, 4]]  three clusters, first and last have contributions for all lists
        # full_clusters = [[0, 0], [1, -1], [2, 1]]  second cluster is missing a point from the second list
        full_clusters = []
        num_point_lists = len(point_lists)  # number of input PointList instances
        for i, cluster in enumerate(cluster_indices):
            # make sure the aggregate indices in list `cluster` are sorted, from lowest to highest
            cluster = sorted(cluster)  # variable `cluster` is a list of aggregate indices

            # new position is the average of all the contributing points
            x = np.average(full_point_list.vx[cluster])
            y = np.average(full_point_list.vy[cluster])
            z = np.average(full_point_list.vz[cluster])

            # variable `full_cluster` is a 1D array of length `num_point_lists`
            # a "full cluster" has an aggregate index for each PointList, otherwise some of the entries in
            # `full_cluster` will remain with a value of POINT_MISSING_INDEX
            full_cluster = np.full(num_point_lists, self.POINT_MISSING_INDEX, dtype=int)

            # convert each index of the aggregated point list into the index of the individual point list it's
            # associated with.
            # for every aggregate index of `cluster`, find where in `point_list_startst` should be inserted in
            # order to preserve the order of `point_list_starts`. This identifies which point list the aggregate
            # index belongs to.
            # Example: If I have three lists of lengths 3, 5, and 2, then `point_list_starts==[0, 3, 8]`. Then
            # np.searchsorted(point_list_starts, [2, 4, 9], side='right') == [1, 2, 3] indicating that
            # aggregated point 2 corresponds to the first point list, point 4 corresponds to the second list, and
            # point 9 corresponds to the third list
            pointlist_indexes = np.searchsorted(point_list_starts, cluster, side='right') - 1  # -1, number-->index

            for pointlist_index, aggregate_index in zip(pointlist_indexes, cluster):
                # TODO if we have two sample points within resolution in the same point list, then
                # TODO we will write to full_cluster[pointlist_index] twice, thus loosing one of the two points
                # TODO instrument scientist confirm this possibility flags a problem with the motors
                full_cluster[pointlist_index] = int(aggregate_index - point_list_starts[pointlist_index])

            merged_positions.append((x, y, z))
            full_clusters.append(full_cluster)

        # prepare to convert x,y,z triplets into a PointList
        x_array, y_array, z_array = np.asarray(merged_positions).transpose()

        return PointList([x_array, y_array, z_array]), full_clusters

    @final
    def fuse_with(self, other_strain: '_StrainField',
                  resolution: float = DEFAULT_POINT_RESOLUTION, criterion: str = 'min_error') -> '_StrainField':
        r"""
        Fuse the current strain scan with another scan taken along the same direction.

        Resolve any occurring overlaps between the scans according to a selection criterion.

        Parameters
        ----------
        other_strain:  ~pyrs.dataobjects.fields._StrainField
        resolution: float
            Two points are considered the same if they are separated by a distance smaller than this quantity
        criterion: str
            Criterion by which to resolve which sample points out of two (or more) ends up being selected,
            while the rest of the sample points are discarded is discarded. Possible values are:
            'min_error': the sample with the minimal uncertainty is selected.

        Returns
        -------
        ~pyrs.dataobjects.fields.StrainField
        """
        # Corner case: attempt to fuse with itself
        if self == other_strain:
            return self

        # collect the individual strains from the two (possibly) composite strains
        # remove possible duplicates by casting the ID of the strains as dictionary keys (order is preserved)
        single_scan_strains: List['StrainFieldSingle'] = []
        if isinstance(self, StrainFieldSingle):
            single_scan_strains.append(self)  # this is an individual strain
        else:
            # get individual strains
            single_scan_strains.extend(self.strains)

        if isinstance(single_scan_strains, StrainFieldSingle):
            if other_strain not in single_scan_strains:
                single_scan_strains.append(other_strain)  # this is an individual strain
        else:  # the other option is that it is a StrainField
            for test_strain in other_strain.strains:  # type: ignore
                if test_strain not in single_scan_strains:
                    single_scan_strains.append(test_strain)  # this is an individual strain

        multi_scan_strain: StrainField = StrainField()  # object to be returned
        multi_scan_strain._strains = single_scan_strains  # copy over everything

        point_lists = [strain.point_list for strain in single_scan_strains]
        # variable `map_points` is a list specifying the sample points
        multi_scan_strain._point_list, map_points = self._calculate_pointlist_map(point_lists, resolution)

        # Identify which sample points from the single scans are chosen to represent
        # each sample point of the multi scan
        single_scan_winner_indexes = []
        single_scan_pointlist_winner_indexes = []
        single_scan_strains_count = len(single_scan_strains)
        if criterion == 'min_error':
            # collect the criterion values from all StrainFieldSingle objects
            strain_errors = [strain.errors for strain in single_scan_strains]

            def get_cost(single_scan_strain_errors, index):
                return np.inf if index == self.POINT_MISSING_INDEX else single_scan_strain_errors[index]

            # loop over each sample point of the multi-scan strain
            for point_list_index in range(len(multi_scan_strain)):
                # in principle, each single scan contributes with a cost for this sample point
                costs = np.zeros(single_scan_strains_count)
                # loop over all single-scan strains, finding the associated cost
                for single_scan_index, single_scan_errors in enumerate(strain_errors):
                    # find the sample point index in the single scan, associated to the sample point
                    single_scan_point_list_index = map_points[point_list_index][single_scan_index]
                    costs[single_scan_index] = get_cost(single_scan_errors, single_scan_point_list_index)
                assert np.all(costs >= 0.0)  # costs are either infinite or positive
                single_scan_winner_index = np.argmin(costs)  # find the single scan with the smallest cost
                single_scan_winner_indexes.append(single_scan_winner_index)
                single_scan_pointlist_winner_indexes.append(map_points[point_list_index][single_scan_winner_index])

            # convert the winners to arrays
            scan_indexes = np.asarray(single_scan_winner_indexes, dtype=int)
            point_indexes = np.asarray(single_scan_pointlist_winner_indexes, dtype=int)
            multi_scan_strain._winners = multi_scan_strain.ChosenSamplePoints(scan_indexes, point_indexes)
        else:
            raise ValueError(f'Unallowed value of criterion="{criterion}"')

        return multi_scan_strain

    def __add__(self, other_strain: '_StrainField') -> '_StrainField':
        r"""
        Fuse the current strain with another strain using the default resolution distance and overlap criterion

        resolution = ~pyrs.dataobjects.constants.DEFAULT_POINT_RESOLUTION
        criterion = 'min_error'

        Parameters
        ----------
        other_strain:  ~pyrs.dataobjects.fields.StrainField
            Right-hand side of operation addition

        Returns
        -------
        ~pyrs.dataobjects.fields.StrainField
        """
        return self.fuse_with(other_strain)

    def __mul__(self, other: Union['_StrainField', List['_StrainField']]) -> Optional[List['StrainField']]:
        r"""
        Stack this strain with another strain, or with a list of strains

        Parameters
        ----------
        other: ~pyrs.dataobjects.fields.StrainField, list
            If a list, each item is a ~pyrs.dataobjects.fields.StrainField object

        Returns
        -------
        list
            list of stacked ~pyrs.dataobjects.fields.StrainField objects.
        """
        stack_kwargs = dict(resolution=DEFAULT_POINT_RESOLUTION, mode='union')
        if isinstance(other, _StrainField):
            return self.__class__.stack_with(self, other, **stack_kwargs)  # type: ignore
        elif isinstance(other, (list, tuple)):
            for strain in other:
                if isinstance(strain, _StrainField) is False:
                    raise TypeError(f'{strain} is not a {str(self.__class__)} object')  # type: ignore
            return self.__class__.stack_with(self, *other, **stack_kwargs)  # type: ignore
        else:
            raise NotImplementedError('Do not know how to multiply these objects')

    def stack_with(self, other_strain: '_StrainField',
                   mode: str = 'union',
                   resolution: float = DEFAULT_POINT_RESOLUTION) -> List['_StrainField']:
        r"""
        Combine the sample points of two strains scanned at different directions.

        The two strains can be single or multi-scan.

        Examples
        --------
        Consider combining two multi-scan strains s0 and s1. Strain s0 is made up of single-scan strains
        s00 and s01 that overlap in some region (the XX in the diagram below). Strain s1 results from the
        combination of three single-scan strains s10, s11, and s12. Strains s0 and s1 overlap considerably

          Stacking s0 over s1                               Stacking s1 over s0
        -----------------------                      --------------------------------
        |     s00    XX  s01  |                      | s10 |     s11    XXXXX  s12  |
        -----------------------                      --------------------------------
        |          s0         |                      |             s1               |
        ------------------------------------         ------------------------------------
            |             s1               |                      |          s0         |
            --------------------------------                      -----------------------
            | s10 |     s11    XXXXX  s12  |                      |     s00    XX  s01  |
            --------------------------------                      -----------------------

        Below we plot the stacked strains s0* and s1*. The value '-1' indicates sample points of s0* that
        are not points of s00 or s01 (similarly for s1*)

           The stacked s0 strain, s0*                      The stacked s` strain, s1*
        ------------------------------------         ------------------------------------
        |             s0*     -1,....... -1|         |             s1*               -1 |
        ------------------------------------         ------------------------------------

        Parameters
        ----------
        other_strain: ~pyrs.dataobjects.fields._StrainField
        mode: str
            A mode to stack the scalar fields. Valid values are 'union' and 'intersection'
        resolution: float
            Two points are considered the same if they are separated by a distance smaller than this quantity

        Raises
        ------
        NotImplementedError
            Stacking currently not supported for 'intersection' and 'common'

        Returns
        -------
        list
            A two-item list containing the two stacked strains
        """
        # validate the mode and convert to a single set of terms
        valid_stack_modes = ('union', 'intersection', 'complete', 'common')
        if mode not in valid_stack_modes:
            raise ValueError(f'{mode} is not a valid stack mode. Valid modes are {valid_stack_modes}')
        if mode in ('intersection', 'common'):
            raise NotImplementedError(f'mode "{mode}"is not currently supported')

        results = []
        if mode in ('union', 'complete'):
            # Stack `self` over `other_strain`. In the aggregated point list, first come the points from `self`,
            # then come the points from `other_strain`
            point_list_union = self.point_list.fuse_with(other_strain.point_list, resolution=resolution)
            if len(point_list_union) == len(self.point_list) == len(other_strain.point_list):
                results = [self, other_strain]
            else:
                strain1 = StrainField()
                strain1._point_list = point_list_union
                strain1._strains = self.strains
                # `strain1._winners` relates each of the points of the single scans of `self` to the points
                # of the aggregated list `point_list_union`. There are points in `point_list_union` with
                # no counterpart in the single scans (see example in the docstring), thus
                # len(strain1._winners) < len(point_list_union)
                if isinstance(self, StrainField):
                    strain1._winners = self._winners

                # Stack `other_strain` over `self`. In the new aggregated point list, first come the points
                # from `other_strain`, then come the points from `self`
                # `point_list_union` and `point_list_union_2` contain the same sample points,
                # BUT WITH DIFFERENT ORDER !!!!
                point_list_union_2 = other_strain.point_list.fuse_with(self.point_list, resolution=resolution)
                strain2 = StrainField()
                strain2._point_list = point_list_union_2
                strain2._strains = other_strain.strains
                if isinstance(other_strain, StrainField):
                    strain2._winners = other_strain._winners  # Notice len(strain2._winners) < len(point_list_union_2)

                # make sure both have the same length as that is the point of stacking
                assert len(point_list_union) == len(point_list_union_2)

                results = [strain1, strain2]
        else:
            raise RuntimeError('Should not be possible to get here')

        return results


class StrainFieldSingle(_StrainField):
    r"""
    This class holds strain information for a single ``PeakCollection``
    and its associated ``PointList``.

    Parameters
    ----------
    resolution: float
        Two points are considered the same if they are separated by a distance smaller than this quantity
    """
    def __init__(self,
                 filename: str = '',
                 projectfile: Optional[HidraProjectFile] = None,
                 peak_tag: str = '',
                 hidraworkspace: Optional[HidraWorkspace] = None,
                 peak_collection: Optional[PeakCollection] = None,
                 point_list: Optional[PointList] = None,
                 resolution: float = DEFAULT_POINT_RESOLUTION) -> None:
        r"""
        Converts a HidraWorkspace and PeakCollection into a ScalarField
        """
        super().__init__()
        self._peak_collection: Optional[PeakCollection] = None
        # this are made as top level property to follow interface of StrainField
        self._point_list: Optional[PointList] = None

        # Create a strain field from a single scan, if so requested
        single_scan_kwargs = dict(filename=filename, projectfile=projectfile, peak_tag=peak_tag,  # type: ignore
                                  hidraworkspace=hidraworkspace, peak_collection=peak_collection,  # type: ignore
                                  point_list=point_list)  # type: ignore
        if True in [bool(v) for v in single_scan_kwargs.values()]:  # at least one argument is not empty
            single_scan_kwargs['resolution'] = resolution
            self._initialize_with_single_scan(**single_scan_kwargs)  # type: ignore

    def _initialize_with_single_scan(self,
                                     filename: str = '',
                                     projectfile: Optional[HidraProjectFile] = None,
                                     peak_tag: str = '',
                                     hidraworkspace: Optional[HidraWorkspace] = None,
                                     peak_collection: Optional[PeakCollection] = None,
                                     point_list: Optional[PointList] = None,
                                     resolution: float = DEFAULT_POINT_RESOLUTION) -> None:
        r"""

        Parameters
        ----------
        filename
        projectfile
        peak_tag
        hidraworkspace
        peak_collection
        point_list
        resolution: float
            Two points are considered the same if they are separated by a distance smaller than this quantity

        Returns
        -------

        """
        # get the workspace and peaks by resolving the supplied inputs
        self._point_list, self._peak_collection = _to_pointlist_and_peaks(filename, peak_tag,
                                                                          projectfile, hidraworkspace,
                                                                          peak_collection,
                                                                          point_list,
                                                                          resolution=resolution)

    def __len__(self):
        return len(self._peak_collection)

    @property
    def filenames(self):
        if not self._peak_collection.projectfilename:
            return []
        else:
            return [self._peak_collection.projectfilename]

    @property
    def peak_collection(self):
        return self._peak_collection

    @property
    def peak_collections(self):
        return [self._peak_collection]

    @property
    def strains(self) -> List['StrainFieldSingle']:
        return [self]

    @property
    def point_list(self):
        return self._point_list

    @property
    def values(self):
        return self._peak_collection.get_strain()[0]

    @property
    def errors(self):
        return self._peak_collection.get_strain()[1]

    @property
    def field(self) -> ScalarFieldSample:
        if self._peak_collection is None:
            raise RuntimeError('PeakCollection has not been set')
        values, errors = self._peak_collection.get_strain()
        full_values = np.full(len(self.point_list), NOT_MEASURED_NUMPY, dtype=float)
        full_errors = np.full(len(self.point_list), NOT_MEASURED_NUMPY, dtype=float)
        full_values[:values.size] = values
        full_errors[:errors.size] = errors
        return ScalarFieldSample('strain', full_values, full_errors,
                                 self.x, self.y, self.z)


def _to_pointlist_and_peaks(filename: str,
                            peak_tag: str,
                            projectfile: Optional[HidraProjectFile],
                            hidraworkspace: Optional[HidraWorkspace],
                            peak_collection: Optional[PeakCollection],
                            point_list: Optional[PointList],
                            resolution: float = DEFAULT_POINT_RESOLUTION) -> Tuple[PointList, PeakCollection]:
    '''Take all of the various ways to supply the :py:obj:PointList and
    :py:obj:PeakCollection and convert them into those actual
    objects. For :py:obj:PeakCollection the first one found in the list
    * ``peak_collection``
    * ``projectfile``

    Similarly, for :py:obj:PoinList the first one found in the list
    * ``point_list``
    * ``hidraworkspace`` - taken from the :py:obj:SampleLogs
    * ``projectfile``

    Parameters
    ----------
        resolution: float
            Two points are considered the same if they are separated by a distance smaller than this quantity

    Raises
    ------
    RuntimeError
        The peak collection contains at least two points that overlap (closer than `resolution`)
    '''
    # load information from a file if it isn't already provided
    closeproject = False
    if filename and not (peak_collection or point_list):
        projectfile = HidraProjectFile(filename, HidraProjectFileMode.READONLY)
        closeproject = True
    elif TYPE_CHECKING:  # only True when running mypy
        projectfile = cast(HidraProjectFile, projectfile)

    # create objects from the project file
    if projectfile:
        # create HidraWorkspace
        if not hidraworkspace:
            hidraworkspace = HidraWorkspace()
            hidraworkspace.load_hidra_project(projectfile, load_raw_counts=False,
                                              load_reduced_diffraction=False)

        # get the PeakCollection
        if not peak_collection:
            peak_tags = projectfile.read_peak_tags()
            # verify peaks were savsed in the file
            if len(peak_tags) == 0:
                raise IOError('File "{}" does not have peaks defined'.format(filename))
            if peak_tag:  # verify the tag is in the file
                if peak_tag not in peak_tags:
                    raise ValueError('Failed to find tag "{}" in file with tags {}'.format(peak_tag, peak_tags))
            else:
                # use the only one if nothing is specified
                if len(peak_tags) == 1:
                    peak_tag = peak_tags[0]
                else:
                    raise RuntimeError('Need to specify peak tag: {}'.format(peak_tags))

            # load the peak_tag from the file
            peak_collection = projectfile.read_peak_parameters(peak_tag)

        # cleanup
        if closeproject:
            projectfile.close()
            del projectfile

        # verify the subruns are parallel
        if hidraworkspace and hidraworkspace.get_sub_runs() != peak_collection.sub_runs:  # type: ignore
            raise RuntimeError('Need to have matching subruns')
    elif TYPE_CHECKING:  # only True when running mypy
        hidraworkspace = cast(HidraWorkspace, hidraworkspace)
        peak_collection = cast(PeakCollection, peak_collection)

    # extract the PointList
    if hidraworkspace:
        if not point_list:
            point_list = hidraworkspace.get_pointlist()
    elif TYPE_CHECKING:  # only True when running mypy
        point_list = cast(PointList, point_list)

    # Check the point list doesn't have overlapping points
    if point_list.has_overlapping_points(resolution):
        raise RuntimeError('At least two sample points are overlapping')

    # verify that everything is set by now
    if (not point_list) or (not peak_collection):
        raise RuntimeError('Do not have both point_list and peak_collection defined')

    if len(point_list) < len(peak_collection):
        msg = 'point_list and peak_collection are not compatible length ({} < {})'.format(len(point_list),
                                                                                          len(peak_collection))
        raise ValueError(msg)

    return point_list, peak_collection


class StrainField(_StrainField):
    r"""
    This class holds the strain information for a composite set of
    ``StrainFieldSingle``. It holds what is needed to create the
    ``ScalarFieldSample`` and to update information about the peaks.
    """

    ChosenSamplePoints = namedtuple('ChosenSamplePoints', ['scan_indexes', 'point_indexes'])
    r"""
    Data structure relating sample points from single scans to sample points of a multi scan

    Examples
    chosen = ChosenSamplePoints([0, 0, 1], [3, 4, 0]) indicates that:
    - the multi scan is made up of three sample points
    - the first sample point in the multi scan is associated to sample point with
      index 3 of `StrainFieldSingle` object with index 0
    - the second sample point in the multi scan is associated to sample point with
      index 4 of `StrainFieldSingle` object with index 0
    - the third and last sample point in the multi scan is associated to sample point with
      index 0 of `StrainFieldSingle` object with index 1

    Parameters
    ----------
    scan_indexes: np.ndarray
        each index identifies one of the StrainFieldSingle instances making up the multi scan
    point_indexes: np.ndarray
        each index identifies one of the sample points in one of the single scans that make up
        the multi scan
    """

    @staticmethod
    def fuse_strains(*args: 'StrainField', resolution: float = DEFAULT_POINT_RESOLUTION,
                     criterion: str = 'min_error') -> '_StrainField':
        r"""
        Bring in together several strains measured along the same direction. Overlaps are resolved
        according to a selection criterion.

        Parameters
        ----------
        args list
            multiple ~pyrs.dataobjects.fields.StrainField objects.
        resolution: float
            Two points are considered the same if they are separated by a distance smaller than this quantity
        criterion: str
            Criterion by which to resolve which sample points out of two (or more) ends up being selected,
            while the rest of the sample points are discarded is discarded. Possible values are:
            'min_error': the sample with the minimal uncertainty is selected.

        Returns
        -------
        ~pyrs.dataobjects.fields.StrainField
        """
        # Validation checks
        assert len(args) > 1, 'More than one strain is needed'
        for strain in args:
            assert isinstance(strain, _StrainField), 'This input is not a StrainField object'
        # Iterative fusing
        strain, strain_other = args[0: 2]  # first two strains in the list
        strain_fused = strain.fuse_with(strain_other, resolution=resolution, criterion=criterion)
        for strain_other in args[2:]:  # fuse with remaining strains, one at a time
            strain_fused = strain_fused.fuse_with(strain_other, resolution=resolution, criterion=criterion)
        return strain_fused

    @staticmethod
    def stack_strains(*strains: 'StrainField',
                      stack_mode: str = 'complete',
                      resolution: float = DEFAULT_POINT_RESOLUTION) -> List['StrainField']:
        r"""
        Evaluate a list of strain fields taken at different directions on a list of commont points.

        The list of common points is obtained by combining the list of points from each strain field.

        Parameters
        ----------
        strains: list
            List of input strain fields.
        stack_mode: str
            A mode to stack the scalar fields. Valid values are 'complete' and 'common'
        resolution: float
        Two points are considered the same if they are separated by a distance smaller than this quantity


        Returns
        -------

        """
        # Validate all strains are strain fields
        for strain in strains:
            assert isinstance(strain, _StrainField), f'{strain} is not a StrainField object'
        fields = [strain.field for strain in strains]
        fields_stacked = stack_scalar_field_samples(*fields, stack_mode=stack_mode, resolution=resolution)
        strains_stacked = list()
        for strain, field_stacked in zip(strains, fields_stacked):
            strain_stacked = StrainField()
            strain_stacked._field = field_stacked
            strain_stacked._peak_collection = strain._peak_collection
            strain_stacked._single_scans = strain._single_scans
            strain_stacked._filenames = strain.filenames[:]
            strains_stacked.append(strain_stacked)
        return strains_stacked

    def __init__(self,
                 filename: str = '',
                 projectfile: Optional[HidraProjectFile] = None,
                 peak_tag: str = '',
                 hidraworkspace: Optional[HidraWorkspace] = None,
                 peak_collection: Optional[PeakCollection] = None,
                 point_list: Optional[PointList] = None) -> None:
        r"""
        Converts a HidraWorkspace and PeakCollection into a ScalarField
        """
        super().__init__()
        # list of underlying StrainFields
        self._strains: List[StrainFieldSingle] = []
        # the first element is the index of the winning StrainField
        # the second index is the index into the strain_values array
        self._winners: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._point_list: Optional[PointList] = None

        # Create a strain field from a single scan, if so requested
        single_scan_kwargs = dict(filename=filename, projectfile=projectfile, peak_tag=peak_tag,  # type: ignore
                                  hidraworkspace=hidraworkspace, peak_collection=peak_collection,  # type: ignore
                                  point_list=point_list)  # type: ignore
        if True in [bool(v) for v in single_scan_kwargs.values()]:  # at least one argument is not empty
            self._strains.append(StrainFieldSingle(**single_scan_kwargs))  # type: ignore
            # copy the pointlist from the only child that exists
            self._point_list = self._strains[0].point_list

        # otherwise it was an empty constructor and only initialize the starting layout

    def __rmul__(self, other: List['StrainField']) -> List['StrainField']:
        r"""
        Stack a list of strains along with this strain.

        Parameters
        ----------
        other: list
            Each item is a ~pyrs.dataobjects.fields.StrainField object.

        Return
        ------
        list
            List of stacked strains. Each item is a ~pyrs.dataobjects.fields.StrainField object.
        """
        if isinstance(other, (list, tuple)):
            for strain in other:
                if not isinstance(strain, _StrainField):
                    raise TypeError(f'{strain} is not a StrainField object')

            return self.__class__.stack_strains(*other, self,
                                                resolution=DEFAULT_POINT_RESOLUTION,
                                                stack_mode='union')
        else:
            raise NotImplementedError(f'Unable to multiply objects of type {str(other.__class__)} and StrainField')

    @property
    def filenames(self) -> List['str']:
        filenames: List[str] = []
        if self._strains is not None:
            for strain in self._strains:
                if isinstance(strain, StrainFieldSingle):
                    filenames.extend(strain.filenames)
                else:
                    msg = 'StrainField encountered a class it can\'t work with: ' \
                        + str(strain.__class__.___name__)
                    raise RuntimeError(msg)
        return filenames

    @property
    def field(self):
        r"""
        Fetch the strain values and errors for the list of sample points.

        Returns
        -------
        ~pyrs.dataobjects.fields.ScalarFieldSamples
        """
        if len(self._strains) == 1:
            if len(self._strains[0]) == len(self._point_list):
                return self._strains[0].field
            else:
                # update the zeroth strain with the extended PointList
                # other values are copied
                filename = self._strains[0].filenames[0] if self._strains[0].filenames else []
                self._strains[0] = StrainFieldSingle(filename=filename,
                                                     peak_collection=self._strains[0].peak_collections[0],
                                                     point_list=self.point_list)
                return self._strains[0].field
        else:
            # make sure there is enough information to create the field
            if not self._winners:
                raise RuntimeError('List of winners has not been initialized')
            if not self.point_list:
                raise RuntimeError('The PointList has not been initialized')

            num_values = len(self)
            values = np.full(num_values, NOT_MEASURED_NUMPY, dtype=float)
            errors = np.full(num_values, NOT_MEASURED_NUMPY, dtype=float)

            # loop over the winning strain indices
            for strain_index in set(self._winners.scan_indexes):
                # pick out the indices for this StrainField
                indices = np.where(self._winners.scan_indexes == strain_index)
                # get handle to the strain and uncertainties
                strain_i, error_i = self._strains[strain_index].peak_collections[0].get_strain()

                # find points of the current single-scan strain's list contributing to the overall list of points
                idx = self._winners.point_indexes[indices]
                values[indices], errors[indices] = strain_i[idx], error_i[idx]

            return ScalarFieldSample('strain', values, errors,
                                     self.point_list.vx,
                                     self.point_list.vy,
                                     self.point_list.vz)

    @property
    def peak_collections(self) -> List[PeakCollection]:
        r"""
        Retrieve the peak collection objects associated to this (possibly composite) strain field.

        Returns
        -------
        list
        """
        if self._strains is None:
            raise RuntimeError('List of peaks was not initialized')
        peaks: List[PeakCollection] = []
        for strain in self._strains:
            peaks.extend(strain.peak_collections)
        return peaks

    @property
    def strains(self) -> List['StrainFieldSingle']:
        r"""
        List of strains associated to individual peak collections

        Returns
        -------
        list
        """
        return self._strains

    @property
    def point_list(self):
        return self._point_list


def generateParameterField(parameter: str,
                           hidraworkspace: HidraWorkspace,
                           peak_collection: PeakCollection) -> ScalarFieldSample:
    '''Converts a HidraWorkspace and PeakCollection into a ScalarFieldSample for a specify peak parameter'''
    # extract positions
    pointlist = hidraworkspace.get_pointlist()

    # put together values
    if parameter in ('Center', 'Height', 'FWHM', 'Mixing', 'A0', 'A1', 'Intensity'):
        values, errors = peak_collection.get_effective_params()  # type: ignore
        values = values[parameter]
        errors = errors[parameter]
    else:  # dspacing_center, d_reference, strain
        values, errors = getattr(peak_collection, f'get_{parameter}')()  # type: ignore

    return ScalarFieldSample(parameter, values, errors,
                             pointlist.vx, pointlist.vy, pointlist.vz)


def aggregate_scalar_field_samples(*args) -> 'ScalarFieldSample':
    r"""
    Bring in together several scalar field samples of the same name. Overlaps can occur
    if a sample point is present in both samples.

    Parameters
    ----------
    args: list
        multiple ~pyrs.dataobjects.fields.ScalarFieldSample objects.

    Returns
    -------
    ~pyrs.dataobjects.fields.ScalarFieldSample
    """
    assert len(args) > 1, 'We need at least two PointList objects to aggregate'
    for arg in args:
        assert isinstance(arg, ScalarFieldSample), 'One of the arguments is not a ScalarFieldSample object.'
    aggregated_field = args[0]  # start with the point list of the first scalar field
    for field_sample in args[1:]:
        aggregated_field = aggregated_field.aggregate(field_sample)  # aggregate remaining lists, one by one
    return aggregated_field


def fuse_scalar_field_samples(*args, resolution: float = DEFAULT_POINT_RESOLUTION,
                              criterion: str = 'min_error') -> 'ScalarFieldSample':
    r"""
    Bring in together several scalar field samples of the same name. Overlaps are resolved
    according to a selection criterion.

    Parameters
    ----------
    args list
        multiple ~pyrs.dataobjects.fields.ScalarFieldSample objects.
    resolution
    criterion: str
        Criterion by which to resolve which sample points out of two (or more) ends up being selected,
        while the rest of the sample points are discarded is discarded. Possible values are:
        'min_error': the sample with the minimal uncertainty is selected.

    Returns
    -------
    ~pyrs.dataobjects.fields.ScalarFieldSample
    """
    aggregated_field = aggregate_scalar_field_samples(*args)
    return aggregated_field.coalesce(criterion=criterion, resolution=resolution)


@unique_enum
class StressType(Enum):
    DIAGONAL = 'diagonal'  # full calculation
    IN_PLANE_STRAIN = 'in-plane-strain'  # assumes strain33 is zero
    IN_PLANE_STRESS = 'in-plane-stress'  # assumes stress33 is zero

    @staticmethod
    def get(stresstype):
        if isinstance(stresstype, StressType):
            return stresstype
        else:
            try:
                stresskey = str(stresstype).upper().replace('-', '_')
                return StressType[stresskey]
            except KeyError:  # give clearer error message
                raise KeyError('Cannot determine stress type from "{}"'.format(stresstype))


class Direction(Enum):
    X = 'x'
    Y = 'y'
    Z = 'z'

    @staticmethod
    def get(direction):
        if isinstance(direction, Direction):
            return direction
        else:
            if direction == '11':
                return Direction.X
            elif direction == '22':
                return Direction.Y
            elif direction == '33':
                return Direction.Z
            try:
                return Direction(str(direction).upper())
            except KeyError:  # give clearer error message
                raise KeyError('Cannot determine direction type from "{}"'.format(direction))

    @property
    def ii(self) -> str:
        r"""
        Two-number string representation of the direction. One of ('11', '22', '33')

        Returns
        -------
        str
        """
        one_to_two = dict(x='11', y='22', z='33')
        return one_to_two[self.value]


class StressField:
    r"""
    Calculate the three diagonal components of the stress tensor, assuming a diagonal strain tensor.

    Strains along two or three directions are used to calculate the stress. Three types of sampling
    are considered:
    - diagonal: strains along three different directions are used
    - in plane strain: the strain along the third direction is, by definition, zero.
    - in plane stress: the stress along the third direction is, by definition, zero.

    The formulas for the calculation of the stress for the different types of sampling are:

    Diagonal (:math:`i` runs from 1 to 3):
    .. math:
        \sigma_{ii} = \frac{E}{1 + \nu} \left( \epsilon_{ii} + \frac{\nu}{1 - 2 \nu} (\epsilon_{11} + \epsilon_{22} + \epsilon_{33}) \right)

    In plane strain (:math:`i` runs from 1 to 3):
    .. math:
        \sigma_{ii} = \frac{E}{1 + \nu} \left( \epsilon_{ii} + \frac{\nu}{1 - 2 \nu} (\epsilon_{11} + \epsilon_{22}) \right(
        \epsilon_{33} \equiv 0

    In plane stress (:math:`i` runs from 1 to 2):
    .. math:
        \sigma_{ii} = \frac{E}{1 + \nu} \left( \epsilon_{ii} + \frac{\nu}{1 - \nu} (\epsilon_{11} + \epsilon_{22} + \epsilon_{33}) \right)
        \sigma_{33} \equiv 0

    where :math:`E` is Young's modulus, and :math:`\nu` is Poisson's ratio/

    Objects of this class store strains and stresses along the three directions, but only
    one direction is accessible at any given time. Function ~pyrs.dataobjects.fields.StressField.select
    allows the user to change the accessible direction. After selection, the strain and stress
    are accessed via properties ~pyrs.dataobjects.fields.StressField.strain
    and ~pyrs.dataobjects.fields.StressField.stress

    Parameters
    ----------
    strain11: ~pyrs.dataobjects.fields.StrainField
        Strain sample along the first direction
    strain22: ~pyrs.dataobjects.fields.StrainField
        Strain sample along the second direction
    strain33: ~pyrs.dataobjects.fields.StrainField
        Strain sample along the third direction
    youngs_modulus: float
    poisson_ratio: float
    stress_type: str, ~pyrs.dataobjects.fields.StressType
        If a string, one of ('diagonal', 'in-plane-strain', 'in-plane-stress')
    """  # noqa E501

    def __init__(self, strain11: StrainField, strain22: StrainField, strain33: StrainField,
                 youngs_modulus: float, poisson_ratio: float,
                 stress_type: Union[StressType, str] = StressType.DIAGONAL) -> None:
        self.stress11, self.stress22, self.stress33 = None, None, None

        self._youngs_modulus = youngs_modulus
        self._poisson_ratio = poisson_ratio

        # Stack strains
        self.stress_type = StressType.get(stress_type)
        strain_triad = [strain11, strain22, strain33]
        self._strain11, self._strain22, self._strain33 = self._stack_strains(*strain_triad)  # type: ignore

        # Enforce self._strain33 is zero for in-plane strain, or back-calculate it when in-plane stress
        if self.stress_type == StressType.IN_PLANE_STRAIN:
            # there is no in-plane strain component
            points = PointList([self.x, self.y, self.z])
            peaks = PeakCollectionLite(str(StressType.IN_PLANE_STRAIN),
                                       np.zeros(self.size, dtype=float),
                                       np.zeros(self.size, dtype=float))
            self._strain33 = StrainField(peak_collection=peaks, point_list=points)
        elif self.stress_type == StressType.IN_PLANE_STRESS:
            self._strain33 = self._strain33_when_inplane_stress()

        # Calculate stress fields, and strain33 if stress_type=StressType.IN_PLANE_STRESS
        stress11, stress22, stress33 = self._calc_stress_components()  # returns unumpy.array objects
        self._initialize_stress_fields(stress11, stress22, stress33)

        # At any given time, the StresField object access only one of the 11, 22, and 33 directions
        self.direction, self._stress_selected, self._strain_selected = None, None, None
        self.select(Direction.X)  # initialize the selected direction to be the 11 direction

    def __getitem__(self, direction) -> ScalarFieldSample:
        r"""
        Stress along one of the directions.

        This operation doesn't change the currently accessible direction.

        Parameters
        ----------
        direction: str
            One of ('11', '22', '33')

        Returns
        -------
        ~pyrs.databojects.fields.ScalarFieldSample
        """
        assert direction in ('11', '22', '33'), 'The direction is not one of ("11", "22", "33")'
        return getattr(self, f'stress{direction}')

    def __iter__(self) -> Iterator[ScalarFieldSample]:
        r"""
        Access the stress along the different directions

        This operation doesn't change the currently accessible direction.

        Returns
        -------
        ~pyrs.dataobjects.fields.ScalarFieldSample
        """
        for stress_component in (self.stress11, self.stress22, self.stress33):
            yield stress_component  # type: ignore

    def _initialize_stress_fields(self, stress11: np.ndarray, stress22: np.ndarray, stress33: np.ndarray) -> None:
        r"""
        Instantiate ScalarFieldSample objects, to represent the stresses.

        The sampling spatial points are those given by the private strain object _strain11

        Parameters
        ----------
        stress11: np.ndarray
            Values and errors of the stress along the first direction
        stress22 np.ndarray
            Values and errors of the stress along the second direction
        stress33 np.ndarray
            Values and errors of the stress along the third direction
        """
        for stress, attr in zip((stress11, stress22, stress33), ('stress11', 'stress22', 'stress33')):
            values, errors = unumpy.nominal_values(stress), unumpy.std_devs(stress)
            setattr(self, attr, ScalarFieldSample('stress', values, errors, self.x, self.y, self.z))

    def _calc_stress_components(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Calculate the values and errors for each of the diagonal stress fields

        The formulas for the calculation of the stress for the different types of sampling are:

        Diagonal (:math:`i` runs from 1 to 3):
        .. math:
            \sigma_{ii} = \frac{E}{1 + \nu} \left( \epsilon_{ii} + \frac{\nu}{1 - 2 \nu} (\epsilon_{11} + \epsilon_{22} + \epsilon_{33}) \right)

        In plane strain (:math:`i` runs from 1 to 3):
        .. math:
            \sigma_{ii} = \frac{E}{1 + \nu} \left( \epsilon_{ii} + \frac{\nu}{1 - 2 \nu} (\epsilon_{11} + \epsilon_{22}) \right(
            \epsilon_{33} \equiv 0

        In plane stress (:math:`i` runs from 1 to 2):
        .. math:
            \sigma_{ii} = \frac{E}{1 + \nu} \left( \epsilon_{ii} + \frac{\nu}{1 - \nu} (\epsilon_{11} + \epsilon_{22} + \epsilon_{33}) \right)
            \sigma_{33} \equiv 0

        where :math:`E` is Young's modulus, and :math:`\nu` is Poisson's ratio/

        Returns
        -------
        list
            Each item is a ~unumpy.array object, corresponding to the values and error for one of the stress fields
        """  # noqa: E501
        youngs_modulus, poisson_ratio = self.youngs_modulus, self.poisson_ratio
        prefactor = youngs_modulus / (1 + poisson_ratio)

        strain11, strain22 = self._strain11.sample, self._strain22.sample  # unumpy.arrays
        sample_zero = unumpy.uarray(np.zeros(self.size, dtype=float), np.zeros(self.size, dtype=float))

        # calculate the additive trace
        if self.stress_type == StressType.DIAGONAL:
            strain33 = self._strain33.sample  # type: ignore
        else:
            strain33 = sample_zero
        f = 1.0 if self.stress_type == StressType.IN_PLANE_STRESS else 2.0
        additive = poisson_ratio * (strain11 + strain22 + strain33) / (1 - f * poisson_ratio)

        # Calculate the stresses
        stress11 = prefactor * (strain11 + additive)
        stress22 = prefactor * (strain22 + additive)
        if self.stress_type in (StressType.DIAGONAL, StressType.IN_PLANE_STRAIN):
            stress33 = prefactor * (strain33 + additive)
        elif self.stress_type == StressType.IN_PLANE_STRESS:
            stress33 = sample_zero
        else:
            raise ValueError('Cannot calculate stress of type {}'.format(self.stress_type))

        return stress11, stress22, stress33

    def _stack_strains(self, strain11: StrainField,
                       strain22: StrainField,
                       strain33: StrainField) -> Union[List[StrainField], Any]:
        r"""
        Stach the strain samples taken along the the three different directions

        Calculation of the stresses require that the three strain samples are defined over the same set
        of sampling points, but this is not usually the case. It is necessary to stack them first. Thus,
        there will be sample points for which only one or two of the strain have been measured. For
        instance, the strain along the third direction (strain33) may be missing at one sample point.
        We then insert `strain33 = nan` at this particular sample point.

        Parameters
        ----------
        strain11: ~pyrs.dataobjects.fields.StrainField
            strain sample along the first direction
        strain22: ~pyrs.dataobjects.fields.StrainField
            strain sample along the second direction
        strain33: ~pyrs.dataobjects.fields.StrainField
            strain sample along the third direction

        Returns
        -------
        list
        """
        if self.stress_type == StressType.DIAGONAL:
            assert strain33 is not None, 'strain33 is None but the selected stress type is "diagonal"'
            return strain11 * strain22 * strain33  # type: ignore
        # strain33 is yet undefined, so it's assigned a value of `None`
        return strain11 * strain22 + [None]  # type: ignore

    def _strain33_when_inplane_stress(self) -> StrainField:
        r"""
        Calculate :math:`\epsilon_{33}` under the assumption :math:`\sigma_{33} \equiv 0`.

        .. math::
            \epsilon_{33} = \frac{\nu}{\nu - 1} (\epsilon_{11} + \epsilon_{22})

        Returns
        -------
        ~pyrs.dataobjects.fields.StrainField
        """
        factor = self.poisson_ratio / (self.poisson_ratio - 1)
        strain33 = factor * (self._strain11.sample + self._strain22.sample)  # unumpy.array
        values, errors = unumpy.nominal_values(strain33), unumpy.std_devs(strain33)
        peaks = PeakCollectionLite(str(StressType.IN_PLANE_STRESS),
                                   values, errors)
        return StrainField(peak_collection=peaks,
                           point_list=PointList([self.x, self.y, self.z]))

    @property
    def size(self) -> int:
        r"""Total number of sampling points, after stacking"""
        return len(self._strain11)

    @property
    def point_list(self) -> PointList:
        r"""
        The stacked set of sample points

        Returns
        -------
        pyrs.dataobjects.sample_logs.PointList
        """
        return self._strain11.point_list

    @property
    def x(self) -> np.ndarray:
        r"""
        Coordinates of the stacked set of sample points along the first direction

        Returns
        -------
        np.ndarray
        """
        return self._strain11.x

    @property
    def y(self) -> np.ndarray:
        r"""
        Coordinates of the stacked set of sample points along the second direction

        Returns
        -------
        np.ndarray
        """
        return self._strain11.y

    @property
    def z(self) -> np.ndarray:
        r"""
        Coordinates of the stacked set of sample points along the third direction

        Returns
        -------
        np.ndarray
        """
        return self._strain11.z

    @property
    def coordinates(self) -> np.ndarray:
        r"""
        Coordinates of the stacked set of sample points

        Returns
        -------
        np.ndarray
        """
        return self._strain11.coordinates

    @property
    def strain11(self) -> StrainField:
        return self._strain11

    @property
    def strain22(self) -> StrainField:
        return self._strain22

    @property
    def strain33(self) -> StrainField:
        return self._strain33  # type: ignore

    @property
    def youngs_modulus(self) -> float:
        r"""
        Input Young's mudulus

        Returns
        -------
        float
        """
        return self._youngs_modulus

    @property
    def poisson_ratio(self) -> float:
        r"""
        Input Poisson's ratio

        Returns
        -------
        float
        """
        return self._poisson_ratio

    @property
    def values(self) -> np.ndarray:
        r"""
        Stress values along the currently selected direction

        Returns
        -------
        np.ndarray
        """
        assert self._stress_selected, 'No direction has yet been selected'
        return self._stress_selected.values

    @property
    def errors(self) -> np.ndarray:
        r"""
        Stress uncertainties along the currently selected direction

        Returns
        -------
        np.ndarray
        """
        assert self._stress_selected, 'No direction has yet been selected'
        return self._stress_selected.errors

    @property
    def strain(self) -> Optional[StrainField]:
        r"""
        Strain (after stacking) along the currently selected direction

        Returns
        -------
        ~pyrs.dataobjects.fields.StrainField
        """
        return self._strain_selected

    @property
    def stress(self) -> Optional[StrainField]:
        r"""
        Strain (after stacking) along the currently selected direction

        Returns
        -------
        ~pyrs.dataobjects.fields.ScalarSampleField
        """
        return self._stress_selected

    def select(self, direction: Union[Direction, str]) -> None:
        r"""
        Select one of the three directions

        Selecting a direction updates the accessible ~pyrs.dataobjects.fields.StressField.strain
        and ~pyrs.dataobjects.fields.StressField.

        Parameters
        ----------
        direction: str, ~pyrs.dataobjects.fields.Direction
            One of strigs '11', '22', '33', or a Direction enumeration object.
        """
        self.direction = Direction.get(direction)
        direction_to_stress = {Direction.X: self.stress11, Direction.Y: self.stress22, Direction.Z: self.stress33}
        direction_to_strain = {Direction.X: self._strain11, Direction.Y: self._strain22, Direction.Z: self._strain33}
        self._stress_selected = direction_to_stress[self.direction]  # type: ignore
        self._strain_selected = direction_to_strain[self.direction]  # type: ignore

    def to_md_histo_workspace(self, name: str = '',
                              interpolate: bool = True,
                              method: str = 'linear', fill_value: float = float('nan'), keep_nan: bool = True,
                              resolution: float = DEFAULT_POINT_RESOLUTION,
                              criterion: str = 'min_error'
                              ) -> IMDHistoWorkspace:
        r"""
        Save the selected stress field into a MDHistoWorkspace. Interpolation of the sample points is carried out
        by default.

        Parameters `method`, `fill_value`, `keep_nan`, `resolution` , and `criterion` are  used only if
        `interpolate` is `True`.

        Parameters
        ----------
        name: str
            Name of the output workspace.
        interpolate: bool
            Interpolate the scalar field sample of a regular 3D grid given by the extents of the sample points.
        method: str
            Method of interpolation. Allowed values are 'nearest' and 'linear'
        fill_value: float
            Value used to fill in for requested points outside the input points.
        keep_nan: bool
            Incorporate :math:`nan` found in the sample points into the interpolated field sample.
        resolution: float
            Two points are considered the same if they are separated by a distance smaller than this quantity.
        criterion: str
            Criterion by which to resolve which sample points out of two (or more) ends up being selected,
            while the rest of the sample points are discarded is discarded. Possible values are:
            'min_error': the sample with the minimal uncertainty is selected.
        Returns
        -------
        MDHistoWorkspace
        """
        export_kwags = dict(interpolate=interpolate, method=method, fill_value=fill_value,
                            keep_nan=keep_nan, resolution=resolution, criterion=criterion)
        return self._stress_selected.to_md_histo_workspace(name, **export_kwags)  # type: ignore


def stack_scalar_field_samples(*fields,
                               stack_mode: str = 'complete',
                               resolution: float = DEFAULT_POINT_RESOLUTION) -> List[ScalarFieldSample]:
    r"""
    Evaluate a list of scalar field samples on a list of points, obtained by combining the list of points of each
    scalar field sample.

    The selection of the evaluation points, as well as the evaluation values for the fields depends on the
    `stack_mode`.

    In the 'complete' stack mode, the evaluation list of points result from the union of the sample points
    from all the input scalar field samples. If, say, a sample point :math:`p_a` of field A is not
    a sample point of field B, then we assign an evaluation value of :math:`nan` for field B
    at sample point :math:`p_a`.

    In the 'common' stack mode, the evaluation list of points result from the intersection of the sample points
    from all the input scalar fields. Hence, all input fields are guaranteed to have values at these
    common evaluation  points.

    Parameters
    ----------
    fields: list
        A list of ~pyrs.dataobjects.fields.ScalarFieldSample objects
    stack_mode: str
        A mode to stack the scalar fields. Valid values are 'complete' and 'common'
    resolution: float
        Two points are considered the same if they are separated by a distance smaller than this quantity

    Returns
    -------
    list
        List of ~pyrs.dataobjects.fields.ScalarFieldSample stacked objects. Input order is preserved
    """
    valid_stack_modes = ('union', 'intersection', 'complete', 'common')
    if stack_mode not in valid_stack_modes:
        raise ValueError(f'{stack_mode} is not a valid stack mode. Valid modes are {valid_stack_modes}')
    # map to cannonical names
    if stack_mode == 'complete':
        stack_mode = 'union'
    if stack_mode == 'common':
        stack_mode = 'intersection'

    # If it so happens that one of the input fields contains two (or more) sampled points separated by
    # less than `resolution` distance, we have to discard all but one of them.
    fields = tuple([field.coalesce(resolution) for field in fields])

    fields_count, fields_indexes = len(fields), list(range(len(fields)))

    # We are going to aggregate the sample points for all the input fields. An index,
    # `aggregated_point_index`, will index all these aggregated points. Thus, each
    # point will have two indexes associated. Index `point_index` will identify the point
    # in the list of points associated to a particular input field.
    # For every aggregated point, we want to remember which input field it came from. Index
    # field_index will identify the input field within list `fields`.
    #
    # List `field_point_index_pair` will give us the `field_index` and the `point_index`
    # for every `aggregated_point_index`.
    #     field_point_index_pair[aggregated_point_index] == (field_index, point_index)
    field_point_index_pair = list()
    current_field_index = 0
    for field in fields:
        field_point_index_pair.extend([(current_field_index, i) for i in range(len(field))])
        current_field_index += 1
    # aggregate the sample points from all scalar fields
    aggregated_points = aggregate_point_lists(*[field.point_list for field in fields])

    # We cluster the aggregated sample points according to euclidean distance. Points within
    # one cluster have mutual distances below resolution, so they can be considered the same sample point.
    # We will associate a "common point" to each cluster, wich is the geometrical center
    # of all the points in the cluster.
    # Each cluster amounts to one common point that can be resolved from the the sample points belonging
    # to other clusters.
    # Each cluster may (or may not) contain one point from each input field. If the cluster contains
    # one point from each input field, then we assert that all the input fields have been measured
    # at the common point.
    # If the cluster is missing a point from one (or more) of the input fields, and we have selected
    # `stack_mode=complete`, then we assign a value of`nan` to those fields not present in the cluster.
    clusters = aggregated_points.cluster(resolution=resolution)

    field_values: List[List] = [[] for _ in fields_indexes]  # measurements of each field at the common points
    field_errors: List[List] = [[] for _ in fields_indexes]  # shape = (input fields, aggregated points)
    x, y, z = [], [], []  # coordinates of the common points
    # Process one cluster at a time. Clusters are returned as a list, with biggest cluster the first and smallest
    # cluster the last one. Each item (each cluster) in a list of aggregated point indexes, representing
    # the points within the cluster
    for aggregated_indexes in clusters:
        # if we selected stack_mode='common' and the cluster is missing at least one point from an input field,
        # then the common point is not common to all fields, so we discard the point
        if stack_mode == 'intersection' and len(aggregated_indexes) < fields_count:
            break  # common point not common to all fields. Discard and go to the next cluster
        # Here we either selected stack_mode=complete or the common point is common to all input fields
        cluster_x, cluster_y, cluster_z = 0, 0, 0  # cluster's common point coordinates
        # initialize the measurements of the input fields in this cluster as 'nan'
        fields_value_in_cluster = [float('nan')] * fields_count
        fields_error_in_cluster = [float('nan')] * fields_count
        # Look up each point within this cluster, finding out the value of each input field.
        for aggregated_index in aggregated_indexes:
            field_index, point_index = field_point_index_pair[aggregated_index]
            field = fields[field_index]  # input field associated to the field_index
            fields_value_in_cluster[field_index] = field.values[point_index]  # update the list of measurments
            fields_error_in_cluster[field_index] = field.errors[point_index]
            cluster_x += field.x[point_index]
            cluster_y += field.y[point_index]
            cluster_z += field.z[point_index]
        # Now we have the coordinates of the common point as well as measurements for all the input fields
        # to be associated to the common point. We update the appropriate lists to store this information
        for field_index in fields_indexes:
            field_values[field_index].append(fields_value_in_cluster[field_index])
            field_errors[field_index].append(fields_error_in_cluster[field_index])
        x.append(cluster_x / len(aggregated_indexes))
        y.append(cluster_y / len(aggregated_indexes))
        z.append(cluster_z / len(aggregated_indexes))

    # For each input field we now have a new, stacked, field. This field is measured at the common points
    return [ScalarFieldSample(fields[i].name, field_values[i], field_errors[i], x, y, z) for i in fields_indexes]
