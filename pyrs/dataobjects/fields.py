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
from enum import Enum
from enum import unique as unique_enum
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from typing import TYPE_CHECKING, cast, Iterator, List, Optional, Tuple, Union
from uncertainties import unumpy

from mantid.simpleapi import mtd, CreateMDWorkspace, BinMD
from mantid.api import IMDHistoWorkspace
from pathlib import Path

from pyrs.core.workspaces import HidraWorkspace
from pyrs.dataobjects.sample_logs import PointList, aggregate_point_lists
from pyrs.peaks import PeakCollection, PeakCollectionLite  # type: ignore
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore

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
        stack_kwargs = dict(resolution=DEFAULT_POINT_RESOLUTION, stack_mode='complete')
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
        extents_str = ','.join([extent.to_createmd for extent in extents])

        # create an empty event workspace of the correct dimensions
        axis_labels = ('x', 'y', 'z')
        CreateMDWorkspace(OutputWorkspace='__tmp', Dimensions=3, Extents=extents_str,
                          Names=','.join(axis_labels), Units='meter,meter,meter')
        # set the bins for the workspace correctly
        aligned_dimensions = [f'{label},{extent.to_binmd}'  # type: ignore
                              for label, extent in zip(axis_labels, extents)]
        aligned_kwargs = {f'AlignedDim{i}': aligned_dimensions[i] for i in range(len(aligned_dimensions))}
        BinMD(InputWorkspace='__tmp', OutputWorkspace=name, **aligned_kwargs)

        # remove original workspace, so sliceviewer doesn't try to use it
        mtd.remove('__tmp')

        # get a handle to the workspace
        wksp = mtd[name]
        # set the signal and errors
        dims = [extent.number_of_bins for extent in extents]
        wksp.setSignalArray(sample.values.reshape(dims))
        wksp.setErrorSquaredArray(np.square(sample.errors.reshape(dims)))

        return wksp

    def to_csv(self, file: str):
        raise NotImplementedError('This functionality has yet to be implemented')

    def export(self, *args, form='MDHistoWokspace', **kwargs):
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
        return exporters[form](*args, **exporter_arguments)


class StrainField:

    @staticmethod
    def fuse_strains(*args, resolution: float = DEFAULT_POINT_RESOLUTION,
                     criterion: str = 'min_error') -> 'ScalarFieldSample':
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
            assert isinstance(strain, StrainField), 'This input is not a StrainField object'
        # Iterative fusing
        strain, strain_other = args[0: 2]  # first two strains in the list
        strain_fused = strain.fuse_with(strain_other, resolution=resolution, criterion=criterion)
        for strain_other in args[2:]:  # fuse with remaining strains, one at a time
            strain_fused = strain_fused.fuse_with(strain_other, resolution=resolution, criterion=criterion)
        return strain_fused

    @staticmethod
    def stack_strains(*strains,
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
            assert isinstance(strain, StrainField), f'{strain} is not a StrainField object'
        fields = [strain._field for strain in strains]
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
        self._peak_collection: Optional[PeakCollection] = None
        self._filenames: List[str] = []
        # when the strain is composed of more than one scan, we keep references to the individual ones
        self._single_scans: List['StrainField'] = []
        # the resolved ScalarFieldSample with the math done
        self._field: Optional[ScalarFieldSample] = None

        # Create a strain field from a single scan, if so requested
        single_scan_kwargs = dict(filename=filename, projectfile=projectfile, peak_tag=peak_tag,  # type: ignore
                                  hidraworkspace=hidraworkspace, peak_collection=peak_collection,  # type: ignore
                                  point_list=point_list)  # type: ignore
        if True in [bool(v) for v in single_scan_kwargs.values()]:  # at least one argument is not empty
            self._initialize_with_single_scan(**single_scan_kwargs)  # type: ignore

    def __add__(self, other_strain):
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

    def __len__(self):
        return len(self._field)

    def __mul__(self, other):
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
        stack_kwargs = dict(resolution=DEFAULT_POINT_RESOLUTION, stack_mode='complete')
        if isinstance(other, StrainField):
            return self.__class__.stack_strains(self, other, **stack_kwargs)
        elif isinstance(other, (list, tuple)):
            for strain in other:
                if isinstance(strain, StrainField) is False:
                    raise TypeError(f'{strain} is not a {str(self.__class__)} object')
            return self.__class__.stack_strains(self, *other, **stack_kwargs)

    def __rmul__(self, other):
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
        stack_kwargs = dict(resolution=DEFAULT_POINT_RESOLUTION, stack_mode='complete')
        if isinstance(other, (list, tuple)):
            for strain in other:
                if isinstance(strain, StrainField) is False:
                    raise TypeError(f'{strain} is not a StrainField object')
            return self.__class__.stack_strains(*other, self, **stack_kwargs)
        else:
            raise NotImplementedError(f'Unable to multiply objects of type {str(other.__class__)} and StrainField')

    @property
    def filenames(self):
        return self._filenames

    @property
    def peak_collection(self):
        r"""
        Retrieve the peak collection associated to the strain field. Only valid when the field is not
        a composite of more than one scan.

        Raises
        ------
        RuntimeError
            There is more than one peak collection associated to this strain field

        Returns
        -------
        ~pyrs.dataobjects.fields.StrainField
        """
        if len(self._single_scans) > 1:
            raise RuntimeError('There is more than one peak collection associated to this strain field')
        return self._peak_collection

    @property
    def peak_collections(self):
        r"""
        Retrieve the peak collection objects associated to this (possibly composite) strain field.

        Returns
        -------
        list
        """
        return [field._peak_collection for field in self._single_scans]

    @property
    def values(self):
        return self._field.values

    @property
    def errors(self):
        return self._field.errors

    @property
    def sample(self):
        r"""
        Uncertainties arrays containing both values and errors.

        Returns
        -------
        ~unumpy.array
        """
        return self._field.sample

    @property
    def point_list(self):
        return self._field.point_list

    @property
    def x(self):
        return self._field.x

    @property
    def y(self):
        return self._field.y

    @property
    def z(self):
        return self._field.z

    @property
    def coordinates(self) -> np.ndarray:
        return self._field.coordinates  # type: ignore

    @staticmethod  # noqa: C901
    def __to_pointlist_and_peaks(filename: str,
                                 peak_tag: str,
                                 projectfile: Optional[HidraProjectFile],
                                 hidraworkspace: Optional[HidraWorkspace],
                                 peak_collection: Optional[PeakCollection],
                                 point_list: Optional[PointList]) -> Tuple[PointList, PeakCollection]:
        '''Take all of the various ways to supply the :py:obj:PointList and
        :py:obj:PeakCollection and convert them into those actual
        objects. For :py:obj:PeakCollection the first one found in the list
        * ``peak_collection``
        * ``projectfile``

        Similarly, for :py:obj:PoinList the first one found in the list
        * ``point_list``
        * ``hidraworkspace`` - taken from the :py:obj:SampleLogs
        * ``projectfile``
        '''
        # load information from a file
        closeproject = False
        if filename:
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

        # verify that everything is set by now
        if (not point_list) or (not peak_collection):
            raise RuntimeError('Do not have both point_list and peak_collection defined')

        if len(point_list) != len(peak_collection):
            msg = 'point_list and peak_collection are not equal length ({} != {})'.format(len(point_list),
                                                                                          len(peak_collection))
            raise ValueError(msg)

        return point_list, peak_collection

    def _initialize_with_single_scan(self,
                                     filename: str = '',
                                     projectfile: Optional[HidraProjectFile] = None,
                                     peak_tag: str = '',
                                     hidraworkspace: Optional[HidraWorkspace] = None,
                                     peak_collection: Optional[PeakCollection] = None,
                                     point_list: Optional[PointList] = None) -> None:
        # get the workspace and peaks by resolving the supplied inputs
        pointlist, peak_collection = StrainField.__to_pointlist_and_peaks(filename, peak_tag,
                                                                          projectfile, hidraworkspace,
                                                                          peak_collection,
                                                                          point_list)

        strain, strain_error = peak_collection.get_strain()  # type: ignore

        # set the names of files used to create the strain
        if filename:  #
            self._filenames = [Path(filename).name]
        elif hidraworkspace and hidraworkspace.hidra_project_file:  # get it from the workspace
            self._filenames = [Path(hidraworkspace.hidra_project_file).name]
        else:
            self._filenames = []  # do not know filenames
        self._peak_collection = peak_collection
        self._single_scans = [self]  # when the strain is composed of more than one scan, we keep references to them
        self._field = ScalarFieldSample('strain', strain, strain_error,
                                        pointlist.vx, pointlist.vy, pointlist.vz)

    def fuse_with(self, other_strain: 'StrainField',
                  resolution: float = DEFAULT_POINT_RESOLUTION, criterion: str = 'min_error') -> 'StrainField':
        r"""
        Fuse the current strain scan with another scan taken along the same direction.

        Resolve any occurring overlaps between the scans according to a selection criterion.

        Parameters
        ----------
        other_strain:  ~pyrs.dataobjects.fields.StrainField
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
        strain = StrainField()  # New empty strain object

        # Check there are no repeated scans
        for scan1 in self._single_scans:
            for scan2 in other_strain._single_scans:
                # Verify if the labels are references to the same object
                if scan1 == scan2:
                    raise RuntimeError(f'{self} and {other_strain} both contain scan {scan1}')
        strain._single_scans = self._single_scans + other_strain._single_scans
        strain._field = self._field.fuse_with(other_strain._field,  # type: ignore
                                              resolution=resolution, criterion=criterion)
        # copy over the filenames
        strain._filenames = []
        strain._filenames.extend(self._filenames)
        strain._filenames.extend(other_strain._filenames)

        return strain

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
        return self._field.to_md_histo_workspace(name, **export_kwags)  # type: ignore


def generateParameterField(parameter: str,
                           hidraworkspace: HidraWorkspace,
                           peak_collection: PeakCollection) -> ScalarFieldSample:
    '''Converts a HidraWorkspace and PeakCollection into a ScalarFieldSample for a specify peak parameter'''
    VX, VY, VZ = 'vx', 'vy', 'vz'

    lognames = hidraworkspace.get_sample_log_names()
    missing = []
    for logname in VX, VY, VZ:
        if logname not in lognames:
            missing.append(logname)
    if missing:
        raise RuntimeError('Failed to find positions in logs. Missing {}'.format(', '.join(missing)))

    # extract positions
    x = hidraworkspace.get_sample_log_values(VX)
    y = hidraworkspace.get_sample_log_values(VY)
    z = hidraworkspace.get_sample_log_values(VZ)

    if parameter in ('Center', 'Height', 'FWHM', 'Mixing', 'A0', 'A1', 'Intensity'):
        values, errors = peak_collection.get_effective_params()  # type: ignore
        values = values[parameter]
        errors = errors[parameter]
    else:  # dspacing_center, d_reference, strain
        values, errors = getattr(peak_collection, f'get_{parameter}')()  # type: ignore

    return ScalarFieldSample(parameter, values, errors, x, y, z)


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
        self._strain11, self._strain22, self._strain33 = self._stack_strains(strain11, strain22, strain33)

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
                       strain33: StrainField) -> Tuple[StrainField, StrainField, Optional[StrainField]]:
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
            return strain11 * strain22 * strain33
        return strain11 * strain22 + [None]  # strain33 is yet undefined, so it's assigned a value of `None`

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
    valid_stack_modes = ('complete', 'common')
    if stack_mode not in valid_stack_modes:
        raise ValueError(f'{stack_mode} is not a valid stack mode. Valid modes are {valid_stack_modes}')

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
        if stack_mode == 'common' and len(aggregated_indexes) < fields_count:
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
