from enum import Enum
from enum import unique as unique_enum
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from typing import TYPE_CHECKING, cast, List, Optional, Tuple, Union
from uncertainties import unumpy

from mantid.simpleapi import mtd, CreateMDWorkspace, BinMD
from mantid.api import IMDHistoWorkspace

from pyrs.core.workspaces import HidraWorkspace
from pyrs.dataobjects.sample_logs import PointList, aggregate_point_lists
from pyrs.peaks import PeakCollection  # type: ignore
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

    def interpolated_sample(self,
                            method: str = 'linear', fill_value: float = float('nan'),
                            keep_nan: bool = True,
                            resolution: float = DEFAULT_POINT_RESOLUTION,
                            criterion: str = 'min_error') -> 'ScalarFieldSample':
        r"""
        Interpolate the scalar field sample of a regular 3D grid given by the extents of the sample points.

        :math:`nan` values are disregarded when doing the interpolation, but they can be incorporated into
        the interpolated values by finding the point in the regular grid closest to a sample point
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
            Criterion by which to resolve which out of two (or more) samples is selected, while the rest is
            discarded. Possible values are:
            'min_error': the sample with the minimal uncertainty is selected.

        Returns
        -------
        ~pyrs.dataobjects.fields.ScalarFieldSample
        """
        # TODO deal with corner case when self.x (or .y or .z) only have one unique coordinate value
        # Corner case: the points fill a regular grid. Thus, there's no need to interpolate.
        if self.point_list.is_a_grid(resolution=resolution):
            return self.coalesce(resolution=resolution, criterion=criterion)  # get rid of duplicate points
        # regular 3D grids using the `extents` of the sample points
        grid_x, grid_y, grid_z = self.point_list.mgrid(resolution=resolution)
        field_finite = self.isfinite  # We need to remove non-finite values prior to interpolation
        field_finite = field_finite.coalesce(resolution=resolution, criterion=criterion)  # for neighbor sample points
        # Corner case: if scalar field sample `self` has `nan` values at the extrema of its extents,
        # then removing these non-finite values will result in scalar field sample `field_finite`
        # having `extents` smaller than those of`self`. Later when interpolating, some of the grid
        # points (determined using `extents` of `self`) will fall outside the `extents` of `field_finite`
        # and their values will have to be filled with the `fill_value` being passed on to function `griddata`.
        xyz = field_finite.coordinates
        values = griddata(xyz, field_finite.values, (grid_x, grid_y, grid_z), method=method, fill_value=fill_value)
        values = values.ravel()  # values has same shape as any of the x, y, z grids
        coordinates = np.array((grid_x, grid_y, grid_z)).T.reshape(-1, 3)
        if keep_nan is True:
            # For each sample point of `self` that has a `nan` field value, find the closest grid point and assign
            # a `nan` value to this point
            point_indexes_with_nan = np.where(np.isnan(self.values))[0]  # points of `self` with field values of `nan`
            if len(point_indexes_with_nan) > 0:
                coordinates_with_nan = self.coordinates[point_indexes_with_nan]
                _, grid_indexes = cKDTree(coordinates).query(coordinates_with_nan, k=1)
                values[grid_indexes] = float('nan')
        errors = griddata(xyz, field_finite.errors, (grid_x, grid_y, grid_z), method=method, fill_value=fill_value)
        errors = errors.ravel()
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
            Criterion by which to resolve which out of two (or more) samples is selected, while the rest is
            discarded. Possible values are:
            'min_error': the sample with the minimal uncertainty is selected (Nan values ignored).
        Returns
        -------
        ~pyrs.dataobjects.fields.ScalarFieldSample
        """
        def min_error(indexes):
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
        target_indexes.extend([point_indexes[0] for point_indexes in clusters[cluster_index:]])

        # create a ScalarFieldSample with the sample points corresponding to the target indexes
        return self.extract(sorted(target_indexes))

    def fuse(self, other: 'ScalarFieldSample',
             resolution: float = DEFAULT_POINT_RESOLUTION, criterion: str = 'min_error') -> 'ScalarFieldSample':
        r"""
        Bring in another scalar field sample and resolve the overlaps according to a selection criteria.

        Two samples are common if their corresponding sample points are apart from each other a distance
        smaller than the resolution distance.

        Parameters
        ----------
        other: ~pyrs.dataobjects.fields.ScalarFieldSample
        resolution: float
            Two points are considered the same if they are separated by a distance smaller than this quantity.
        criterion: str
            Criterion by which to resolve which out of two (or more) samples is selected, while the rest is
            discarded. Possible values are:
            'min_error': the sample with the minimal uncertainty is selected.
        Returns
        -------
        ~pyrs.dataobjects.fields.ScalarFieldSample
        """
        aggregate_sample = self.aggregate(other)  # combine both scalar field samples
        return aggregate_sample.coalesce(resolution=resolution, criterion=criterion)

    def to_md_histo_workspace(self, name: str, units: str = 'meter',
                              interpolate=True,
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
        units: str
            Units of the sample points.
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
            Criterion by which to resolve which out of two (or more) samples is selected, while the rest is
            discarded. Possible values are:
            'min_error': the sample with the minimal uncertainty is selected.
        Returns
        -------
        MDHistoWorkspace
        """
        # TODO units should be a member of this class
        if interpolate is True:
            sample = self.interpolated_sample(keep_nan=keep_nan, resolution=resolution, criterion=criterion)
        else:
            sample = self
        extents = sample.point_list.extents(resolution=resolution)  # triad of DirectionExtents objects
        for extent in extents:
            assert extent[0] < extent[1], f'min value of {extent} is not smaller than max value'
        extents_str = ','.join([extent.to_createmd for extent in extents])

        units_triad = ','.join([units] * 3)  # 'meter,meter,meter'

        # create an empty event workspace of the correct dimensions
        axis_labels = ('x', 'y', 'z')
        CreateMDWorkspace(OutputWorkspace='__tmp', Dimensions=3, Extents=extents_str,
                          Names=','.join(axis_labels), Units=units_triad)
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
            units ('meter'): str, length units of the sample points
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
        exporters_arguments = dict(MDHistoWorkspace=('name', 'units'), CSV=('file',))
        # Call the exporter
        exporter_arguments = {arg: kwargs[arg] for arg in exporters_arguments[form]}
        return exporters[form](*args, **exporter_arguments)


class StrainField(ScalarFieldSample):
    def __init__(self, filename: str = '',
                 projectfile: Optional[HidraProjectFile] = None,
                 peak_tag: str = '',
                 hidraworkspace: Optional[HidraWorkspace] = None,
                 peak_collection: Optional[PeakCollection] = None) -> None:
        '''Converts a HidraWorkspace and PeakCollection into a ScalarField'''
        VX, VY, VZ = 'vx', 'vy', 'vz'

        # get the workspace and peaks by resolving the supplied inputs
        hidraworkspace, peak_collection = StrainField.__to_wksp_and_peaks(filename, peak_tag,
                                                                          projectfile, hidraworkspace,
                                                                          peak_collection)

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

        strain, strain_error = peak_collection.get_strain()  # type: ignore

        # TODO the fixed name shouldn't bee needed with inheritence
        return super().__init__('strain', strain, strain_error, x, y, z)

    @staticmethod  # noqa: C901
    def __to_wksp_and_peaks(filename: str,
                            peak_tag: str,
                            projectfile: Optional[HidraProjectFile],
                            hidraworkspace: Optional[HidraWorkspace],
                            peak_collection: Optional[PeakCollection]) -> Tuple[HidraWorkspace, PeakCollection]:
        # load information from a file
        closeproject = False
        if filename:
            projectfile = HidraProjectFile(filename, HidraProjectFileMode.READONLY)
            closeproject = True
        elif TYPE_CHECKING:
            projectfile = cast(HidraProjectFile, projectfile)

        # create objects from the project file
        if projectfile:
            # create HidraWorkspace
            if not hidraworkspace:
                hidraworkspace = HidraWorkspace()
                hidraworkspace.load_hidra_project(projectfile, load_raw_counts=False, load_reduced_diffraction=False)

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
        elif TYPE_CHECKING:
            hidraworkspace = cast(HidraWorkspace, hidraworkspace)
            peak_collection = cast(PeakCollection, peak_collection)

        # verify that everything is set by now
        if (not hidraworkspace) or (not peak_collection):
            raise RuntimeError('Do not have both hidraworkspace and peak_collection defined')

        # convert the information into a usable form for setting up this object
        if hidraworkspace.get_sub_runs() != peak_collection.sub_runs:  # type: ignore
            raise RuntimeError('Need to have matching subruns')

        return hidraworkspace, peak_collection


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
    according to a selection criteria.

    Parameters
    ----------
    args list
        multiple ~pyrs.dataobjects.fields.ScalarFieldSample objects.
    resolution
    criterion: str
        Criterion by which to resolve which out of two (or more) samples is selected, while the rest is
        discarded. Possible values are:
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


class StressField(ScalarFieldSample):
    def __init__(self, strain11, strain22, strain33, youngs_modulus: float, poisson_ratio: float,
                 stress_type=StressType.DIAGONAL) -> None:
        self.direction = Direction.X  # as good of a default as any
        self.stress_type = StressType.get(stress_type)

        # currently only supports equal sized strains
        if len(strain11.values) != len(strain22.values) != len(strain33.values):
            raise RuntimeError('Number of StrainField values must be equal')

        # compare point lists to make sure they are identical
        if strain11.point_list != strain22.point_list:
            raise RuntimeError('Point lists for strain11 and strain22 are not compatible')
        if (self.stress_type == StressType.DIAGONAL) and (strain11.point_list != strain33.point_list):
            raise RuntimeError('Point lists for strain11 and strain33 are not compatible')

        # extract strain values - these are the epsilons in the equations
        # currently assumes all strains are parallel
        epsilon11 = unumpy.uarray(strain11.values, strain11.errors)
        epsilon22 = unumpy.uarray(strain22.values, strain22.errors)
        epsilon33 = unumpy.uarray(strain33.values, strain33.errors)

        # calculate stress
        if self.stress_type == StressType.DIAGONAL:
            self.__calc_diagonal_stress(youngs_modulus, poisson_ratio, epsilon11, epsilon22, epsilon33)
        elif self.stress_type == StressType.IN_PLANE_STRAIN:
            self.__calc_in_plane_strain(youngs_modulus, poisson_ratio, epsilon11, epsilon22)
        elif self.stress_type == StressType.IN_PLANE_STRESS:
            self.__calc_in_plane_stress(youngs_modulus, poisson_ratio,  epsilon11, epsilon22, epsilon33)
        else:
            raise ValueError('Cannot calculate stress of type {}'.format(self.stress_type))

        # set-up information for PointList
        x = strain11.point_list.vx
        y = strain11.point_list.vy
        z = strain11.point_list.vz
        # TODO need to fix up super.__init__ to not need the copy
        return super().__init__('stress', self.values, self.errors, x, y, z)

    def __calc_diagonal_stress(self, youngs_modulus, poisson_ratio, strain11, strain22, strain33):
        prefactor = youngs_modulus / (1 + poisson_ratio)
        additive = poisson_ratio * (strain11 + strain22 + strain33) / (1 - 2 * poisson_ratio)

        self.stress11 = prefactor * (strain11 + additive)
        self.stress22 = prefactor * (strain22 + additive)
        self.stress33 = prefactor * (strain33 + additive)

    def __calc_in_plane_strain(self, youngs_modulus, poisson_ratio, strain11, strain22):
        '''This assumes strain in the 33 direction (epsilon33) is zero'''
        prefactor = youngs_modulus / (1 + poisson_ratio)
        additive = poisson_ratio * (strain11 + strain22) / (1 - 2 * poisson_ratio)

        self.stress11 = prefactor * (strain11 + additive)
        self.stress22 = prefactor * (strain22 + additive)
        self.stress33 = prefactor * additive

    def __calc_in_plane_stress(self, youngs_modulus, poisson_ratio, strain11, strain22, strain33):
        '''This assumes stress in the 33 direction (sigma33) is zero'''
        prefactor = youngs_modulus / (1 + poisson_ratio)
        additive = poisson_ratio * (strain11 + strain22) / (1 - poisson_ratio)

        self.stress11 = prefactor * (strain11 + additive)
        self.stress22 = prefactor * (strain22 + additive)
        length = len(strain33)  # assumed to be zero
        self.stress33 = unumpy.uarray(np.zeros(length, dtype=float), np.zeros(length, dtype=float))

    @property
    def values(self) -> np.ndarray:
        if self.direction == Direction.X:
            return unumpy.nominal_values(self.stress11)
        elif self.direction == Direction.Y:
            return unumpy.nominal_values(self.stress22)
        elif self.direction == Direction.Z:
            return unumpy.nominal_values(self.stress33)

    @property
    def errors(self) -> np.ndarray:
        if self.direction == Direction.X:
            return unumpy.std_devs(self.stress11)
        elif self.direction == Direction.Y:
            return unumpy.std_devs(self.stress22)
        elif self.direction == Direction.Z:
            return unumpy.std_devs(self.stress33)

    def select(self, direction: str):
        self.direction = Direction.get(direction)


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

    # If it so happens for one of the input fields that it contains two (or more) sampled points separated by
    # less than `resolution` distance, we have to discard all but one of them.
    fields = tuple([field.coalesce(resolution) for field in fields])
    fields_count, fields_indexes = len(fields), list(range(len(fields)))

    # list `field_point_index_pair` gives us the scalar field index and the point index in the list
    # of sample points of that scalar field, for a given aggregated point index, that is:
    #     field_point_index_pair[aggregated_point_index] == (field_index, point_index)
    field_point_index_pair = list()
    current_field_index = 0
    for field in fields:
        field_point_index_pair.extend([(current_field_index, i) for i in range(len(field))])
        current_field_index += 1

    # aggregate the sample points from all scalar fields
    aggregated_points = aggregate_point_lists(*[field.point_list for field in fields])

    # cluster the aggregated sample points according to euclidean distance and resolution distance
    # Each cluster amounts to one sample point that can be resolved from the rest of sample points.
    # This sample point will have evaluations of one or more of the input fields.
    # If the cluster contains evaluations of all input fields, the associated sample point is a point
    # common to all input fields.
    # If one or more fields are not evaluated at the cluster's sample points, we set the evaluations
    # to `nan` for those missin fields when `stack_mode=complete`
    clusters = aggregated_points.cluster(resolution=resolution)

    # Look up each cluster.
    # The sample point associated to the cluster will be the average of the sample points contained in the
    # cluster. Remember all this sample points are within `resolution` distance from each other.
    #
    field_values: List[List] = [[] for _ in fields_indexes]  # evaluations of each field at the cluster's sample points
    field_errors: List[List] = [[] for _ in fields_indexes]  # shape = (input fields, aggregated points)
    x, y, z = [], [], []  # coordinates for the cluster's sample points
    for aggregated_indexes in clusters:
        if stack_mode == 'common' and len(aggregated_indexes) < fields_count:
            break  # there's a field missing in this cluster, thus it's not a sample point common to all fields
        cluster_x, cluster_y, cluster_z = 0, 0, 0  # cluster's sample point coordinates
        fields_value_in_cluster = [float('nan')] * fields_count  # value of each field at the cluster's sample point
        fields_error_in_cluster = [float('nan')] * fields_count
        for aggregated_index in aggregated_indexes:
            field_index, point_index = field_point_index_pair[aggregated_index]
            field = fields[field_index]  # ScalarFieldSample object, one of the input fields
            fields_value_in_cluster[field_index] = field.values[point_index]
            fields_error_in_cluster[field_index] = field.errors[point_index]
            cluster_x += field.x[point_index]
            cluster_y += field.y[point_index]
            cluster_z += field.z[point_index]
        for field_index in fields_indexes:
            field_values[field_index].append(fields_value_in_cluster[field_index])
            field_errors[field_index].append(fields_error_in_cluster[field_index])
        x.append(cluster_x / len(aggregated_indexes))
        y.append(cluster_y / len(aggregated_indexes))
        z.append(cluster_z / len(aggregated_indexes))

    # Construct the output ScalarFieldSample objects evaluated at the cluster's sample points
    return [ScalarFieldSample(fields[i].name, field_values[i], field_errors[i], x, y, z) for i in fields_indexes]
