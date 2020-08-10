import copy
import functools
import numpy as np
from uncertainties import ufloat

from pyrs.core.workspaces import HidraWorkspace
from pyrs.dataobjects.constants import DEFAULT_POINT_RESOLUTION
from pyrs.dataobjects.fields import ScalarFieldSample
from pyrs.peaks import PeakCollection
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode

NOT_MEASURED = float('nan')  # indicates we didn't measure a quantity at a particular sample point


class MultiScan:

    def __init__(self, in_plane=None, resolution=DEFAULT_POINT_RESOLUTION):
        self.sample_points = []  # list of SamplePoint objects
        self.in_plane = in_plane  # one of (False, 'strain', 'stress')
        self.resolution = resolution
        # For each sample point, select which subset of peak measurements we pick
        self.filters = {'peak_tag': None, 'direction': None, 'run': None}
        self.poisson_ratio = None
        self.youngs_modulus = None

    def clear_filters(self):
        self.filters = {'peak_tag': None, 'direction': None, 'run': None}

    def check_filters(self):
        r"""
        Filter rules:
        - 'peak_tag' is required always
        - 'direction' and 'run' cannot be passed together. A particular run number already implies a direction, thus
          passing a run number and a direction can lead to no measurements if they contradict each other.
        - 'run' is required when retrieving other properties than strain or stress.
        """
        assert self.filters['peak_tag'] is not None, '"peak_tag" is required always'
        msg = '"direction" and "run" cannot be passed together'
        assert self.filters['direction'] is None or self.filters['run'] is None, msg

    def append(self, projectfile, direction):
        measurements = PeakMeasurement.load_project(projectfile, direction)

        for measurement in measurements:  # insert each measurement in the appropriate sample point
            xyz = measurement.xyz
            overlapping = False  # assume there's no sample point appropriate for this measurement
            for sample_point in self.sample_points:  # TODO optimize this double-loop, maybe a clustering?
                if np.sqrt((xyz - sample_point.xyz)**2) < self.resolution:
                    sample_point.append(measurement)  # append to existing sample point
                    overlapping = True
                    break
            if overlapping is False:  # new sample point
                self.sample_points.append(SamplePoint().append(measurement))

    @property
    def coordinates(self):
        np.array([s.xyz for s in self.sample_points])

    def values_and_errors(self, quantity):
        r"""
        Query the sample points for a particular quantity

        Parameters
        ----------
        quantity: str
            One of d0, strain, stress, FWHM
        """
        return zip(*[s.value_and_error(quantity) for s in self.sample_points])

    def to_md_histo_workspace(self, quantity):
        r"""
        Create an MD Histo Workspace

        Parameters
        ----------
        quantity: str
            One of d0, strain, stress, FWHM
        """
        values, errors = self.values_and_errors(quantity)
        x, y, z = self.coordinates.T
        field = ScalarFieldSample(quantity, values, errors, x, y, z)
        return field.to_md_histo_workspace(resolution=self.resolution)


class SamplePoint:

    def __init__(self):
        self.multi_scan = None  # reference to parent MultiScan object
        self.d0 = None
        self.measurements = []

    @property
    def xyz(self):
        return np.mean([m.xyz for m in self.measurements])

    def append(self, measurement):
        r"""Append a new measurement"""
        self.measurements.append(measurement)

    def compliants(self):
        r"""Find out which measurements are compliant with the filter selections"""
        assert self.multi_scan.check_filters()
        # ignore filtering selections with a value of None
        filters = {k: v for k, v in self.multi_scan.filters.items() if v is not None}
        # Find out which measurements comply with the filters
        measurements_compliant = []
        for measurement in self.measurements:
            complies = True
            for filter_key, filter_value in filters:
                if getattr(measurement, filter_key) != filter_value:
                    complies = False
                    break
            if complies:
                measurements_compliant.append(measurement)
        return measurements_compliant

    def value_and_error(self, quantity):
        r"""
        Parameters
        ----------
        quantity: str
            One of d, FWHM, strain, stress
        """
        if quantity in ('strain', 'stress'):  # a quantity not recorded by the experiment, but derived
            sample = getattr(self, quantity)  # an uncertainties.ufloat instance
            return sample.f, sample.n
        else:  # a quantity recorded by the experiment, find if for each measurement compliant with the filters
            measurements_compliant = self.compliants()
            if len(measurements_compliant) == 0: # no measurement complies with the filters
                return NOT_MEASURED, NOT_MEASURED
            elif len(measurements_compliant) == 1:
                measurement = measurements_compliant[0]
                return measurement.value_and_error(quantity)
            else:  # overlapping measurements
                values, errors = [m.value_and_error for m in measurements_compliant]
                minpos = errors.index(min(errors))  # measurement with the minimum error
                return values[minpos], errors[minpos]

    @property
    def strain(self):
        if self.multi_scan.filters['direction'] == '33':
            if self.multi_scan.in_plane == 'strain':
                return ufloat(0.0, 0.0)  # no strain by definition
            elif self.multi_scan.in_plane == 'stress':  # can be calculated with strain11 and strain22
                filters_other = copy.deepcopy(self.multi_scan.filters).update({'direction': '11'})
                strain11 = self._strain(filters_other)
                filters_other = copy.deepcopy(self.multi_scan.filters).update({'direction': '22'})
                strain22 = self._strain(filters_other)
                ps = self.multi_scan.poisson_ratio
                return ps * (ps - 1) * (strain11 + strain22)
        return self._strain()

    def _strain(self, filters_alternative=None):
        r"""Fish out the strain from the measurements"""
        filters = self.multi_scan.filters if filters_alternative is None else filters_alternative
        strain_candidates = [measurement.strain for measurement in self.compliants]
        if len(strain_candidates) == 0:
            return ufloat(NOT_MEASURED, NOT_MEASURED)  # no measurement complies with the filters
        elif len(strain_candidates) == 1:
            return strain_candidates[0]  # only one measurement complies with the filters
        else:
            return sorted(strain_candidates, key=lambda s: s.std_dev)[0]  # find the one with the smallest error

    @property
    def stress(self):
        # First, look at special cases
        if self.multi_scan.filters['direction'] == '33' and self.multi_scan.in_plane == 'stress':
            return ufloat(0.0, 0.0)  # by definition, no stress

        filters_other = copy.deepcopy(self.multi_scan.filters).update({'direction': '11'})
        strain11 = self._strain(filters_other)
        filters_other = copy.deepcopy(self.multi_scan.filters).update({'direction': '22'})
        strain22 = self._strain(filters_other)
        if self.multi_scan.filters['direction'] == '33' and self.multi_scan.in_plane == 'strain':
            strain33 = ufloat(0.0, 0.0)
        else:
            filters_other = copy.deepcopy(self.multi_scan.filters).update({'direction': '11'})
            strain33 = self._strain(filters_other)
        ym, ps = self.multi_scan.youngs_modulus, self.multi_scan.poisson_ratio
        term_isotropic = ps * (1 - 2 * ps) * (strain11 + strain22 + strain33)
        strainii = {'11': strain11, '22': strain22, '33': strain33}[self.multi_scan.filters['direction']]
        return ym * ps * strainii / (1 + ps)


class PeakMeasurement:
    @staticmethod
    def load_project(filename, direction):
        r"""
        Loads the peaks information from a Hydra project file, assigned to a particular direction

        Parameters
        ----------
        filename: str
            Absolute path
        direction: str
            one of '11', '22', '33'

        Returns
        -------
        list
            List of PeakMeasument objects
        """
        projectfile = HidraProjectFile(filename, HidraProjectFileMode.READONLY)
        hidraworkspace = HidraWorkspace()
        hidraworkspace.load_hidra_project(projectfile, load_raw_counts=False, load_reduced_diffraction=False)

        peak_tags = projectfile.read_peak_tags()
        if len(peak_tags) == 0:
            raise IOError('File "{}" does not have peaks defined'.format(filename))

        subruns = hidraworkspace.get_sub_runs()
        vx = hidraworkspace.get_sample_log_values('vx')
        vy = hidraworkspace.get_sample_log_values('vy')
        vz = hidraworkspace.get_sample_log_values('vz')

        measurements = list()
        for peak_tag in peak_tags:
            peak_collection = projectfile.read_peak_parameters(peak_tag)
            # verify the subruns are parallel
            if hidraworkspace.get_sub_runs() != peak_collection.sub_runs:  # type: ignore
                raise RuntimeError('Need to have matching subruns')
            d_values, d_errors = peak_collection.get_dspacing_center()
            all_params_values, all_params_errors = peak_collection.get_effective_params()
            for index, subrun in enumerate(subruns):
                peak = PeakMeasurement()
                peak.filename = filename
                peak.peak_collection = peak_collection
                peak.peak_tag = peak_tag
                peak.xyz = np.ndarray([vx[index], vy[index], vz[index]])
                peak.peak_properties['d'] = d_values[index], d_errors[index]
                param_values, param_errors = all_params_values[index], all_params_errors[index]
                for param_index, name in enumerate(['Center', 'Height', 'FWHM', 'Mixing', 'A0', 'A1', 'Intensity']):
                    peak.peak_properties[name] = param_values[param_index], param_errors[param_index]
            measurements.append(peak)

        projectfile.close()
        del projectfile
        return measurements

    def __init__(self, *args, **kwargs):
        self.filename = None
        self.peak_collection = None
        self.peak_tag = None
        self.peak_properties = {}
        self.xyz = None
















    def __init__(self):
        r"""Represents information for a peak. Almost like a database entry"""
        self.sample_point = None  # sample point associated to this measurement
        self.projectfile = None  # name of the hydra project file containing the peak collection
        self.run = None  # run number
        self.direction = None  # one of ('11', '22', '33')
        self.peak_collection = None  # reference to PeakCollection object
        self.subrun_index = None
        self.peak_tag = None
        self.peak_properties = {'d': None, 'FWHM': None}  # cache
        self.xyz = None

    def value_and_error(self, quantity):
        return self.peak_collection[quantity]

    @property
    def strain(self):
        return (ufloat(*self.peak_properties['d']) - self.sample_point.d0) / self.sample_point.d0
