# This is the virtual base class as the fitting frame
import numpy as np
from pyrs.core import workspaces
from pyrs.utilities import rs_project_file
from pyrs.core import peak_profile_utility
from pyrs.core.peak_profile_utility import PeakShape
from pyrs.utilities import checkdatatypes


class PeakFitEngine(object):
    """
    Virtual peak fit engine
    """

    def __init__(self, workspace, mask_name):
        """Initialization

        Parameters
        ----------
        workspace : pyrs.core.workspaces.HidrawWorkspace
            Hidra Workspace holding data
        mask_name : str
            name of mask ID (or main/None) for reduced diffraction data

        """
        # check
        checkdatatypes.check_type('Diffraction workspace', workspace, workspaces.HidraWorkspace)

        # for scipy: keep the np.array will be good enough
        self._hidra_wksp = workspace  # wksp = workspace
        self._mask_name = mask_name

        # wave length information
        self._wavelength_dict = None

        # Center of mass of peaks:  key = peak tag, value = ???
        self._peak_com_dict = dict()

        # Fitted peak parameters
        self._peak_collection_dict = dict()   # key: peak tag, value: PeakCollection

        return

    def calculate_center_of_mass(self, peak_tag, peak_range):
        """Calculate center of mass of peaks in the Mantid MatrixWorkspace as class variable

        Output: calculated peaks' centers of mass will be recored to self._peak_com_dict

        Parameters
        ----------
        peak_tag : str
            peak tag
        peak_range : tuple
            boundary of peak on 2theta-axis

        Returns
        -------
        None

        """
        # Create the array to hold center of mass and observed data point with highest value
        sub_run_array = self._hidra_wksp.get_sub_runs()
        num_sub_runs = len(sub_run_array)
        peak_center_vec = np.ndarray(shape=(num_sub_runs, 2), dtype='float')

        # Calculate COM and highest data point in 2theta
        for iws, sub_run_i in enumerate(sub_run_array):
            # Get 2theta and Y
            vec_x, vec_y = self._hidra_wksp.get_reduced_diffraction_data(sub_run_i, None)

            # Filter 2theta in range
            x_range = np.where((vec_x > peak_range[0]) & (vec_x < peak_range[1]))
            vec_x = vec_x[x_range]
            vec_y = vec_y[x_range]

            # Calculate COM
            if len(vec_x) == 0:
                # No data within range
                peak_center_vec[iws, 0] = np.nan
                peak_center_vec[iws, 1] = np.nan
            else:
                # There are data within rang
                peak_center_vec[iws, 0] = np.sum(vec_x * vec_y) / np.sum(vec_y)
                peak_center_vec[iws, 1] = vec_x[np.argmax(vec_y, axis=0)]
            # END-IF-ELSE
        # END-FOR

        self._peak_com_dict[peak_tag] = peak_center_vec

        return

    def convert_peaks_centers_to_d(self, peak_tag, wave_length):
        """Calculate peak positions in d-spacing after peak fitting is done

        Output: result will be saved to self._peak_center_d_vec

        Parameters
        ----------
        peak_tag : str
            Peak tag retrieve the peaks
        wave_length : float
            uniform wave length or wave length for each sub run

        Returns
        -------
        None

        """
        # Get peaks' centers
        # FIXME - this interface will be changed
        peaks = self._peak_collection_dict[peak_tag]
        sub_runs, cost_chi2, peak_params_values, peak_params_errors = peaks.get_effective_parameter_values()
        center_2theta_value_array = peak_params_values[0]
        center_2theta_error_array = peak_params_errors[0]

        # Calculate peaks' centers in dSpacing
        # d = lambda / 2 * sin(theta)
        center_d_value_array = wave_length * 0.5 / np.sin(center_2theta_value_array * 0.5 * np.pi / 180.)
        # sigma(d) = d * abs(sigma(theta) / theta)
        center_d_error_array = center_d_value_array * np.abs(center_2theta_error_array / center_d_value_array)

        return center_d_value_array, center_d_error_array

    def export_to_hydra_project(self, hydra_project_file, peak_tag):
        """Export fit result from this fitting engine instance to Hidra project file

        Parameters
        ----------
        hydra_project_file : pyrs.utilities.rs_project_file.HidraProjectFile
        peak_tag

        Returns
        -------

        """
        # Check input
        checkdatatypes.check_type('Hidra project file', hydra_project_file, rs_project_file.HidraProjectFile)

        hydra_project_file.write_peak_fit_result(self._peak_collection_dict[peak_tag])

        return

    def fit_peaks(self, peak_tag, sub_run_range, peak_function_name, background_function_name, peak_center,
                  peak_range):
        """Fit peaks with option to calculate peak center in d-spacing

        Parameters
        ----------
        peak_tag
        sub_run_range : 2-tuple
            start sub run (None as first 1) and end sub run (None as last 1) for
            range of sub runs (including both end) to refine
        peak_function_name
        background_function_name
        peak_center
        peak_range

        Returns
        -------

        """
        raise NotImplementedError('Virtual base class member method fit_peaks()')

    # TODO FIXME - peak_tag, peak_center and peak_range will be replaced by a PeakObject class (namedtuple)
    def fit_multiple_peaks(self, sub_run_range, peak_function_name, background_function_name, peak_tag_list,
                           peak_center_list, peak_range_list):
        """Fit multiple peaks on multiple sub runs

        Parameters
        ----------
        sub_run_range : 2-tuple
            start sub run (None as first 1) and end sub run (None as last 1) for
            range of sub runs (including both end) to refine
        peak_function_name : str
            name of peak profile function
        background_function_name : str
            name of background function
        peak_tag_list : list
            list of str for peak tags
        peak_center_list : list
            list of float for peak centers
        peak_range_list : list
            list of 2-tuple for each peak's range in 2-theta

        Returns
        -------
        List of ~pyrs.core.peak_collection.PeakCollection

        """
        raise NotImplementedError('Virtual base class member method fit_multiple_peaks')

    @staticmethod
    def _fit_peaks_checks(sub_run_range, peak_function_name, background_function_name, peak_center, peak_range):
        """Check parameters used to fit peaks

        Parameters
        ----------
        sub_run_range : 2-tuple
            range of sub runs including the last run specified
        peak_function_name : str
            name of peak function
        background_function_name :
            name of background function
        peak_center: float or np.ndarray
            peak centers
        peak_range:

        Returns
        -------
        None
        """
        checkdatatypes.check_tuple('Sub run numbers range', sub_run_range, 2)
        checkdatatypes.check_string_variable('Peak function name', peak_function_name,
                                             allowed_values=['Gaussian', 'Voigt', 'PseudoVoigt', 'Lorentzian'])
        checkdatatypes.check_string_variable('Background function name', background_function_name,
                                             allowed_values=['Linear', 'Flat', 'Quadratic'])
        if not isinstance(peak_center, float or np.ndarray):
            raise AssertionError('Peak center {} must be float or np.array'.format(peak_center))
        checkdatatypes.check_tuple('Peak range', peak_range, 2)

        return

    def get_calculated_peak(self, sub_run_number):
        """
        get the calculated peak's value
        :return:
        """
        raise NotImplementedError('Virtual base class member method get_calculated_peak()')

    def get_number_scans(self):
        """ Get number of scans in input data to fit
        :return:
        """
        raise NotImplementedError('Virtual base class member method get_number_scans()')

    def get_hidra_workspace(self):
        """
        Get the HidraWorkspace instance associated with this peak fit engine
        :return:
        """
        assert self._hidra_wksp is not None, 'No HidraWorkspace has been set up.'

        return self._hidra_wksp

    def get_peaks(self, peak_tag):
        return self._peak_collection_dict[peak_tag]

    @staticmethod
    def get_peak_param_names(peak_function, is_effective):
        """ Get the peak parameter names
        :param peak_function: None for default/current peak function
        :param is_effective:
        :return:
        """
        if is_effective:
            # Effective parameters
            param_names = peak_profile_utility.EFFECTIVE_PEAK_PARAMETERS[:]
            if peak_function == 'PseudoVoigt':
                param_names.append('Mixing')

        else:
            # Native parameters
            try:
                param_names = PeakShape.getShape(peak_function).native_parameters
            except KeyError as key_err:
                raise RuntimeError('Peak type {} not supported.  The supported peak functions are {}.  FYI: {}'
                                   ''.format(peak_function,
                                             PeakShape.keys(), key_err))

        return param_names

    def set_wavelength(self, wavelengths):
        """

        :param wavelengths:
        :return:
        """
        # TODO - #80 NOW - Implement
        self._wavelength_dict = wavelengths

        return
