# This is the virtual base class as the fitting frame
import numpy
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
        """
        initialization
        :param workspace: HidraWorksapce containing the diffraction data
        :param mask_name: name of mask ID (or main/None) for reduced diffraction data
        """
        # check
        checkdatatypes.check_type('Diffraction workspace', workspace, workspaces.HidraWorkspace)

        # for scipy: keep the numpy array will be good enough
        self._hd_workspace = workspace  # hd == HiDra
        self._mask_name = mask_name

        # wave length information
        self._wavelength_dict = None

        # Fitted peak parameters
        self._peak_collection_dict = dict()   # key: peak tag, value: PeakCollection

        # for fitted result
        # self._peak_center_vec = None  # 2D vector for observed center of mass and highest data point
        # self._peak_center_d_vec = None  # 1D vector for calculated center in d-spacing

        # FIXME - These data structures are replaced by PeakCollection
        # Peak function
        # self._peak_function_name = None
        # self._background_function_name = None
        # self._fit_cost_array = None
        # self._peak_params_value_array = None
        # self._peak_params_error_array = None

        # shall be a structured numpy array
        # columns are peak and background parameters names, rows are index corresponding to sorted run numbers

        return

    def calculate_peak_position_d(self, wave_length):
        """Calculate peak positions in d-spacing

        Output: result will be saved to self._peak_center_d_vec

        Parameters
        ----------
        wave_length : float or numpy.ndarray(dtype=float)
            uniform wave length or wave length for each sub run

        Returns
        -------
        None

        """
        # TODO/FIXME - BROKEN! - Must have a better way than try and guess
        # Assumption: This is a private method working from the workspace directory
        #          OR This is a public method that will not be called within FitPeaks() with peak tag

        print(wave_length)
        print(self._peak_collection_dict.keys())

        # try:
        #     r = self.get_fitted_params(peak_tag, param_name_list=['PeakCentre'], including_error=True)
        # except KeyError:
        #     r = self.get_fitted_params(peak_tag, param_name_list=['centre'], including_error=True)
        # sub_run_vec = r[0]
        # params_vec = r[2]
        #
        # # Other parameters
        # num_sub_runs = sub_run_vec.shape[0]
        #
        # # Process wave length
        # if isinstance(wave_length, numpy.ndarray):
        #     assert wave_length.shape[0] == num_sub_runs
        #     various_wl = True
        #     wl = 0
        # else:
        #     various_wl = False
        #     wl = wave_length
        #
        # # init vector for peak center in d-spacing with error
        # self._peak_center_d_vec = numpy.ndarray((params_vec.shape[1], 2), params_vec.dtype)
        #
        # for sb_index in range(num_sub_runs):
        #     # convert to d-spacing: both fitted value and fitting error
        #     # set wave length if various to sub runs
        #     if various_wl:
        #         wl = wave_length[sb_index]
        #
        #     # calculate peak position and propagating fitting error
        #     for sub_index in range(2):
        #         peak_i_2theta_j = params_vec[0][sb_index][sub_index]
        #         if wl:
        #             peak_i_d_j = wl * 0.5 / math.sin(peak_i_2theta_j * 0.5 * math.pi / 180.)
        #         else:
        #             # case for None or zero
        #             peak_i_d_j = -1  # return a non-physical number
        #         self._peak_center_d_vec[sb_index][0] = peak_i_d_j
        # # END-FOR

        return

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

        # Get parameter values
        # sub_run_vec = self._hd_workspace.get_sub_runs()

        # hydra_project_file.write_peak_fit_result(peak_tag, self._peak_function_name, self._background_function_name,
        #                                          sub_run_vec, self._fit_cost_array, self._peak_params_value_array,
        #                                          self._peak_params_error_array)

        return

    def fit_peaks(self, peak_tag, sub_run_range, peak_function_name, background_function_name, peak_center,
                  peak_range, cal_center_d):
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
        cal_center_d

        Returns
        -------

        """
        raise NotImplementedError('Virtual base class member method fit_peaks()')

    # # TODO NEW - Implement
    # def fit_multiple_peaks(self, sub_run_range, peak_function_name, background_function_name,
    #                        peak_center_list, peak_range_list, cal_center_d):
    #     """Fit multiple peaks
    #
    #     Parameters
    #     ----------
    #     sub_run_range
    #     peak_function_name
    #     background_function_name
    #     peak_center_list
    #     peak_range_list
    #     cal_center_d
    #
    #     Returns
    #     -------
    #     List of ~pyrs.core.peak_collection.PeakCollection
    #
    #     """
    #     return

    @staticmethod
    def _fit_peaks_checks(sub_run_range, peak_function_name, background_function_name, peak_center, peak_range,
                          cal_center_d):
        """Check parameters used to fit peaks

        Parameters
        ----------
        sub_run_range : 2-tuple
            range of sub runs including the last run specified
        peak_function_name : str
            name of peak function
        background_function_name :
            name of background function
        peak_center: float or numpy ndarray
            peak centers
        peak_range:
        cal_center_d : boolean
            flag to calculate d-spacing of fitted peaks

        Returns
        -------
        None
        """
        checkdatatypes.check_tuple('Sub run numbers range', sub_run_range, 2)
        checkdatatypes.check_string_variable('Peak function name', peak_function_name,
                                             allowed_values=['Gaussian', 'Voigt', 'PseudoVoigt', 'Lorentzian'])
        checkdatatypes.check_string_variable('Background function name', background_function_name,
                                             allowed_values=['Linear', 'Flat', 'Quadratic'])
        checkdatatypes.check_bool_variable('Flag to calculate peak center in d-spacing', cal_center_d)
        if not isinstance(peak_center, float or numpy.ndarray):
            raise AssertionError('Peak center {} must be float or numpy array'.format(peak_center))
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
        assert self._hd_workspace is not None, 'No HidraWorkspace has been set up.'

        return self._hd_workspace

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
