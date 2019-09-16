# This is the core of PyRS serving as the controller of PyRS and hub for all the data
from pyrs.utilities import checkdatatypes
from pyrs.core import instrument_geometry
from pyrs.utilities import file_util
from pyrs.core import peak_fit_factory
import mantid_fit_peak
import strain_stress_calculator
import reduction_manager
import polefigurecalculator
import os
import numpy
from pandas import DataFrame

# Define Constants
SUPPORTED_PEAK_TYPES = ['PseudoVoigt', 'Gaussian', 'Voigt']  # 'Lorentzian': No a profile of HB2B


class PyRsCore(object):
    """
    PyRS core
    """
    def __init__(self):
        """
        initialization
        """
        # declaration of class members
        self._reduction_manager = reduction_manager.HB2BReductionManager()

        # working environment
        if os.path.exists('tests/testdata/'):
            self._working_dir = 'tests/testdata/'
        else:
            self._working_dir = os.getcwd()

        # These are for peak fitting and etc.
        # TODO - AFTER #72 - Better to refactor!
        self._last_optimizer = None  # None or RsPeakFitEngine/MantidPeakFitEngine instance

        # container for optimizers
        self._peak_fit_controller = None
        self._optimizer_dict = dict()

        # pole figure calculation
        self._pole_figure_calculator_dict = dict()
        self._last_pole_figure_calculator = None

        # strain and stress calculator
        self._ss_calculator_dict = dict()   # [session name][strain/stress type: 1/2/3] = ss calculator
        self._curr_ss_session = None
        self._curr_ss_type = None

        return

    def new_strain_stress_session(self, session_name, is_plane_stress, is_plane_strain):
        """ Create a new strain/stress session by initializing a new StrainStressCalculator instance
        :param session_name: name of strain/stress session to query
        :param is_plane_stress: flag for being plane stress (specific equation)
        :param is_plane_strain:
        :return:
        """
        ss_type = self._get_strain_stress_type_key(is_plane_strain, is_plane_stress)
        new_ss_calculator = strain_stress_calculator.StrainStressCalculator(session_name, is_plane_strain,
                                                                            is_plane_stress)

        self._ss_calculator_dict[session_name] = dict()
        self._ss_calculator_dict[session_name][ss_type] = new_ss_calculator
        self._curr_ss_session = session_name
        self._curr_ss_type = ss_type

        return

    @property
    def peak_fitting_controller(self):
        """
        return handler to peak fitting manager
        :return:
        """
        return self._peak_fitting_controller

    @property
    def current_data_reference_id(self):
        """
        get the current/latest data reference ID
        :return:
        """
        return self._curr_data_key

    @property
    def strain_stress_calculator(self):
        """
        return the handler to strain/stress calculator
        :return:
        """
        if self._curr_ss_session is None:
            return None

        return self._ss_calculator_dict[self._curr_ss_session][self._curr_ss_type]

    @property
    def working_dir(self):
        """
        get working directory
        :return:
        """
        return self._working_dir

    @working_dir.setter
    def working_dir(self, user_dir):
        """
        set working directory
        :param user_dir:
        :return:
        """
        checkdatatypes.check_file_name(user_dir, check_writable=False, is_dir=True)

        self._working_dir = user_dir

        return

    # TODO FIXME - NOW - Broken
    def calculate_pole_figure(self, data_key, detector_id_list):
        """ API method to calculate pole figure by a specified data key
        :param data_key:
        :param detector_id_list:
        :return:
        """
        # check input
        checkdatatypes.check_string_variable('Data key/ID', data_key)
        if detector_id_list is None:
            detector_id_list = self.get_detector_ids(data_key)
        else:
            checkdatatypes.check_list('Detector IDs', detector_id_list)

        # get peak intensities from fitting
        # peak_intensities = self.get_peak_intensities(data_key, detector_id_list)

        # initialize pole figure
        self._last_pole_figure_calculator = polefigurecalculator.PoleFigureCalculator()
        self._pole_figure_calculator_dict[data_key] = self._last_pole_figure_calculator

        # set up pole figure logs and get it
        log_names = [('2theta', '2theta'),
                     ('omega', 'omega'),
                     ('chi', 'chi'),
                     ('phi', 'phi')]

        for det_id in detector_id_list:
            # get intensity and log value
            log_values = self.data_center.get_scan_index_logs_values((data_key, det_id), log_names)

            try:
                optimizer = self._get_optimizer((data_key, det_id))
                peak_intensities = optimizer.get_peak_intensities()
            except AttributeError as att:
                raise RuntimeError('Unable to get peak intensities. Check whether peaks have been fit.  FYI: {}'
                                   ''.format(att))
            fit_info_dict = optimizer.get_peak_fit_parameters()

            # add value to pole figure calculate
            self._last_pole_figure_calculator.add_input_data_set(det_id, peak_intensities, fit_info_dict, log_values)
        # END-FOR

        # do calculation
        self._last_pole_figure_calculator.calculate_pole_figure(detector_id_list)

        return

    def get_pole_figure_values(self, data_key, detector_id_list, max_cost):
        """ API method to get the (N, 3) array for pole figures
        :param data_key:
        :param detector_id_list:
        :param max_cost:
        :return:
        """
        pole_figure_calculator = self._pole_figure_calculator_dict[data_key]
        assert isinstance(pole_figure_calculator, polefigurecalculator.PoleFigureCalculator),\
            'Pole figure calculator type mismatched. Input is of type {0} but expected as {1}.' \
            ''.format(type(pole_figure_calculator), 'polefigurecalculator.PoleFigureCalculato')

        if detector_id_list is None:
            detector_id_list = pole_figure_calculator.get_detector_ids()
        else:
            checkdatatypes.check_list('Detector ID list', detector_id_list)

        # get all the pole figure vectors
        vec_alpha = None
        vec_beta = None
        vec_intensity = None
        for det_id in detector_id_list:
            print ('[DB...BAt] Get pole figure from detector {0}'.format(det_id))
            # get_pole_figure returned 2 tuple.  we need the second one as an array for alpha, beta, intensity
            sub_array = pole_figure_calculator.get_pole_figure_vectors(det_id, max_cost)[1]
            vec_alpha_i = sub_array[:, 0]
            vec_beta_i = sub_array[:, 1]
            vec_intensity_i = sub_array[:, 2]

            print ('Det {} # data points = {}'.format(det_id, len(sub_array)))
            # print ('alpha: {0}'.format(vec_alpha_i))

            if vec_alpha is None:
                vec_alpha = vec_alpha_i
                vec_beta = vec_beta_i
                vec_intensity = vec_intensity_i
            else:
                vec_alpha = numpy.concatenate((vec_alpha, vec_alpha_i), axis=0)
                vec_beta = numpy.concatenate((vec_beta, vec_beta_i), axis=0)
                vec_intensity = numpy.concatenate((vec_intensity, vec_intensity_i), axis=0)
            # END-IF-ELSE
            print ('Updated alpha: size = {0}: {1}'.format(len(vec_alpha), vec_alpha))
        # END-FOR

        return vec_alpha, vec_beta, vec_intensity

    def get_pole_figure_value(self, data_key, detector_id, log_index):
        """
        get pole figure value of a certain measurement identified by data key and log index
        :param data_key:
        :param detector_id
        :param log_index:
        :return:
        """
        checkdatatypes.check_int_variable('Scan log #', log_index, (0, None))

        alpha, beta = self._last_pole_figure_calculator.get_pole_figure_1_pt(detector_id, log_index)

        # log_index_list, pole_figures = self._last_pole_figure_calculator.get_pole_figure_vectors(detector_id, max_cost=None)
        # if len(pole_figures) < log_index + 1:
        #     alpha = 0
        #     beta = 0
        # else:
        #     try:
        #         alpha = pole_figures[log_index][0]
        #         beta = pole_figures[log_index][1]
        #     except ValueError as val_err:
        #         raise RuntimeError('Given detector {0} scan log index {1} of data IDed as {2} is out of range as '
        #                            '({3}, {4})  (error = {5})'
        #                            ''.format(detector_id, log_index, data_key, 0, len(pole_figures), val_err))
        # # END-IF-ELSE

        return alpha, beta

    @staticmethod
    def _check_data_key(data_key_set):
        if isinstance(data_key_set, tuple) and len(data_key_set) == 2:
            data_key, sub_key = data_key_set
        elif isinstance(data_key_set, tuple):
            raise RuntimeError('Wrong!')
        else:
            data_key = data_key_set
            checkdatatypes.check_string_variable('Data reference ID', data_key)
            sub_key = None

        return data_key, sub_key

    def fit_peaks(self, project_name, sub_run_list, peak_type, background_type, peaks_fitting_setup):
        """
        Fit a single peak on each diffraction pattern selected from client-specified

        Note:
        - peaks_info: consider use cases for multiple non-overlapped peaks fitting

        :param project_name: Name of current project for loading, peak fitting and etc.
        :param sub_run_list: None as the default as All;
        :param peak_type:
        :param background_type:
        :param peaks_fitting_setup: dict containing peak information [peak tag] = {Center: xx, Range: [2theta1, 2theta2]}
        :return:
        """
        # Check input
        checkdatatypes.check_string_variable('Project name', project_name, allow_empty=False)

        # Get peak fitting controller
        if project_name in self._optimizer_dict:
            # if it does exist
            self._peak_fit_controller = self._optimizer_dict[project_name]
            workspace = self._peak_fit_controller.get_hidra_workspace()  # TODO FIXME - #81 NOW - No such method
        else:
            # create a new one
            # get workspace
            workspace = self.reduction_manager.get_hidra_workspace(project_name)
            # create a controller from factory
            self._peak_fit_controller = peak_fit_factory.PeakFitEngineFactory.getInstance('Mantid')(
                workspace, None)
            # set wave length: TODO - #81+ - shall be a way to use calibrated or non-calibrated
            wave_length_dict = workspace.get_wave_length(calibrated=False, throw_if_not_set=True)
            self._peak_fit_controller.set_wavelength(wave_length_dict)

            # add to dictionary
            self._optimizer_dict[project_name] = self._peak_fit_controller
        # END-IF-ELSE

        # Check Inputs
        checkdatatypes.check_dict('Peak fitting (information) parameters', peaks_fitting_setup)
        checkdatatypes.check_string_variable('Peak type', peak_type, peak_fit_factory.SupportedPeakProfiles)
        checkdatatypes.check_string_variable('Background type', background_type,
                                             peak_fit_factory.SupportedBackgroundTypes)

        # Deal with sub runs
        if sub_run_list is None:
            sub_run_list = workspace.get_sub_runs()
        else:
            checkdatatypes.check_list('Sub run numbers', sub_run_list)
        sub_run_range = sub_run_list[0], sub_run_list[-1]

        # Fit peaks
        peak_tags = sorted(peaks_fitting_setup.keys())
        print ('[INFO] Fitting peak: {}'.format(peak_tags))

        error_message = ''
        for peak_tag_i in peak_tags:
            # get fit setup parameters
            try:
                peak_center = peaks_fitting_setup[peak_tag_i]['Center']
                peak_range = peaks_fitting_setup[peak_tag_i]['Range']
            except KeyError as key_err:
                raise KeyError('Peak fitting parameter info-dict for peak (tag) {} must have keys as '
                               'Center and Range but not {}.  FYI: {}'
                               ''.format(peak_tag_i, peaks_fitting_setup[peak_tag_i].keys(), key_err))

            # fit peak
            try:
                self._peak_fit_controller.fit_peaks(sub_run_range, peak_type, background_type, peak_center, peak_range,
                                                    cal_center_d=True)
            except RuntimeError as run_err:
                error_message += 'Failed to fit (tag) {} due to {}\n'.format(peak_tag_i, run_err)
        # END-FOR

        print ('[ERROR] {}'.format(error_message))

        return

    # TODO FIXME - NOW - Broken now
    def fit_peaks_old(self, data_key_set, scan_index, peak_type, background_type, fit_range):
        """
        fit a single peak of a measurement in a multiple-log scan
        :param data_key_set:
        :param scan_index:
        :param peak_type:
        :param background_type:
        :param fit_range
        :return: reference ID
        """
        # Check inputs
        if isinstance(data_key_set, tuple) and len(data_key_set) == 2:
            data_key, sub_key = data_key_set
        elif isinstance(data_key_set, tuple):
            raise RuntimeError('Wrong!')
        else:
            data_key = data_key_set
            checkdatatypes.check_string_variable('Data reference ID', data_key)
            sub_key = None

        checkdatatypes.check_string_variable('Peak type', peak_type)
        checkdatatypes.check_string_variable('Background type', background_type)

        # get scan indexes
        if scan_index is None:
            scan_index_list = self._data_manager.get_scan_range(data_key, sub_key)
        elif isinstance(scan_index, int):
            # check range
            scan_index_list = [scan_index]
        elif isinstance(scan_index, list):
            scan_index_list = scan_index
        else:
            raise RuntimeError('Scan index ({0}) is not supported.'.format(scan_index))

        # get data
        diff_data_list = list()
        # TODO - FUTURE - sun run will be used to replace log_index
        for log_index in scan_index_list:
            diff_data = self._data_manager.get_diffraction_intensity_vector(data_key_set, log_index)
            diff_data_list.append(diff_data)
        # END-FOR

        if sub_key is None:
            ref_id = '{0}'.format(data_key)
        else:
            ref_id = '{0}_{1}'.format(data_key, sub_key)
        peak_optimizer = mantid_fit_peak.MantidPeakFitEngine(scan_index_list, diff_data_list, ref_id=ref_id)

        # observed COM and highest Y value data point
        print ('[DB...BAT] Fit Engine w/ Key: {0}'.format(data_key_set))
        peak_optimizer.calculate_center_of_mass()
        # fit peaks
        peak_optimizer.fit_peaks(peak_type, background_type, fit_range, None)

        # convert peak center to d-spacing
        if sub_key is None:
            wave_length_vec = self.data_center.get_sample_log_values(data_key, sample_log_name='Wavelength')
        else:
            # texture analysis case
            wave_length_vec = self.data_center.get_sample_log_values((data_key, sub_key), sample_log_name='Wavelength')
        peak_optimizer.calculate_peak_position_d(wave_length_vec=wave_length_vec)

        self._last_optimizer = peak_optimizer
        self._optimizer_dict[data_key_set] = self._last_optimizer

        return ref_id

    def _get_optimizer(self, fit_session_name):
        """
        get optimizer.
        raise exception if optimizer to return is None
        :param fit_session_name:
        :return:
        """
        if fit_session_name in self._optimizer_dict:
            # with data key
            optimizer = self._optimizer_dict[fit_session_name]
            print ('Return optimizer in dictionary: {0}'.format(optimizer))
        else:
            # data key exist but optimizer does not: could be not been fitted yet
            raise RuntimeError('Unable to find optimizer related to data with reference ID {0} of type {1}.'
                               'Current keys are {2}.'
                               ''.format(fit_session_name, type(fit_session_name), self._optimizer_dict.keys()))

        return optimizer

    @staticmethod
    def _get_strain_stress_type_key(is_plane_strain, is_plane_stress):
        """
        blabla
        :param is_plane_strain:
        :param is_plane_stress:
        :return: 1: regular unconstrained, 2: plane strain, 3: plane stress
        """
        if is_plane_strain:
            return 2
        elif is_plane_stress:
            return 3

        return 1

    def get_detector_ids(self, data_key):
        """
        get detector IDs for the data loaded as h5 list
        :param data_key:
        :return:
        """
        self._check_data_key(data_key)

        return self._data_manager.get_sub_keys(data_key)

    def get_diffraction_data(self, session_name, sub_run, mask):
        """ get diffraction data of a certain session/wokspace
        :param session_name: name of session for workspace
        :param sub_run: sub run of the diffraction ata
        :param mask: String as mask ID for reduced diffraction data
        :return: tuple: vec_2theta, vec_intensit
        """
        diff_data_set = self._reduction_manager.get_reduced_diffraction_data(session_name, sub_run, mask)

        return diff_data_set

    def get_modeled_data(self, session_name, sub_run):
        """ Get calculated data according to fitted model
        :param session_name:
        :param sub_run:
        :return:
        """
        # Check input
        checkdatatypes.check_int_variable('Sub run numbers', sub_run, (0, None))
        checkdatatypes.check_string_variable('Project/session name', session_name, allow_empty=False)

        # get data key
        optimizer = self._get_optimizer(session_name)
        data_set = optimizer.get_calculated_peak(sub_run)

        return data_set

    # TODO - #81 - Code quality
    def get_peak_fitting_result(self, project_name, return_format, effective_parameter):
        """ Get peak fitting result
        Note: this provides a convenient method to retrieve information
        :param project_name:
        :param return_format:
        :param effective_parameter:
        :return:
        """
        from pyrs.utilities.rs_project_file import HidraConstants

        # Get peak fitting controller
        if project_name in self._optimizer_dict:
            # if it does exist
            peak_fitter = self._optimizer_dict[project_name]
        else:
            raise RuntimeError('{} not exist'.format(project_name))

        # Param names
        param_names = peak_fitter.get_peak_param_names(None, False)

        # Get the parameter values
        if effective_parameter:
            # TODO - #84 ASAAP - Impelment it!
            raise NotImplementedError('Effective parameters... ASAP')
            peak_fitter.get_fitted_effective_params(param_names)
        else:
            sub_run_vec, chi2_vec, param_vec = peak_fitter.get_fitted_params(param_names, including_error=True)

        if return_format == dict:
            # The output format is a dictionary for each parameter including sub-run
            param_data_dict = dict()
            for para_name, param_index in enumerate(param_names):
                if param_name == HidraConstants.SUB_RUNS:
                    param_data_dict[para_name] = sub_run_vec
                else:
                    param_data_dict[para_name] = param_vec[param_index]

        elif return_format == numpy.ndarray:
            # numpy array
            # initialize
            array_shape = sub_run_vec.shape[0], 2+param_vec.shape[0]
            param_data = numpy.ndarray(shape=array_shape, dtype='float')

            # set value
            param_data[:, 0] = sub_run_vec
            param_data[:, 1] = chi2_vec
            for j in range(param_vec.shape[0]):
                param_data[:, j+1] = param_vec[j, :, 0]   # data for all sub run

        elif return_format == DataFrame:
            # pandas data frame
            # TODO - #84+ - Implement pandas DataFrame ASAP
            raise NotImplementedError('ASAP')

        else:
            # Not supported case
            raise RuntimeError('not supported')

        # Insert sub run and chi-square as the beginning 2 parameters
        param_names.insert(0, HidraConstants.SUB_RUNS)
        param_names.insert(1, HidraConstants.PEAK_FIT_CHI2)

        return param_names, param_data

    # TODO - TONIGHT NOW - Need to migrate to new get_fitted_params
    def get_peak_fit_param_value(self, data_key, param_name_list, max_cost):
        """
        get a specific parameter's fitted value
        :param data_key:
        :param param_name:
        :param max_cost: if not None, then filter out the bad (with large cost) fitting
        :return: 3-tuple
        """
        # check input
        # TODO - #80 - Again data_key shall be replaced by ...
        optimizer = self._get_optimizer(None)

        if max_cost is None:
            sub_run_vec, chi2_vec, param_vec = optimizer.get_fitted_params(param_name_list, False)
        else:
            sub_run_vec, chi2_vec, param_vec = optimizer.get_good_fitted_params(param_name_list, max_cost)

        return sub_run_vec, chi2_vec, param_vec

    def get_peak_fit_param_value_error(self, data_key, param_name, max_cost):
        """ Get a specific peak parameter's fitting value with error
        :param data_key:
        :param param_name:
        :param max_cost:
        :return: 2-tuple: (1) (n, ) for sub runs (2) array as (n, 2) such that [i, 0] is value and [i, 1] is error
        """
        # check input
        fit_engine = self._get_optimizer(data_key)

        if max_cost is None:
            sub_run_vec, param_error_vec = fit_engine.get_fitted_params(param_name,  inlcuding_error=True)
        else:
            sub_run_vec, param_error_vec = fit_engine.get_good_fitted_params(param_name, max_cost, inlcuding_error=True)

        return sub_run_vec, param_error_vec

    # TODO FIXME - TONIGHT NOW - This method shall be migrated to newer API to get parameter values
    def get_peak_fit_params_in_dict(self, data_key):
        """
        export the complete set of peak parameters fitted to a dictionary
        :param data_key:
        :return:
        """
        # check input
        optimizer = self._get_optimizer(None)  # TODO FIXME - #80 - Need a new mechanism for data key!

        # get names, detector IDs (for check) & log indexes
        param_names = optimizer.get_function_parameter_names()
        if False:
            # TODO FIXME - #80 - Need to think of how to deal with mask (aka detector ID)
            detector_ids = self.get_detector_ids(data_key)
            print ('[DB...BAT] Detector IDs = {}'.format(detector_ids))
            if detector_ids is not None:
                raise NotImplementedError('Multiple-detector ID case has not been considered yet. '
                                          'Contact developer for this issue.')

        # Get peak parameters value
        sub_run_vec, chi2_vec, params_values_matrix = self.get_peak_fit_param_value(data_key, param_names,
                                                                                    max_cost=None)

        # init dictionary
        fit_param_value_dict = dict()

        # set: this shall really good to be a pandas DataFrame or labelled numpy matrix  TODO FIXME #80+
        for index in range(len(sub_run_vec)):
            # get sub runs and chi2
            sub_run_i = sub_run_vec[index]
            chi2_i = chi2_vec[index]
            fit_param_value_dict[sub_run_i] = {'cost': chi2_i}

            for p_index in range(len(param_names)):
                param_name = param_names[p_index]
                fit_param_value_dict[sub_run_i][param_name] = params_values_matrix[p_index, index, 0]
            # END-FOR
        # END-FOR

        return fit_param_value_dict

    def get_peak_fit_scan_log_indexes(self, data_key):
        """
        get the scan log indexes from an optimizer
        :param data_key:
        :return: list of integers
        """
        # check input
        optimizer = self._get_optimizer(data_key)

        return optimizer.get_scan_indexes()

    def get_peak_center_of_mass(self, data_key):
        """
        get 'observed' center of mass of a peak
        :param data_key:
        :return:
        """
        optimizer = self._get_optimizer(data_key)

        return optimizer.get_observed_peaks_centers()[:, 0]

    def get_peak_intensities(self, data_key_pair):
        """
        get the peak intensities
        :param data_key_pair:
        :return: a dictionary (key = detector ID) of dictionary (key = scan index, value = peak intensity)
        """
        optimizer = self._get_optimizer(data_key_pair)
        peak_intensities = optimizer.get_peak_intensities()

        return peak_intensities

    def load_hidra_project(self, hidra_h5_name, project_name, load_detector_counts=True, load_diffraction=False):
        """
        Load a HIDRA project file
        :param hidra_h5_name: name of HIDRA project file in HDF5 format
        :param project_name: name of the reduction project specified by user to trace
        :param load_detector_counts:
        :param load_diffraction:
        :return: HidraWorkspace instance
        """
        # Initialize session
        self._reduction_manager.init_session(project_name)

        # Load project
        ws = self._reduction_manager.load_hidra_project(hidra_h5_name, False, load_detector_counts, load_diffraction)

        return ws

    def save_diffraction_data(self, project_name, file_name):
        """

        :param project_name:
        :param file_name:
        :return:
        """
        self.reduction_manager.save_reduced_diffraction(project_name, file_name)

        return

    def save_peak_fit_result(self, project_name, hidra_file_name, peak_tag):
        """ Save peak fit result to file with original data
        :param data_key:
        :param src_rs_file_name:
        :param target_rs_file_name:
        :return:
        """
        # TODO - #81 - Doc!
        if project_name is None:
            optimizer = self._peak_fit_controller
        else:
            optimizer = self._optimizer_dict[project_name]

        optimizer.export_fit_result(hidra_file_name, peak_tag)

        return

    def save_pole_figure(self, data_key, detectors, file_name, file_type):
        """
        save pole figure/export pole figure
        :param data_key:
        :param detectors: a list of detector (ID)s or None (default for all detectors)
        :param file_name:
        :param file_type:
        :return:
        """
        checkdatatypes.check_string_variable('Data key', data_key)

        if data_key in self._pole_figure_calculator_dict:
            self._pole_figure_calculator_dict[data_key].export_pole_figure(detectors, file_name, file_type)
        else:
            raise RuntimeError('Data key {0} is not calculated for pole figure.  Current data keys contain {1}'
                               ''.format(data_key, self._pole_figure_calculator_dict.keys()))

        return

    def reduce_diffraction_data(self, session_name, two_theta_step, pyrs_engine, mask_file_name=None,
                                geometry_calibration=None, sub_run_list=None):
        """ Reduce all sub runs in a workspace from detector counts to diffraction data
        :param session_name:
        :param two_theta_step:
        :param pyrs_engine:
        :param mask_file_name:
        :param geometry_calibration: True/file name/AnglerCameraDetectorShift/None), False, None/(False, )
        :param sub_run_list: list of sub run numbers or None (for all)
        :return:
        """
        # Mask file
        if mask_file_name:
            mask_info = self._reduction_manager.load_mask_file(mask_file_name)
            mask_id = mask_info[2]
            print ('L650 Mask ID = {}'.format(mask_id))
        else:
            mask_id = None

        # Geometry calibration
        if geometry_calibration is None or geometry_calibration is False:
            # No apply
            apply_calibration = False
        elif isinstance(geometry_calibration, str):
            # From a Json file
            calib_shift = instrument_geometry.AnglerCameraDetectorShift(0, 0, 0, 0, 0, 0)
            calib_shift.from_json(geometry_calibration)
            apply_calibration = calib_shift
        elif isinstance(geometry_calibration, instrument_geometry.AnglerCameraDetectorShift):
            # Already a AnglerCameraDetectorShift instance
            apply_calibration = geometry_calibration
        elif geometry_calibration is True:
            # Use what is loaded from file or set to workspace before
            apply_calibration = True
        else:
            raise RuntimeError('Argument geometry_calibration of value {} and type {} is not supported'
                               ''.format(geometry_calibration, type(geometry_calibration)))
        # END-IF-ELSE

        self._reduction_manager.reduce_diffraction_data(session_name, apply_calibration,
                                                        two_theta_step, pyrs_engine,
                                                        mask_id, sub_run_list)

        return

    @property
    def reduction_manager(self):
        """
        get the reference to reduction engine
        :return:
        """
        return self._reduction_manager

    def reset_strain_stress(self, is_plane_strain, is_plane_stress):
        """ reset the strain and stress calculation due to change of type

        :param is_plane_strain:
        :param is_plane_stress:
        :return:
        """
        if self._curr_ss_session is None:
            raise RuntimeError('Current session is not named.')
        elif self._curr_ss_session not in self._ss_calculator_dict:
            print ('[WARNING] Current strain/stress session does not exist.')
            return

        ss_type_index = self._get_strain_stress_type_key(is_plane_strain, is_plane_stress)
        if ss_type_index == self._curr_ss_type:
            raise RuntimeError('Same strain/stress type (plane strain = {}, plane stress = {}'
                               ''.format(is_plane_strain, is_plane_stress))

        # rename the current strain stress name
        # saved_ss_name = self._curr_ss_session + '_{}_{}'.format(is_plane_strain, is_plane_stress)
        # self._ss_calculator_dict[self._curr_ss_session].rename(saved_ss_name)
        # prev_calculator = self._ss_calculator_dict[self._curr_ss_session]
        # self._ss_calculator_dict[saved_ss_name] = prev_calculator

        # reset new strain/stress calculator
        new_ss_calculator = self.strain_stress_calculator.migrate(is_plane_strain, is_plane_stress)

        self._ss_calculator_dict[self._curr_ss_session][ss_type_index] = new_ss_calculator
        self._curr_ss_type = ss_type_index

        return self._curr_ss_session

    def save_nexus(self, data_key, file_name):
        """
        save data in a MatrixWorkspace to Mantid processed NeXus file
        :param data_key:
        :param file_name:
        :return:
        """
        # Check
        optimizer = self._get_optimizer(data_key)

        # get the workspace name
        try:
            matrix_name = optimizer.get_data_workspace_name()
            # save
            file_util.save_mantid_nexus(matrix_name, file_name)
        except RuntimeError as run_err:
            raise RuntimeError('Unable to write to NeXus because Mantid fit engine is not used.\nError info: {0}'
                               ''.format(run_err))

        try:
            matrix_name = optimizer.get_center_of_mass_workspace_name()
            # save
            dir_name = os.path.dirname(file_name)
            base_name = os.path.basename(file_name)
            file_name = os.path.join(dir_name, base_name.split('.')[0] + '_com.nxs')
            file_util.save_mantid_nexus(matrix_name, file_name)
        except RuntimeError as run_err:
            raise RuntimeError('Unable to write COM to NeXus because Mantid fit engine is not used.\nError info: {0}'
                               ''.format(run_err))

        return

    @property
    def supported_peak_types(self):
        """
        list of supported peaks' types for fitting
        :return:
        """
        return SUPPORTED_PEAK_TYPES[:]
