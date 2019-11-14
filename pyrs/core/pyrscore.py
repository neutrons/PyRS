# This is the core of PyRS serving as the controller of PyRS and hub for all the data
from pyrs.utilities import checkdatatypes
from pyrs.core import instrument_geometry
from pyrs.utilities import file_util
from pyrs.core import peak_fit_factory
from pyrs.utilities.rs_project_file import HidraConstants, HydraProjectFile, HidraProjectFileMode
from pyrs.core import strain_stress_calculator
from pyrs.core import reduction_manager
from pyrs.core import polefigurecalculator
import os
import numpy

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
        # Declare services
        # reduction service
        self._reduction_service = reduction_manager.HB2BReductionManager()

        # peak fitting service
        # last/current peak fitting controller
        # None or RsPeakFitEngine/MantidPeakFitEngine instance
        self._peak_fit_controller = None  # current peak fitting controller,
        self._peak_fitting_dict = dict()  # dict of peak fitting controller. key = tag, value = peak fit controller

        # pole figure calculation
        self._pole_figure_calculator_dict = dict()
        self._last_pole_figure_calculator = None

        # strain and stress calculator
        self._ss_calculator_dict = dict()   # [session name][strain/stress type: 1/2/3] = ss calculator
        self._curr_ss_session = None
        self._curr_ss_type = None

        # Working environment
        self._working_dir = None

        return

    @property
    def peak_fitting_service(self):
        """
        return handler to current/last peak fitting controller
        :return:
        """
        return self._peak_fit_controller

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
        w_dir = self._working_dir
        if w_dir is None:
            w_dir = os.getcwd()

        return w_dir

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

    def calculate_pole_figure(self, project_name, solid_angles):
        """Calculate pole figure

        Previous API: data_key, detector_id_list

        Returns
        -------

        """
        raise NotImplementedError('Project {} of solid angles {} need to be implemented soon!'
                                  ''.format(project_name, solid_angles))

    def create_fit_engine_instance(self, fit_tag):
        # get workspace
        workspace = self.reduction_service.get_hidra_workspace(fit_tag)
        # create a controller from factory
        self._peak_fitting_dict[fit_tag] = peak_fit_factory.PeakFitEngineFactory.getInstance('Mantid')(workspace,
                                                                                                       None)
        # set wave length: TODO - #81+ - shall be a way to use calibrated or non-calibrated
        wave_length_dict = workspace.get_wave_length(calibrated=False, throw_if_not_set=False)
        if wave_length_dict is not None:
            self._peak_fitting_dict[fit_tag].set_wavelength(wave_length_dict)
            # self._peak_fit_controller.set_wavelength(wave_length_dict)

        return workspace

    def fit_peaks(self, fit_tag, sub_run_list, peak_type, background_type, peaks_fitting_setup):
        """Fit a single peak on each diffraction pattern selected from client-specified

        Note:
        - peaks_info: consider use cases for multiple non-overlapped peaks fitting

        Parameters
        ----------
        fit_tag: str
            Name of current project for loading, peak fitting and etc.
        sub_run_list: list or None
            sub runs to fit peak. None is the default for fitting all sub runs
        peak_type
        background_type
        peaks_fitting_setup: dict
            dict containing peak information [peak tag] = {Center: xx, Range: [2theta1, 2theta2]}
        Returns
        -------
        None
        """
        # Check input
        checkdatatypes.check_string_variable('Project name', fit_tag, allow_empty=False)

        # Create new one if it does not exist
        if fit_tag not in self._peak_fitting_dict:
            self.create_fit_engine_instance(fit_tag)

        # Get controller by 'fitting tag' and set it to current peak_fit_controller
        self._peak_fit_controller = self._peak_fitting_dict[fit_tag]
        workspace = self._peak_fit_controller.get_hidra_workspace()

        # FIXME - Temporarily left as reference.  Delete as soon as all tests are passed
        # # Get peak fitting controller
        # if fit_tag in self._peak_fitting_dict:
        #     # if it does exist
        #     self._peak_fit_controller = self._peak_fitting_dict[fit_tag]
        #     workspace = self._peak_fit_controller.get_hidra_workspace()
        # else:
        #     # create a new one
        #     # get workspace
        #     workspace = self.reduction_service.get_hidra_workspace(fit_tag)
        #     # create a controller from factory
        #     self._peak_fit_controller = peak_fit_factory.PeakFitEngineFactory.getInstance('Mantid')(
        #         workspace, None)
        #     # set wave length: TODO - #81+ - shall be a way to use calibrated or non-calibrated
        #     wave_length_dict = workspace.get_wave_length(calibrated=False, throw_if_not_set=False)
        #     if wave_length_dict is not None:
        #         self._peak_fit_controller.set_wavelength(wave_length_dict)
        #
        #     # add to dictionary
        #     self._peak_fitting_dict[fit_tag] = self._peak_fit_controller
        # # END-IF-ELSE

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
        print('[INFO] Fitting peak: {}'.format(peak_tags))

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

        print('[ERROR] {}'.format(error_message))

        return

    def _get_peak_fitting_controller(self, fit_tag):
        """
        get optimizer.
        raise exception if optimizer to return is None
        :param fit_tag:
        :return:
        """
        if fit_tag in self._peak_fitting_dict:
            # with data key
            optimizer = self._peak_fitting_dict[fit_tag]
            print('Return optimizer in dictionary: {0}'.format(optimizer))
        else:
            # data key exist but optimizer does not: could be not been fitted yet
            raise RuntimeError('Unable to find optimizer related to data with reference ID {0} of type {1}.'
                               'Current keys are {2}.'
                               ''.format(fit_tag, type(fit_tag), self._peak_fitting_dict.keys()))

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

    #  get_detector_ids(self, data_key) removed due to new workflow to calculate pole figure and API

    def get_diffraction_data(self, session_name, sub_run, mask):
        """ get diffraction data of a certain session/wokspace
        :param session_name: name of session for workspace
        :param sub_run: sub run of the diffraction ata
        :param mask: String as mask ID for reduced diffraction data
        :return: tuple: vec_2theta, vec_intensit
        """
        diff_data_set = self._reduction_service.get_reduced_diffraction_data(session_name, sub_run, mask)

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
        optimizer = self._get_peak_fitting_controller(session_name)
        data_set = optimizer.get_calculated_peak(sub_run)

        return data_set

    def get_peak_fitting_result(self, project_name, return_format, effective_parameter):
        """ Get peak fitting result
        Note: this provides a convenient method to retrieve information
        :param project_name:
        :param return_format:
        :param effective_parameter:
        :return:
        """
        # Get peak fitting controller
        if project_name in self._peak_fitting_dict:
            # if it does exist
            peak_fitter = self._peak_fitting_dict[project_name]
        else:
            raise RuntimeError('{} not exist'.format(project_name))

        # Param names
        param_names = peak_fitter.get_peak_param_names(None, False)

        # Get the parameter values
        if effective_parameter:
            # Retrieve effective peak parameters
            sub_run_vec, chi2_vec, param_vec = peak_fitter.get_fitted_effective_params(param_names,
                                                                                       including_error=True)
        else:
            sub_run_vec, chi2_vec, param_vec = peak_fitter.get_fitted_params(param_names, including_error=True)

        if return_format == dict:
            # The output format is a dictionary for each parameter including sub-run
            param_data_dict = dict()
            for param_index, param_name in enumerate(param_names):
                if param_name == HidraConstants.SUB_RUNS:
                    param_data_dict[param_name] = sub_run_vec
                else:
                    param_data_dict[param_name] = param_vec[param_index]
                # END-IF-ELSE
            # END-FOR

            # add sub runs and chi2
            param_data_dict[HidraConstants.SUB_RUNS] = sub_run_vec
            param_data_dict[HidraConstants.PEAK_FIT_CHI2] = chi2_vec

            # Rename for return
            param_data = param_data_dict

        elif return_format == numpy.ndarray:
            # numpy array
            # initialize
            array_shape = sub_run_vec.shape[0], param_vec.shape[0] + 2   # add 2 more columns for sub run and chi2
            param_data = numpy.ndarray(shape=array_shape, dtype='float')

            # set value: set sub runs and chi2 vector to 0th and 1st column
            param_data[:, 0] = sub_run_vec
            param_data[:, 1] = chi2_vec
            for j in range(param_vec.shape[0]):
                param_data[:, j + 1] = param_vec[j, :, 0]   # data for all sub run

        else:
            # Not supported case
            raise RuntimeError('not supported')

        # Insert sub run and chi-square as the beginning 2 parameters
        param_names.insert(0, HidraConstants.SUB_RUNS)
        param_names.insert(1, HidraConstants.PEAK_FIT_CHI2)

        return param_names, param_data

    def get_peak_center_of_mass(self, data_key):
        """
        get 'observed' center of mass of a peak
        :param data_key:
        :return:
        """
        optimizer = self._get_peak_fitting_controller(data_key)

        return optimizer.get_observed_peaks_centers()[:, 0]

    def get_peak_intensities(self, data_key_pair):
        """
        get the peak intensities
        :param data_key_pair:
        :return: a dictionary (key = detector ID) of dictionary (key = scan index, value = peak intensity)
        """
        optimizer = self._get_peak_fitting_controller(data_key_pair)
        peak_intensities = optimizer.get_peak_intensities()

        return peak_intensities

    def get_pole_figure_value(self, data_key, detector_id, log_index):
        """
        get pole figure value of a certain measurement identified by data key and log index
        :param data_key:
        :param detector_id
        :param log_index:
        :return:
        """
        assert self
        print('[ERROR] Pole figure from {} at detector ID {}/{} is not implemented'
              ''.format(data_key, detector_id, log_index))
        return None, None

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
            print('[DB...BAt] Get pole figure from detector {0}'.format(det_id))
            # get_pole_figure returned 2 tuple.  we need the second one as an array for alpha, beta, intensity
            sub_array = pole_figure_calculator.get_pole_figure_vectors(det_id, max_cost)[1]
            vec_alpha_i = sub_array[:, 0]
            vec_beta_i = sub_array[:, 1]
            vec_intensity_i = sub_array[:, 2]

            print('Det {} # data points = {}'.format(det_id, len(sub_array)))
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
            print('Updated alpha: size = {0}: {1}'.format(len(vec_alpha), vec_alpha))
        # END-FOR

        return vec_alpha, vec_beta, vec_intensity

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
        self._reduction_service.init_session(project_name)

        # Load project
        ws = self._reduction_service.load_hidra_project(hidra_h5_name, False, load_detector_counts, load_diffraction)

        return ws

    def save_diffraction_data(self, project_name, file_name):
        """ Save (reduced) diffraction data to HiDRA project file
        :param project_name: HiDRA wokspace reference or name
        :param file_name:
        :return:
        """
        self.reduction_service.save_reduced_diffraction(project_name, file_name)

        return

    def save_peak_fit_result(self, project_name, hidra_file_name, peak_tag, overwrite=True):
        """ Save the result from peak fitting to HiDRA project file
        Parameters
        ----------
        project_name: str
            name of peak fitting session
        hidra_file_name: String
            project file to export peaks fitting result to
        peak_tag : str
            peak tag
        overwrite: bool
            Flag to append to an existing file or overwrite it

        Returns
        -------

        """
        if project_name is None:
            optimizer = self._peak_fit_controller
        else:
            optimizer = self._peak_fitting_dict[project_name]

        # Determine the file IO mode
        if os.path.exists(hidra_file_name) and overwrite is False:
            # file exists and user does not want overwrite: READWRITE mode
            file_mode = HidraProjectFileMode.READWRITE
        else:
            # starting as a new file
            file_mode = HidraProjectFileMode.OVERWRITE

        # Create HiDRA project file
        hidra_project_file = HidraProjectFile(hidra_file_name, file_mode)
        # Export peaks
        optimizer.export_to_hydra_project(hidra_project_file, peak_tag)
        # Close
        hidra_project_file.close()

        return

    @staticmethod
    def save_pole_figure(data_key, detectors, file_name, file_type):
        """
        save pole figure/export pole figure
        :param data_key:
        :param detectors: a list of detector (ID)s or None (default for all detectors)
        :param file_name:
        :param file_type:
        :return:
        """
        raise NotImplementedError('{}/{}/{}/{} need to be implemented'
                                  ''.format(data_key, detectors, file_name, file_type))

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
            mask_info = self._reduction_service.load_mask_file(mask_file_name)
            mask_id = mask_info[2]
            print('L650 Mask ID = {}'.format(mask_id))
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

        self._reduction_service.reduce_diffraction_data(session_name, apply_calibration,
                                                        two_theta_step, pyrs_engine,
                                                        mask_id, sub_run_list)

        return

    @property
    def reduction_service(self):
        """
        get the reference to reduction engine
        :return:
        """
        return self._reduction_service

    def reset_strain_stress(self, is_plane_strain, is_plane_stress):
        """ reset the strain and stress calculation due to change of type

        :param is_plane_strain:
        :param is_plane_stress:
        :return:
        """
        if self._curr_ss_session is None:
            raise RuntimeError('Current session is not named.')
        elif self._curr_ss_session not in self._ss_calculator_dict:
            print('[WARNING] Current strain/stress session does not exist.')
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
        optimizer = self._get_peak_fitting_controller(data_key)

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

    def slice_data(self, even_nexus, splicer_id):
        """Event split a NeXus file

        Reference: PyVDRive
                    status, message = self.get_controller().slice_data(raw_file_name, self._currSlicerKey,
                                                           reduce_data=False,
                                                           vanadium=None,
                                                           save_chopped_nexus=False,
                                                           output_dir=os.getcwd(),
                                                           export_log_type='loadframe')

        Parameters
        ----------
        even_nexus
        splicer_id: str
            ID to locate an event slicer (splitter) which has been set up
        Returns
        -------

        """
        # TODO - Implement method
        raise NotImplementedError('Implement this ASAP')

    @property
    def supported_peak_types(self):
        """
        list of supported peaks' types for fitting
        :return:
        """
        return SUPPORTED_PEAK_TYPES[:]
