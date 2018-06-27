# This is the core of PyRS serving as the controller of PyRS and hub for all the data
import scandataio
import datamanagers
import peakfitengine
from pyrs.utilities import checkdatatypes
import numpy as np
import mantid_fit_peak
import scandataio
import polefigurecalculator

# Define Constants
SUPPORTED_PEAK_TYPES = ['Gaussian', 'Voigt', 'PseudoVoigt', 'Lorentzian']


class PyRsCore(object):
    """
    PyRS core
    """
    def __init__(self):
        """
        initialization
        """
        # declaration of class members
        self._file_io_controller = scandataio.DiffractionDataFile()  # a I/O instance for standard HB2B file
        self._data_manager = datamanagers.RawDataManager()

        # working environment
        self._working_dir = 'tests/testdata/'

        # current/default status
        self._curr_data_key = None
        self._last_optimizer = None

        # container for optimizers
        self._optimizer_dict = dict()

        # pole figure calculation
        self._pole_figure_calculator_dict = dict()
        self._last_pole_figure_calculator = None

        return

    @property
    def file_controller(self):
        """
        return handler to data loader and saver
        :return:
        """
        return self._file_io_controller

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
    def data_center(self):
        """
        return handler to data center which stores and manages all the data loaded and processed
        :return:
        """
        return self._data_manager

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
        checkdatatypes.check_file_name('Working directory', user_dir, check_writable=False, is_dir=True)

        self._working_dir = user_dir

        return

    def calculate_pole_figure(self, data_key, detector_id_list):
        """ calculate pole figure
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

            optimizer = self._get_optimizer((data_key, det_id))
            peak_intensities = optimizer.get_peak_intensities()

            # add value to pole figure calcualte
            self._last_pole_figure_calculator.add_input_data_set(det_id, peak_intensities, log_values)
        # END-FOR

        # do calculation
        self._last_pole_figure_calculator.calculate_pole_figure(detector_id_list)

        return

    def get_pole_figures(self, data_key, detector_id_list):
        """
        get the (N, 3) array for pole figures
        :param data_key:
        :param detector_id_list:
        :return:
        """
        pole_figure_calculator = self._pole_figure_calculator_dict[data_key]
        assert isinstance(pole_figure_calculator, polefigurecalculator.PoleFigureCalculator), 'Pole figure calculator ' \
                                                                                              'type mismatched'

        if detector_id_list is None:
            detector_id_list = pole_figure_calculator.get_detector_ids()

        pole_figure_array = None
        for det_id in detector_id_list:
            # get_pole_figure returned 2 tuple.  we need the second one as an array for alpha, beta, intensity
            sub_array = pole_figure_calculator.get_pole_figure(det_id)[1]
            if pole_figure_array is None:
                pole_figure_array = sub_array
            else:
                numpy.expand_array(pole_figure_array, sub_array)
        # END-FOR

        return pole_figure_array

    def get_pole_figure_value(self, data_key, detector_id, log_index):
        """
        get pole figure value of a certain measurement identified by data key and log index
        :param data_key:
        :param detector_id
        :param log_index:
        :return:
        """
        checkdatatypes.check_int_variable('Scan log #', log_index, (0, None))

        log_index_list, pole_figures = self._last_pole_figure_calculator.get_pole_figure(detector_id)
        if len(pole_figures) < log_index + 1:
            alpha = 0
            beta = 0
        else:
            try:
                alpha = pole_figures[log_index][0]
                beta = pole_figures[log_index][1]
            except ValueError as val_err:
                raise RuntimeError('Given detector {0} scan log index {1} of data IDed as {2} is out of range as '
                                   '({3}, {4})  (error = {5})'
                                   ''.format(detector_id, log_index, data_key, 0, len(pole_figures), val_err))
        # END-IF-ELSE

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

    def fit_peaks(self, data_key_set, scan_index, peak_type, background_type, fit_range):
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
        for log_index in scan_index_list:
            diff_data = self._data_manager.get_data_set(data_key_set, log_index)
            diff_data_list.append(diff_data)
        # END-FOR

        if sub_key is None:
            ref_id = '{0}'.format(data_key)
        else:
            ref_id = '{0}_{1}'.format(data_key, sub_key)
        peak_optimizer = mantid_fit_peak.MantidPeakFitEngine(diff_data_list, ref_id=ref_id)

        # observed COM and highest Y value data point
        peak_optimizer.calculate_center_of_mass()

        peak_optimizer.fit_peaks(peak_type, background_type, fit_range, None)

        self._last_optimizer = peak_optimizer
        self._optimizer_dict[data_key_set] = self._last_optimizer

        return ref_id

    def _get_optimizer(self, data_key):
        """
        get optimizer.
        raise exception if optimizer to return is None
        :param data_key:
        :return:
        """
        # check input
        if self._last_optimizer is None:
            # never been used
            return None
        elif data_key is None and self._last_optimizer is not None:
            # by default: current optimizer
            print ('Return last')
            optimizer = self._last_optimizer
        elif data_key in self._optimizer_dict:
            # with data key
            optimizer = self._optimizer_dict[data_key]
            print ('Return in dictionary: {0}'.format(optimizer))
        else:
            raise RuntimeError('Unable to find optimizer related to data with reference ID {0} of type {1}.'
                               'Current keys are {2}.'
                               ''.format(data_key, type(data_key), self._optimizer_dict.keys()))

        return optimizer

    def get_detector_ids(self, data_key):
        """
        get detector IDs for the data loaded as h5 list
        :param data_key:
        :return:
        """
        self._check_data_key(data_key)

        return self._data_manager.get_sub_keys(data_key)

    def get_fit_parameters(self, data_key=None):
        """
        get the fitted function's parameters
        :param data_key:
        :return:
        """
        # check input
        optimizer = self._get_optimizer(data_key)
        if optimizer is None:
            return None

        return optimizer.get_function_parameter_names()

    def get_peak_fit_param_value(self, data_key, param_name):
        """
        get a specific parameter's fitted value
        :param data_key:
        :param param_name:
        :return:
        """
        # check input
        optimizer = self._get_optimizer(data_key)

        return optimizer.get_fitted_params(param_name)

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

    def get_diff_data(self, data_key, scan_log_index):
        """
        get diffraction data of a certain
        :param data_key:
        :param scan_log_index:
        :return:
        """
        # get data key: by default for single data set but not for pole figure!
        if data_key is None:
            data_key = self._curr_data_key
            if data_key is None:
                raise RuntimeError('There is no current loaded data.')
        # END-IF

        # get data
        diff_data_set = self._data_manager.get_data_set(data_key, scan_log_index)

        return diff_data_set

    def get_modeled_data(self, data_key, scan_log_index):
        """
        get calculated data according to fitted model
        :param data_key:
        :param scan_log_index:
        :return:
        """
        checkdatatypes.check_int_variable('Scan index', scan_log_index, (0, None))
        # get data key
        if data_key is None:
            data_key = self._curr_data_key
            if data_key is None:
                raise RuntimeError('There is no current loaded data.')
        # END-IF

        optimizer = self._get_optimizer(data_key)
        if optimizer is None:
            data_set = None
        else:
            data_set = optimizer.get_calculated_peak(scan_log_index)

        return data_set

    def load_rs_raw(self, h5file):
        """
        load HB2B raw h5 file
        :param h5file:
        :return: str as message
        """
        diff_data_dict, sample_log_dict = self._file_io_controller.load_rs_file(h5file)

        data_key = self.data_center.add_raw_data(diff_data_dict, sample_log_dict, h5file, replace=True)
        message = 'Load {0} (Ref ID {1})'.format(h5file, data_key)

        # set to current key
        self._curr_data_key = data_key

        return data_key, message

    def load_rs_raw_set(self, det_h5_list):
        """
        Load a set of HB2B raw daa file with information of detector IDs
        :param det_h5_list:
        :return:
        """
        # read data and store in arrays managed by dictionary with detector IDs
        diff_data_dict, sample_log_dict = self._file_io_controller.load_rs_file_set(det_h5_list)
        print ('INFO: detector keys: {0}'.format(diff_data_dict.keys()))

        data_key = self.data_center.add_raw_data_set(diff_data_dict, sample_log_dict, det_h5_list, replace=True)
        message = 'Load {0} (Ref ID {1})'.format(det_h5_list, data_key)

        # set to current key
        self._curr_data_key = data_key

        return data_key, message

    def save_pole_figure(self, data_key, detector, file_name, file_type):
        """
        save pole figure/export pole figure
        :param data_key:
        :param detector:
        :param file_name:
        :param file_type:
        :return:
        """
        checkdatatypes.check_string_variable('Data key', data_key)

        if data_key in self._pole_figure_calculator_dict:
            self._pole_figure_calculator_dict[data_key].export_pole_figure(detector, file_name, file_type)
        else:
            raise RuntimeError('Data key {0} is not calculated for pole figure.  Current data keys contain {1}'
                               ''.format(data_key, self._pole_figure_calculator_dict.keys()))

        return

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
        except RuntimeError as run_err:
            raise RuntimeError('Unable to write to NeXus because Mantid fit engine is not used.\nError info: {0}'
                               ''.format(run_err))

        # save
        scandataio.save_mantid_nexus(matrix_name, file_name)

        return

    @property
    def supported_peak_types(self):
        """
        list of supported peaks' types for fitting
        :return:
        """
        return SUPPORTED_PEAK_TYPES[:]
