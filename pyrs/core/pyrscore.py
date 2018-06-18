# This is the core of PyRS serving as the controller of PyRS and hub for all the data
import scandataio
import datamanagers
import peakfitengine
from pyrs.utilities import checkdatatypes
import numpy as np
import mantid_fit_peak
import scandataio


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
        checdatatypes.check_file_name('Working directory', user_dir, check_writable=False, is_dir=True)

        self._working_dir = user_dir

        return

    def calculate_pole_figure(self, data_key, peak_type, background_type, peak_range, use_mantid_engine):
        """ calculate pole figure
        :param data_key:
        :param peak_type:
        :param background_type:
        :param peak_range:
        :param use_mantid_engine:
        :return:
        """
        import polefigurecalculator

        # Check inputs
        checdatatypes.check_string_variable('Data reference ID', data_key)
        checdatatypes.check_string_variable('Peak type', peak_type)
        checdatatypes.check_string_variable('Background type', background_type)
        checdatatypes.check_bool_variable('Flag to use Mantid as fit engine', use_mantid_engine)

        # get scans
        scan_index_list = self._data_manager.get_scan_range(data_key)

        # construct data set
        pole_figure_data_dict = dict()
        for scan_index in scan_index_list:
            # get diffraction data
            reflection = dict()
            reflection['diff data'] = self._data_manager.get_data_set(data_key, scan_index)
            # get sample logs
            reflection['omega'] = self._data_manager.get_sample_log_values(data_key, 'omega')
            reflection['2theta'] = self._data_manager.get_sample_log_values(data_key, '2theta')
            reflection['chi'] = self._data_manager.get_sample_log_values(data_key, 'chi')
            reflection['phi'] = self._data_manager.get_sample_log_values(data_key, 'phi')
            # add
            pole_figure_data_dict[scan_index] = reflection
        # END-FOR

        # call pole figure calculator
        curr_pf_calculator = polefigurecalculator.PoleFigureCalculator()
        curr_pf_calculator.execute(pole_figure_data_dict, peak_type, background_type, peak_range, use_mantid_engine)

        self._pole_figure_calculator[data_key] = curr_pf_calculator

        return

    def fit_peaks(self, data_key, scan_index, peak_type, background_type, fit_range):
        """
        fit a single peak of a measurement in a multiple-log scan
        :param data_key:
        :param scan_index:
        :param peak_type:
        :param background_type:
        :param fit_range
        :return: reference ID
        """
        # Check inputs
        checdatatypes.check_string_variable('Data reference ID', data_key)
        checdatatypes.check_string_variable('Peak type', peak_type)
        checdatatypes.check_string_variable('Background type', background_type)

        # get scan indexes
        if scan_index is None:
            scan_index_list = self._data_manager.get_scan_range(data_key)
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
            diff_data = self._data_manager.get_data_set(data_key, log_index)
            diff_data_list.append(diff_data)
        # END-FOR

        ref_id = 'TODO FIND A GOOD NAMING CONVENTION'
        peak_optimizer = mantid_fit_peak.MantidPeakFitEngine(diff_data_list, ref_id=ref_id)

        # observed COM and highest Y value data point
        peak_optimizer.calculate_center_of_mass()

        peak_optimizer.fit_peaks(peak_type, background_type, fit_range, None)

        self._last_optimizer = peak_optimizer
        self._optimizer_dict[data_key] = self._last_optimizer

        return ref_id

    def _get_optimizer(self, data_key):
        """
        get optimizer.
        raise exception if optimizer to return is None
        :param data_key:
        :return:
        """
        # check input
        if data_key is None and self._last_optimizer is not None:
            # by default: current optimizer
            optimizer = self._last_optimizer
        elif data_key in self._optimizer_dict:
            # with data key
            optimizer = self._optimizer_dict[data_key]
        else:
            raise RuntimeError('Unable to find optimizer related to data with reference ID {0} of type {1}.'
                               'Or there is NO optimizer ever created.'.format(data_key, type(data_key)))

        return optimizer

    def get_fit_parameters(self, data_key=None):
        """
        get the fitted function's parameters
        :param data_key:
        :return:
        """
        # check input
        optimizer = self._get_optimizer(data_key)

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

    def get_peak_intensities(self, data_key):
        """
        get the peak intensities
        :param data_key:
        :return: a dictionary (key = scan index, value = peak intensity)
        """
        optimizer = self._get_optimizer(data_key)

        return optimizer.get_peak_intensities()

    def get_diff_data(self, data_key, scan_log_index):
        """
        get diffraction data of a certain
        :param data_key:
        :param scan_log_index:
        :return:
        """
        # get data key
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
        checdatatypes.check_int_variable('Scan index', scan_log_index, (0, None))
        # get data key
        if data_key is None:
            data_key = self._curr_data_key
            if data_key is None:
                raise RuntimeError('There is no current loaded data.')
        # END-IF

        optimizer = self._get_optimizer(data_key)
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

    def load_rs_raw_set(self, h5file_list):
        """

        :param h5file_list:
        :return:
        """
        # TODO: docs!
        diff_data_dict, sample_log_dict = self._file_io_controller.load_rs_file_set(h5file_list)

        data_key = self.data_center.add_raw_data_set(diff_data_dict, sample_log_dict, h5file_list, replace=True)
        message = 'Load {0} (Ref ID {1})'.format(h5file_list, data_key)

        # set to current key
        self._curr_data_key = data_key

        return data_key, message

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
