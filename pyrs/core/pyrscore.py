# This is the core of PyRS serving as the controller of PyRS and hub for all the data
import scandataio
import datamanagers
import peakfitengine
import rshelper


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
        # TODO
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
        rshelper.check_file_name('Working directory', user_dir, check_writable=False, is_dir=True)

        self._working_dir = user_dir

        return

    def fit_peak(self, data_key, scan_index, peak_type, background_type):
        """
        fit a single peak of a measurement in a multiple-log scan
        :param data_key:
        :param scan_index:
        :param peak_type:
        :param background_type:
        :return:
        """
        # get data
        diff_data = self._data_manager.get_data_set(data_key, scan_index)

        # TODO Refactor the following
        # These shall be real
        vec_x_list = list()
        vec_y_list = list()
        for log_index in sorted(diff_data_dict.keys()):
            vec_x = diff_data_dict[log_index][0]
            vec_y = diff_data_dict[log_index][1]
            vec_x_list.append(vec_x)
            vec_y_list.append(vec_y)
        datax = np.concatenate(vec_x_list, axis=0)
        datay = np.concatenate(vec_y_list, axis=0)
        ws_full = CreateWorkspace(DataX=datax, DataY=datay, NSpec=len(vec_x_list))
        FitPeaks(InputWorkspace='ws_full', OutputWorkspace='full_fitted', PeakCenters='82', PeakFunction='Gaussian',
                 BackgroundType='Linear',
                 PositionTolerance=3, OutputPeakParametersWorkspace='param_m', FittedPeaksWorkspace='model_full',
                 FitWindowBoundaryList='79, 85')

        return

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

    def load_rs_raw(self, h5file):
        """
        load HB2B raw h5 file
        :param h5file:
        :return: str as message
        """
        diff_data_dict, sample_log_dict = self._file_io_controller.load_rs_file(h5file)

        data_key = self.data_center.add_raw_data(diff_data_dict, sample_log_dict, h5file, replace=True)
        message = 'Load {0} with reference ID {1}'.format(h5file, data_key)

        # set to current key
        self._curr_data_key = data_key

        return data_key, message
