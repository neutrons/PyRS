from pyrs.core import reduction_manager
from pyrs.utilities import checkdatatypes
from pyrs.core import mask_util
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode
from pyrs.utilities import calibration_file_io
from matplotlib import pyplot as plt


def _mask_detectors(counts_vec, mask_file=None):
    # ignoring returns two_theta and note
    mask_vec, _, _ = mask_util.load_pyrs_mask(mask_file)
    if counts_vec.shape != mask_vec.shape:
        raise RuntimeError('Counts vector and mask vector has different shpae')

    masked_counts_vec = counts_vec * mask_vec

    return masked_counts_vec


class ReductionApp(object):
    """
    Data reduction application
    """

    def __init__(self, use_mantid_engine=False):
        """
        initialization
        """
        self._use_mantid_engine = use_mantid_engine
        self._reduction_manager = reduction_manager.HB2BReductionManager()
        self._hydra_ws = None   # HidraWorkspace used for reduction

        # initialize reduction session with a general name (single session script)
        self._session = 'GeneralHB2BReduction'
        self._hydra_file_name = None

        return

    @staticmethod
    def import_calibration_file(configuration_file):
        """ set up the geometry configuration file
        :param configuration_file:
        :return:
        """
        if configuration_file.lower().endswith('.h5'):
            # this returns a dict
            geometry_config = calibration_file_io.import_calibration_info_file(configuration_file)
        elif configuration_file.lower().endswith('.json'):
            # this returns a AnglerCameraDetectorShift
            geometry_config = calibration_file_io.read_calibration_json_file(configuration_file)[0]
        else:
            # this returns a AnglerCameraDetectorShift
            geometry_config = calibration_file_io.import_calibration_ascii_file(configuration_file)

        return geometry_config

    @property
    def use_mantid_engine(self):
        """Status to use Mantid as reduction engine to convert counts to diffraction pattern

        Returns
        -------
        bool
            True to indicate the reduction is done by Mantid algorithm and instrument geometry

        """
        return self._use_mantid_engine

    @use_mantid_engine.setter
    def use_mantid_engine(self, value):
        """Set flag to use mantid reduction engine (True) or PyRS reduction engine (False)

        Parameters
        ----------
        value

        Returns
        -------

        """
        checkdatatypes.check_bool_variable('Flag to use Mantid as reduction engine', value)

        self._use_mantid_engine = value

        return

    def get_diffraction_data(self, sub_run):
        """Get 2theta diffraction data

        Parameters
        ----------
        sub_run : int
            sub run number

        Returns
        -------
        ~numpy.ndarray, ~numpy.ndarray

        """
        vec_x, vec_y = self._reduction_manager.get_reduced_diffraction_data(self._session, sub_run)

        return vec_x, vec_y

    def get_sub_runs(self):
        return self._reduction_manager.get_sub_runs(self._session)

    def load_project_file(self, data_file):
        # init session
        self._reduction_manager.init_session(self._session)
        # load data: from raw counts to reduced data
        self._hydra_ws = self._reduction_manager.load_hidra_project(data_file, True, True, True)

        self._hydra_file_name = data_file

        return

    def load_hidra_workspace(self, hd_workspace):
        """Load a HidraWorkspace

        Parameters
        ----------
        hd_workspace : pyrs.core.workspaces.HidraWorkspace
            HidraWorkspace containing raw counts

        Returns
        -------

        """
        # set the workspace to self
        self._hydra_ws = hd_workspace
        # set workspace to reduction manager
        self._reduction_manager.init_session(self._session, self._hydra_ws)

    def reduce_data(self, sub_runs, instrument_file, calibration_file, mask, mask_id=None,
                    van_file=None, num_bins=1000):
        """Reduce data from HidraWorkspace

        Parameters
        ----------
        sub_runs : List or numpy.ndarray or None
            sub run numbers to reduce
        instrument_file
        calibration_file : str or None
            path of calibration file (optionally)
        mask : str or numpy.ndarray or None
            Mask name or mask (value) array.  None for no mask
        mask_id : str or None
            ID for mask.  If mask ID is None and if default universal mask exists, the default will be
            applied to all data
        van_file : str or None
            HiDRA project file containing vanadium counts or event NeXus file
        num_bins : int
            number of bins

        Returns
        -------

        """
        # Check inputs
        if sub_runs is None or not bool(sub_runs):  # None or empty list
            sub_runs = self._hydra_ws.get_sub_runs()

        # instrument file
        if instrument_file is not None:
            print('instrument file: {}'.format(instrument_file))
            # TODO - #84 - Implement

        # calibration file - WARNING the access to the calibration is radically different
        # depending on the value of this thing that is named like it is a bool
        geometry_calibration = False
        if calibration_file is not None:
            if calibration_file.lower().endswith('.json'):
                calib_values = calibration_file_io.read_calibration_json_file(calibration_file_name=calibration_file)
                geometry_calibration = calib_values[0]
                wave_length = calib_values[2]
                self._hydra_ws.set_wavelength(wave_length, True)
            else:
                geometry_calibration =\
                    calibration_file_io.import_calibration_ascii_file(geometry_file_name=calibration_file)
        # END-IF

        # Vanadium
        if van_file is not None:
            # vanadium file is given
            van_array, van_duration = self._reduction_manager.load_vanadium(van_file)
            if van_duration is not None:
                van_array /= van_duration
        else:
            # no vanadium
            van_array = None

        self._reduction_manager.reduce_diffraction_data(self._session,
                                                        apply_calibrated_geometry=geometry_calibration,
                                                        num_bins=num_bins,
                                                        use_pyrs_engine=not self._use_mantid_engine,
                                                        sub_run_list=sub_runs,
                                                        mask=mask,
                                                        mask_id=mask_id,
                                                        vanadium_counts=van_array)

    def plot_reduced_data(self, sub_run_number=None):

        if sub_run_number is None:
            sub_runs = self._reduction_manager.get_sub_runs(self._session)
        else:
            sub_runs = [sub_run_number]

        for sub_run_i in sub_runs:
            vec_x, vec_y = self._reduction_manager.get_reduced_diffraction_data(self._session, sub_run_i)
            plt.plot(vec_x, vec_y)
        plt.show()

    def save_diffraction_data(self, output_file_name=None, append_mode=False):
        """Save reduced diffraction data to Hidra project file
        Parameters
        ----------
        output_file_name: None or str
            if None, then append result to the input file
        append_mode : bool
            flag to force project file in appending/READWRITE mode
        Returns
        -------

        """
        # Determine output file name and writing mode
        if output_file_name is None or output_file_name == self._hydra_file_name:
            file_name = self._hydra_file_name
            mode = HidraProjectFileMode.READWRITE
        elif append_mode:
            # Append to existing file
            file_name = output_file_name
            mode = HidraProjectFileMode.READWRITE
        else:
            file_name = output_file_name
            mode = HidraProjectFileMode.OVERWRITE

        # Sanity check
        if file_name is None:
            raise RuntimeError('Output file name is not set property.  There is no default file name'
                               'or user specified output file name.')

        # Generate project file instance
        out_file = HidraProjectFile(file_name, mode)

        # If it is a new file, the sample logs and other information shall be exported too
        if mode == HidraProjectFileMode.OVERWRITE:
            self._hydra_ws.save_experimental_data(out_file, ignore_raw_counts=True)

        # Calibrated wave length shall be written
        self._hydra_ws.save_wavelength(out_file)

        # Write & close
        self._hydra_ws.save_reduced_diffraction_data(out_file)
