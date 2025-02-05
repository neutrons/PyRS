from pyrs.core import reduction_manager
from pyrs.core import mask_util
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore
from pyrs.utilities import calibration_file_io
from matplotlib import pyplot as plt


def _mask_detectors(counts_vec, mask_file=None):
    # ignoring returns two_theta and note
    mask_vec, _, _ = mask_util.load_pyrs_mask(mask_file)
    if counts_vec.shape != mask_vec.shape:
        raise RuntimeError('Counts vector and mask vector has different shpae')

    masked_counts_vec = counts_vec * mask_vec

    return masked_counts_vec


class ReductionApp:
    """
    Data reduction application
    """

    def __init__(self):
        """
        initialization
        """
        self._reduction_manager = reduction_manager.HB2BReductionManager()
        self._hydra_ws = None   # HidraWorkspace used for reduction

        # initialize reduction session with a general name (single session script)
        self._session = 'GeneralHB2BReduction'
        self._hydra_file_name = None
        self._sub_runs = None

        return

    def get_diffraction_data(self, sub_run, mask_id=None):
        """Get 2theta diffraction data

        Parameters
        ----------
        sub_run : int
            sub run number

        Returns
        -------
        ~numpy.ndarray, ~numpy.ndarray

        """
        vec_x, vec_y, vec_error = self._reduction_manager.get_reduced_diffraction_data(self._session, sub_run, mask_id)

        return vec_x, vec_y, vec_error

    def get_raw_counts(self, sub_run, mask_id=None):
        """Get raw diffraction data

        Parameters
        ----------
        sub_run : int
            sub run number

        Returns
        -------
        ~numpy.ndarray, ~numpy.ndarray

        """

        return self._hydra_ws._raw_counts[sub_run]

    def get_sub_runs(self):
        return self._reduction_manager.get_sub_runs(self._session)

    def get_reduced_sub_runs(self):
        return self._sub_runs

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
                    van_file=None, num_bins=1000, eta_step=None, eta_min=-8.2, eta_max=8.2,
                    min_2theta=None, max_2theta=None, delta_2theta=None):
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
        eta_step : float
            angular step size for out-of-plane reduction
        eta_min : float
            min angle for out-of-plane reduction
        eta_max : float
            max angle for out-of-plane reduction
        min_2theta : float or None
            min 2theta
        max_2theta : float or None
            max 2theta
        delta_2theta : float or None
            2theta increment in the reduced diffraction data

        Returns
        -------

        """
        # Check inputs
        if (sub_runs is None) or (len(sub_runs) == 0):  # None or empty list
            self._sub_runs = self._hydra_ws.get_sub_runs()
        else:
            # sort array to make sure the sub-run data are written into project files in increasing order
            self._sub_runs = sorted(sub_runs)

        # instrument file
        if instrument_file is not None:
            print('instrument file: {}'.format(instrument_file))
            # TODO - #84 - Implement

        # calibration file - WARNING the access to the calibration is radically different
        # depending on the value of this thing that is named like it is a bool
        if self._hydra_ws.get_detector_shift() is not None:
            geometry_calibration = True
        else:
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

        else:
            # no vanadium
            van_array = None
            van_duration = None

        self._reduction_manager.reduce_diffraction_data(self._session,
                                                        apply_calibrated_geometry=geometry_calibration,
                                                        min_2theta=min_2theta,
                                                        max_2theta=max_2theta,
                                                        num_bins=num_bins,
                                                        sub_run_list=self._sub_runs,
                                                        delta_2theta=delta_2theta,
                                                        mask=mask,
                                                        mask_id=mask_id,
                                                        vanadium_counts=van_array,
                                                        van_duration=van_duration,
                                                        eta_step=eta_step,
                                                        eta_min=eta_min,
                                                        eta_max=eta_max)

    def plot_reduced_data(self, sub_run_number=None):

        if sub_run_number is None:
            sub_runs = self._reduction_manager.get_sub_runs(self._session)
        else:
            sub_runs = [sub_run_number]

        for sub_run_i in sub_runs:
            vec_x, vec_y = self._reduction_manager.get_reduced_diffraction_data(self._session, sub_run_i)
            plt.plot(vec_x, vec_y)
        plt.show()

    def save_diffraction_data(self, output_file_name=None, append_mode=False, ignore_raw_counts=True):
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
            self._hydra_ws.save_experimental_data(out_file, sub_runs=self._sub_runs,
                                                  ignore_raw_counts=ignore_raw_counts)

        # Calibrated wave length shall be written
        self._hydra_ws.save_wavelength(out_file)

        # Write & close
        self._hydra_ws.save_reduced_diffraction_data(out_file, sub_runs=self._sub_runs)
