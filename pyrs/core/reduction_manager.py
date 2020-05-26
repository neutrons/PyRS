# Reduction engine including slicing
from __future__ import (absolute_import, division, print_function)  # python3 compatibility
import os
import numpy as np
from pyrs.core import workspaces
from pyrs.core import instrument_geometry
from pyrs.core import mask_util
from pyrs.core import reduce_hb2b_mtd
from pyrs.core import reduce_hb2b_pyrs
from pyrs.core.nexus_conversion import NeXusConvertingApp
from pyrs.dataobjects import HidraConstants
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode
from pyrs.utilities import calibration_file_io
from pyrs.utilities import checkdatatypes


class HB2BReductionManager(object):
    """
    A data reduction manager of HB2B

    1. It can work with both PyHB2BReduction and MantidHB2BReduction seamlessly
    2. It can always compare the results between 2 reduction engines
    3. It shall provide an API to calibration optimization
    """

    def __init__(self):
        """ initialization
        """
        # # calibration manager
        # self._calibration_manager = calibration_file_io.CalibrationManager()
        # self._geometry_calibration = instrument_geometry.AnglerCameraDetectorShift

        # workspace name or array vector
        self._curr_workspace = None
        self._curr_session_name = None
        self._session_dict = dict()  # [Project name/ID] = workspace / counts vector

        # Reduction engine
        self._last_reduction_engine = None

        # IDF
        self._mantid_idf = None

        # Vanadium
        self._van_ws = None

        # (default) number of bins
        self._num_bins = 2500

        # masks
        self._loaded_mask_files = list()
        self._loaded_mask_dict = dict()

        # Outputs
        self._output_directory = None

    @staticmethod
    def _generate_ws_name(file_name, is_nexus):
        ws_name = os.path.basename(file_name).split('.')[0]
        if is_nexus:
            # flag to show that there is no need to load instrument again
            ws_name = '{}__nexus'.format(ws_name)

        return ws_name

    def get_last_reduction_engine(self):
        """
        Get the reduction engine recently used
        :return:
        """
        return self._last_reduction_engine

    def get_reduced_diffraction_data(self, session_name, sub_run=None, mask_id=None):
        """ Get the reduce data
        :param session_name:
        :param sub_run:
        :param mask_id:
        :return: 2-vectors: 2theta and intensity
        """
        checkdatatypes.check_string_variable('Session name', session_name, list(self._session_dict.keys()))
        workspace = self._session_dict[session_name]

        data_set = workspace.get_reduced_diffraction_data(sub_run, mask_id)

        return data_set

    def get_sub_runs(self, session_name):
        """
        Get sub runs from a workspace belonged to a session
        :param session_name:
        :return:
        """
        checkdatatypes.check_string_variable('Session name', session_name, list(self._session_dict.keys()))
        workspace = self._session_dict[session_name]

        return workspace.get_sub_runs()

    def get_sub_run_detector_counts(self, session_name, sub_run):
        """
        Get the detector counts
        :param session_name:
        :param sub_run:
        :return:
        """
        checkdatatypes.check_string_variable('Session name', session_name, list(self._session_dict.keys()))
        workspace = self._session_dict[session_name]

        return workspace.get_detector_counts(sub_run)

    def get_sample_log_value(self, session_name, log_name, sub_run):
        """Get an individual sample log's value for a sub run

        :param session_name:
        :param log_name:
        :param sub_run:
        :return:
        """
        workspace = self._session_dict[session_name]

        log_value = workspace.get_sample_log_value(log_name, sub_run)

        return log_value

    def get_sample_logs_values(self, session_name, log_names):
        """ Get sample logs' value
        :param session_name:
        :param log_names:
        :return: List of ...
        """
        workspace = self._session_dict[session_name]

        log_values = list()
        for log_name in log_names:
            log_value_array = workspace.get_sample_log_values(log_name)
            log_values.append(log_value_array)

        return log_values

    def get_sample_logs_names(self, session_name, can_plot):
        """
        Get the names of all sample logs in the workspace
        :param session_name:
        :param can_plot:
        :return:
        """
        workspace = self._session_dict[session_name]

        sample_logs = workspace.sample_log_names

        return sample_logs

    def get_sub_run_2theta(self, session_name, sub_run):
        """
        Get the detector arm's 2theta position of a sub run
        :param session_name: name of the session for locating workspace
        :param sub_run:
        :return:
        """
        checkdatatypes.check_string_variable('Session name', session_name, list(self._session_dict.keys()))
        workspace = self._session_dict[session_name]

        return workspace.get_detector_2theta(sub_run)

    def get_detector_counts(self, session_name, sub_run):
        """ Get the raw counts from detector of the specified sub run
        :param session_name: name of the session for locating workspace
        :param sub_run: sub run number (integer)
        :return: array of detector counts
        """
        checkdatatypes.check_int_variable('Sub run number', sub_run, (0, None))
        workspace = self._session_dict[session_name]

        return workspace.get_detector_counts(sub_run)

    def get_hidra_workspace(self, session_name):
        """ Get the HIDRA workspace
        :param session_name: string as the session/workspace name
        :return: HidraWorkspace instance
        """
        checkdatatypes.check_string_variable('Session name', session_name, list(self._session_dict.keys()))

        # Check availability
        if session_name not in self._session_dict:
            raise RuntimeError('Session/HidraWorkspace {} does not exist. Available sessions/workspaces are {}'
                               ''.format(session_name, self._session_dict.keys()))

        workspace = self._session_dict[session_name]
        return workspace

    def init_session(self, session_name, hidra_ws=None):
        """
        Initialize a new session of reduction and thus to store data according to session name
        :return:
        """
        # Check inputs
        checkdatatypes.check_string_variable('Reduction session name', session_name)
        if session_name == '':
            raise RuntimeError('Session name {} is empty'.format(session_name))
        elif session_name in self._session_dict:
            print('[WARNING] Session {} is previously taken.  The HidraWorkspace associated '
                  'will be replaced if new HidraWorkspace is not None ({})'
                  ''.format(session_name, hidra_ws is None))

        if hidra_ws is None:
            # session is initialized without HidraWorkspace
            self._curr_workspace = workspaces.HidraWorkspace()
        else:
            # session starts with a HidraWorkspace
            checkdatatypes.check_type('HidraWorkspace', hidra_ws, workspaces.HidraWorkspace)
            self._curr_workspace = hidra_ws

        # set the current session name and add HidraWorkspace to dict
        self._curr_session_name = session_name
        self._session_dict[session_name] = self._curr_workspace

    def load_hidra_project(self, project_file_name, load_calibrated_instrument, load_detectors_counts,
                           load_reduced_diffraction):
        """ Load hidra project file and then CLOSE!
        :param project_file_name:
        :param load_calibrated_instrument:
        :param load_detectors_counts: Flag to load detector counts
        :param load_reduced_diffraction: Flag to reduced diffraction data
        :return: HidraWorkspace instance
        """
        # check inputs
        checkdatatypes.check_file_name(project_file_name, True, False, False, 'Project file to load')

        # Check
        if self._curr_workspace is None:
            raise RuntimeError('Call init_session to create a ReductionWorkspace')

        # PyRS HDF5
        # Check permission of file to determine the RW mode of HidraProject file
        if os.access(project_file_name, os.W_OK):
            # Read/Write: Append mode
            file_mode = HidraProjectFileMode.READWRITE
        else:
            # Read only
            file_mode = HidraProjectFileMode.READONLY
        project_h5_file = HidraProjectFile(project_file_name, mode=file_mode)

        # Load
        self._curr_workspace.load_hidra_project(project_h5_file,
                                                load_raw_counts=load_detectors_counts,
                                                load_reduced_diffraction=load_reduced_diffraction)

        # Close
        project_h5_file.close()
        return self._curr_workspace

    def load_instrument_file(self, instrument_file_name):
        """
        Load instrument (setup) file to current "workspace"
        :param instrument_file_name:
        :return:
        """
        # Check
        if self._curr_workspace is None:
            raise RuntimeError('Call init_session to create a ReductionWorkspace')

        instrument = calibration_file_io.import_instrument_setup(instrument_file_name)
        self._curr_workspace.set_instrument(instrument)

    def load_mask_file(self, mask_file_name):
        """ Load mask file to 1D array and auxiliary information
        :param mask_file_name:
        :return:
        """
        mask_vec, two_theta, note = mask_util.load_pyrs_mask(mask_file_name)

        # register the masks
        self._loaded_mask_files.append(mask_file_name)

        mask_id = os.path.basename(mask_file_name).split('.')[0] + '_{}'.format(hash(mask_file_name) % 100)
        self._loaded_mask_dict[mask_id] = mask_vec, two_theta, mask_file_name

        return two_theta, note, mask_id

    def load_vanadium(self, van_project_file):
        """Load vanadium from HiDRA project file

        Parameters
        ----------
        van_project_file : str
            vanadium HiDRA project file or NeXus file

        Returns
        -------
        ~numpy.narray, float
            1D array as vanadium counts and duration of vanadium run (second)

        """
        checkdatatypes.check_file_name(van_project_file, True, False, False, 'Vanadium project/NeXus file')

        if van_project_file.endswith('.nxs.h5'):
            # Input is nexus file
            # reduce with PyRS/Python
            converter = NeXusConvertingApp(van_project_file, mask_file_name=None)
            self._van_ws = converter.convert(use_mantid=False)

        else:
            # Input is HiDRA project file
            self._van_ws = workspaces.HidraWorkspace(name=van_project_file)

            # PyRS HDF5
            project_h5_file = HidraProjectFile(van_project_file, mode=HidraProjectFileMode.READONLY)

            # Load
            self._van_ws.load_hidra_project(project_h5_file,
                                            load_raw_counts=True,
                                            load_reduced_diffraction=False)

            # Close project file
            project_h5_file.close()

        # Process the vanadium counts
        sub_runs = self._van_ws.get_sub_runs()
        assert len(sub_runs) == 1, 'There shall be more than 1 sub run in vanadium project file'

        # get vanadium data
        van_array = self._van_ws.get_detector_counts(sub_runs[0]).astype(np.float64)

        # get vanadium run duration
        van_duration = self._van_ws.get_sample_log_value(HidraConstants.SUB_RUN_DURATION, sub_runs[0])

        return van_array, van_duration

    @staticmethod
    def _do_stat_to_van(van_array):
        # DEBUG: do statistic
        print('[INFO] Vanadium Counts: min = {}, max = {}, average = {}'
              ''.format(np.min(van_array), np.max(van_array), np.average(van_array)))
        # do statistic on each pixel with hard-coded range of counts
        # i.e., do a histogram on the vanadium counts
        count_range = [0, 2, 5, 10, 20, 30, 40, 50, 60, 80, 150, 300]
        per_count = 0
        pixel_count = 0
        for irange in range(len(count_range) - 1):
            threshold_min = count_range[irange]
            threshold_max = count_range[irange + 1]
            good_van_array = van_array[(van_array >= threshold_min) & (van_array < threshold_max)]
            print('[INFO] {} <= ... < {}: pixels number = {}  Percentage = {}'
                  ''.format(threshold_min, threshold_max,  good_van_array.size,
                            good_van_array.size * 1. / van_array.size))
            pixel_count += good_van_array.size
            per_count += good_van_array.size * 1. / van_array.size
        # END-FOR

        # Mask out zero count
        print('[DEBUG] VANADIUM Before Mask: {}\n\t# of NaN = {}'
              ''.format(van_array, np.where(np.isnan(van_array))[0].size))
        van_array[van_array < 3] = np.nan
        print('[DEBUG] VANADIUM After  Mask: {}\n\t# of NaN = {}'
              ''.format(van_array, np.where(np.isnan(van_array))[0].size))

    def get_loaded_mask_files(self):
        """
        Get the list of file names (full path) that have been loaded
        :return:
        """
        return self._loaded_mask_files[:]

    def get_mask_ids(self):
        """
        get IDs for loaded masks
        :return:
        """
        return sorted(self._loaded_mask_dict.keys())

    def get_mask_vector(self, mask_id):
        """
        Get the detector mask
        :param mask_id:  String as ID
        :return: a 1D array (0: mask, 1: keep)
        """
        checkdatatypes.check_string_variable('Mask ID', mask_id, list(self._loaded_mask_dict.keys()))

        return self._loaded_mask_dict[mask_id][0]

    def set_geometry_calibration(self, geometry_calibration):
        """ Load and apply calibration file
        :param geometry_calibration:
        :return:
        """
        # TODO FIXME - #81 NOWNOW - Still not sure how to apply!
        checkdatatypes.check_type('Geometry calibration', geometry_calibration,
                                  instrument_geometry.AnglerCameraDetectorShift)

        self._geometry_calibration = geometry_calibration

    def get_vanadium_counts(self, normalized):
        """Get vanadium counts of each pixel from current/default vanadium (HidraWorkspace)

        Usage: this will be called in order to fetch vanadium counts to reduce_diffraction_data()

        Returns
        -------
        numpy.ndarray
            1D vanadium counts array

        """
        if self._van_ws is None:
            raise RuntimeError('There is no default vanadium set up in reduction service')
        else:
            # get vanadium
            sub_run = self._van_ws.get_sub_runs()[0]
            van_counts_array = self._van_ws.get_detector_counts(sub_run)

            if normalized:
                van_duration = self._van_ws.get_sample_log_value(HidraConstants.SUB_RUN_DURATION, sub_run)
                van_counts_array /= van_duration

        return van_counts_array

    def reduce_diffraction_data(self, session_name, apply_calibrated_geometry, num_bins, use_pyrs_engine, sub_run_list,
                                mask, mask_id, vanadium_counts=None, van_duration=None, normalize_by_duration=True,
                                eta_step=None, eta_min=None, eta_max=None):
        """Reduce ALL sub runs in a workspace from detector counts to diffraction data

        Parameters
        ----------
        session_name
        apply_calibrated_geometry : ~AnglerCameraDetectorShift or bool
            3 options (1) user-provided AnglerCameraDetectorShift
                                          (2) True (use the one in workspace) (3) False (no calibration)
        num_bins : int
            number of bins
        use_pyrs_engine : bool
            flag to use PyRS engine; otherwise, use Mantid as diffraction pattern reduction engine
        mask : numpy.ndarray
            Mask
        mask_id : str or None
            ID for mask.  If mask ID is None, then it is the default universal mask applied to all data
        sub_run_list : List of None
            sub runs
        vanadium_counts : None or ~numpy.ndarray
            vanadium counts of each detector pixels for normalization
            If vanadium duration is recorded, the vanadium counts are normalized by its duration in seconds
        eta_step : float
            angular step size for out-of-plane reduction
        eta_min : float
            min angle for out-of-plane reduction
        eta_max : float
            max angle for out-of-plane reduction

        Returns
        -------
        None

        """
        # Get workspace
        if session_name is None:  # default as current session/workspace
            workspace = self._curr_workspace
        else:
            workspace = self._session_dict[session_name]

        # Process mask: No mask, Mask ID and mask vector
        default_mask = workspace.get_detector_mask(is_default=True, mask_id=None)
        if mask is None:
            # No use mask:  use default detector mask.  It could be None but does not matter
            mask_vec = default_mask
        elif isinstance(mask, str):
            # mask is determined by mask ID
            mask_vec = self.get_mask_vector(mask)
        else:
            # user supplied an array for mask
            checkdatatypes.check_numpy_arrays('Mask', [mask], dimension=1, check_same_shape=False)
            mask_vec = mask
        # END-IF-ELSE

        # Operate AND with default mask
        if default_mask is not None:
            mask_vec *= default_mask

        # Apply (or not) instrument geometry calibration shift
        if isinstance(apply_calibrated_geometry, instrument_geometry.AnglerCameraDetectorShift):
            det_pos_shift = apply_calibrated_geometry
        elif apply_calibrated_geometry:
            det_pos_shift = workspace.get_detector_shift()
        else:
            det_pos_shift = None
        # END-IF-ELSE
        print('[DB...BAT] Det Position Shift: {}'.format(det_pos_shift))

        if sub_run_list is None:
            sub_run_list = workspace.get_sub_runs()

        # Determine whether normalization by time is supported
        if normalize_by_duration and not workspace.has_sample_log(HidraConstants.SUB_RUN_DURATION):
            raise RuntimeError('Workspace {} does not have sample log {}.  Existing logs are {}'
                               ''.format(workspace, HidraConstants.SUB_RUN_DURATION,
                                         workspace.get_sample_log_names()))

        # Reset workspace's 2theta matrix and intensities
        workspace.reset_diffraction_data()

        for sub_run in sub_run_list:
            # get the duration
            if normalize_by_duration:
                duration_i = workspace.get_sample_log_value(HidraConstants.SUB_RUN_DURATION,
                                                            sub_run)
            else:
                # not normalized
                duration_i = 1.

            if eta_step is None:
                # reduce sub run
                self.reduce_sub_run_diffraction(workspace, sub_run, det_pos_shift,
                                                use_mantid_engine=not use_pyrs_engine,
                                                mask_vec_tuple=(mask_id, mask_vec),
                                                num_bins=num_bins,
                                                sub_run_duration=duration_i,
                                                vanadium_counts=vanadium_counts,
                                                van_duration=van_duration)
            else:
                # reduce sub run texture
                self.reduce_sub_run_texture(workspace, sub_run, det_pos_shift,
                                            use_mantid_engine=not use_pyrs_engine,
                                            mask_vec_tuple=(mask_id, mask_vec),
                                            num_bins=num_bins,
                                            sub_run_duration=duration_i,
                                            vanadium_counts=vanadium_counts,
                                            van_duration=van_duration,
                                            eta_step=eta_step,
                                            eta_min=eta_min,
                                            eta_max=eta_max)

            # END-IF (texture)

        # END-FOR (sub run)

    def setup_reduction_engine(self, workspace, sub_run, geometry_calibration, use_mantid_engine):
        """Setup reduction engine to reduce data (workspace or vector) to 2-theta ~ I

        Builds a new 2theta pixel map if none is present or if the detector has moved

        Parameters
        ----------
        workspace : HidraWorkspace
            workspace with detector counts and position
        sub_run : integer
            sub run number in workspace to reduce
        geometry_calibration : instrument_geometry.AnglerCameraDetectorShift
            instrument geometry to calculate diffraction pattern
        use_mantid_engine : boolean
            flag to use Mantid as reduction engine which is rarely used

        Returns
        -------
        None

        """

        # Get the raw data
        raw_count_vec = workspace.get_detector_counts(sub_run)

        # Retrieve 2-theta and L2 from loaded workspace (DAS)
        two_theta = workspace.get_detector_2theta(sub_run)
        l2 = workspace.get_l2(sub_run)

        if sub_run > 1:
            rebuild_instrument = two_theta != workspace.get_detector_2theta(sub_run - 1)
        else:
            rebuild_instrument = True

        # Convert 2-theta from DAS convention to Mantid/PyRS convention
        mantid_two_theta = -two_theta

        # Set up reduction engine and also
        if not rebuild_instrument:
            reduction_engine = self._last_reduction_engine
            reduction_engine.set_raw_counts(raw_count_vec)
        else:
            if use_mantid_engine:
                reduction_engine = reduce_hb2b_mtd.MantidHB2BReduction(self._mantid_idf)
            else:
                reduction_engine = reduce_hb2b_pyrs.PyHB2BReduction(workspace.get_instrument_setup())

            # Set up reduction engine
            reduction_engine.set_experimental_data(mantid_two_theta, l2, raw_count_vec)
            reduction_engine.build_instrument(geometry_calibration)

        return reduction_engine

    # NOTE: Refer to compare_reduction_engines_tst
    def reduce_sub_run_diffraction(self, workspace, sub_run, geometry_calibration, use_mantid_engine,
                                   mask_vec_tuple, min_2theta=None, max_2theta=None, num_bins=1000,
                                   sub_run_duration=None, vanadium_counts=None, van_duration=None):
        """Reduce import data (workspace or vector) to 2-theta ~ I

        The binning of 2theta is linear in range (min, max) with given resolution

        1. 2-theta range:
            If nothing is given, then the default range is from detector arm's 2theta value
           going up and down with half of default_two_theta_range
        2. 2-theta resolution/step size:
            If 2theta resolution is not given, num_bins will be used to determine resolution with 2-theta range;
            Otherwise, use resolution

        Normalization to time/duration
        ------------------------------
        If both sub run duration and vanadium duration are given
        normalized intensity = raw histogram / vanadium histogram * vanadium duration / sub run duration

        Parameters
        ----------
        workspace : HidraWorkspace
            workspace with detector counts and position
        sub_run : integer
            sub run number in workspace to reduce
        geometry_calibration : instrument_geometry.AnglerCameraDetectorShift
            instrument geometry to calculate diffraction pattern
        use_mantid_engine : boolean
            flag to use Mantid as reduction engine which is rarely used
        mask_vec_tuple : tuple (str, numpy.ndarray)
            mask ID and 1D array for masking (1 to keep, 0 to mask out)
        min_2theta : float or None
            min 2theta
        max_2theta : float or None
            max 2theta
        num_bins : float or None
            2theta resolution/step
        num_bins : int
            number of bins
        sub_run_duration: float or None
            If None, then no normalization to time (duration) will be done. Otherwise, intensity will be
            normalized by time (duration)
        vanadium_counts : numpy.ndarray or None
            detector pixels' vanadium for efficiency and normalization.
            If vanadium duration is recorded, the vanadium counts are normalized by its duration in seconds

        Returns
        -------
        None

        """

        # Setup reduction enegine
        reduction_engine = self.setup_reduction_engine(workspace, sub_run, geometry_calibration, use_mantid_engine)

        # Apply mask
        mask_id, mask_vec = mask_vec_tuple

        # Histogram
        bin_centers, hist, variances = self.convert_counts_to_diffraction(reduction_engine,
                                                                          (min_2theta, max_2theta),
                                                                          num_bins, mask_vec, vanadium_counts)

        if van_duration is not None:
            hist *= van_duration
            variances *= van_duration

        # record
        workspace.set_reduced_diffraction_data(sub_run, mask_id, bin_centers, hist, variances)
        self._last_reduction_engine = reduction_engine

    def generate_eta_roi_vector(self, eta_step, eta_min, eta_max):

        """Generate vector of out-of-plane angle centers

        Determines list of out-of-plane centers for texture reduction
        generated list is centered at eta of 0
        e.g.; 0, +/-step, +/-2step, +/-3step, ...

        ------------------------------
        Parameters
        ----------
        eta_step : float
            angular step size for out-of-plane reduction
        eta_min : float
            min angle for out-of-plane reduction
        eta_max : float
            max angle for out-of-plane reduction

        Returns
        -------
        numpy array

        """

        # set eta bounds to default if None is provided
        if eta_min is None:
            eta_min = -8.2
        if eta_max is None:
            eta_max = 8.2

        # center list is generated in two steps to ensure 0 centering
        if eta_min < 0:
            eta_roi_start = 0
        else:
            eta_roi_start = eta_min

        Upper = np.arange(eta_roi_start, eta_max - eta_step / 2., eta_step)

        if eta_min < 0:
            Lower = np.arange(-1 * eta_step, eta_min + eta_step / 2., -1 * eta_step)
        else:
            Lower = np.array([])

        return np.concatenate((Upper, Lower))

    # NOTE: Refer to compare_reduction_engines_tst
    def reduce_sub_run_texture(self, workspace, sub_run, geometry_calibration, use_mantid_engine,
                               mask_vec_tuple, min_2theta=None, max_2theta=None, num_bins=1000,
                               sub_run_duration=None, vanadium_counts=None, van_duration=None,
                               eta_step=None, eta_min=None, eta_max=None):

        """Reduce import data (workspace or vector) to 2-theta ~ I

        The binning of 2theta is linear in range (min, max) with given resolution

        1. 2-theta range:
            If nothing is given, then the default range is from detector arm's 2theta value
           going up and down with half of default_two_theta_range
        2. 2-theta resolution/step size:
            If 2theta resolution is not given, num_bins will be used to determine resolution with 2-theta range;
            Otherwise, use resolution

        Normalization to time/duration
        ------------------------------
        If both sub run duration and vanadium duration are given
        normalized intensity = raw histogram / vanadium histogram * vanadium duration / sub run duration

        Parameters
        ----------
        workspace : HidraWorkspace
            workspace with detector counts and position
        sub_run : integer
            sub run number in workspace to reduce
        geometry_calibration : instrument_geometry.AnglerCameraDetectorShift
            instrument geometry to calculate diffraction pattern
        use_mantid_engine : boolean
            flag to use Mantid as reduction engine which is rarely used
        mask_vec_tuple : tuple (str, numpy.ndarray)
            mask ID and 1D array for masking (1 to keep, 0 to mask out)
        min_2theta : float or None
            min 2theta
        max_2theta : float or None
            max 2theta
        num_bins : float or None
            2theta resolution/step
        num_bins : int
            number of bins
        sub_run_duration: float or None
            If None, then no normalization to time (duration) will be done. Otherwise, intensity will be
            normalized by time (duration)
        vanadium_counts : numpy.ndarray or None
            detector pixels' vanadium for efficiency and normalization.
            If vanadium duration is recorded, the vanadium counts are normalized by its duration in seconds
        eta_step : float
            angular step size for out-of-plane reduction
        eta_min : float
            min angle for out-of-plane reduction
        eta_max : float
            max angle for out-of-plane reduction

        Returns
        -------
        None

        """

        # Setup reduction enegine
        reduction_engine = self.setup_reduction_engine(workspace, sub_run, geometry_calibration, use_mantid_engine)

        # Apply mask
        mask_id, mask_vec = mask_vec_tuple

        # Get vector of pixel eta positions
        eta_vec = reduction_engine.get_eta_value()

        # validate input
        if abs(eta_step) is not eta_step:
            eta_step = abs(eta_step)

        # Generate eta roi vector
        eta_roi_vec = self.generate_eta_roi_vector(eta_step, eta_min, eta_max)

        for eta_cent in eta_roi_vec:
            # define mask to isolate narrow eta wedge
            eta_mask = np.zeros_like(eta_vec)
            eta_mask[eta_vec > (eta_cent + eta_step / 2.)] = 1
            eta_mask[eta_vec < (eta_cent - eta_step / 2.)] = 1
            eta_mask[mask_vec] = 1

            # Histogram data
            bin_centers, hist, variances = self.convert_counts_to_diffraction(reduction_engine,
                                                                              (min_2theta, max_2theta),
                                                                              num_bins, eta_mask, vanadium_counts)

            if van_duration is not None:
                hist *= van_duration
                variances *= van_duration

            if mask_id is None:
                eta_mask_id = 'eta_{}'.format(eta_cent)
            else:
                eta_mask_id = '{}_eta_{}'.format(mask_id, eta_cent)

            # record
            workspace.set_reduced_diffraction_data(sub_run, eta_mask_id, bin_centers, hist, variances)

        self._last_reduction_engine = reduction_engine

    def convert_counts_to_diffraction(self, reduction_engine,
                                      two_theta_range, num_bins, mask_array, vanadium_array):

        """Histogram detector counts with detectors' 2theta angle

        Parameters
        ----------
        reduction_engine
        two_theta_range : (min_2theta, max_2theta)
            min_2theta : float or None
                min 2theta
            max_2theta : float or None
                max 2theta
        num_bins : float or None
            2theta resolution/step
        mask_array : numpy.ndarray or None
            mask: 1 to keep, 0 to mask (exclude)
        vanadium_counts : numpy.ndarray or None
            detector pixels' vanadium for efficiency and normalization.
            If vanadium duration is recorded, the vanadium counts are normalized by its duration in seconds

        Returns
        -------
        numpy.ndarray, numpy.ndarray, numpy.ndarray
            2theta bins, histogram of counts, variances of counts

        """

        # Default minimum and maximum 2theta are related with
        min_2theta, max_2theta = two_theta_range

        # Get the 2theta values for all pixels
        pixel_2theta_array = reduction_engine.instrument.get_pixels_2theta(1)

        bin_boundaries_2theta = self.generate_2theta_histogram_vector(min_2theta, num_bins, max_2theta,
                                                                      pixel_2theta_array, mask_array)

        # Histogram
        data_set = reduction_engine.reduce_to_2theta_histogram(bin_boundaries_2theta,
                                                               mask_array=mask_array,
                                                               is_point_data=True,
                                                               vanadium_counts_array=vanadium_array)
        bin_centers = data_set[0]
        hist = data_set[1]
        variances = data_set[2]

        return bin_centers, hist, variances

    @staticmethod
    def generate_2theta_histogram_vector(min_2theta, num_bins, max_2theta,
                                         pixel_2theta_array, mask_array):
        """Generate a 1-D array for histogram 2theta bins

        Parameters
        ----------
        min_2theta : float or None
            minimum 2theta or None
        num_bins : int
            nubmer of bins
        max_2theta : float  or None
             maximum 2theta and must be integer
        pixel_2theta_array : numpy.ndarray
            2theta of each detector pixel
        mask_array : numpy.ndarray or None
            array of mask

        Returns
        -------
        numpy.ndarray
            2theta values serving as bin boundaries, such its size is 1 larger than num_bins

        """
        # If default value is required: set the default
        if min_2theta is None or max_2theta is None:
            # check inputs
            if mask_array is None:
                checkdatatypes.check_numpy_arrays('Pixel 2theta angles', [pixel_2theta_array], 1, False)
            else:
                checkdatatypes.check_numpy_arrays('Pixel 2theta position and mask array',
                                                  [pixel_2theta_array, mask_array], 1, True)
                # mask
                pixel_2theta_array = pixel_2theta_array[np.where(mask_array == 1)]
            # END-IF

            if min_2theta is None:
                # lower boundary of 2theta for bins is the minimum 2theta angle of all the pixels
                min_2theta = np.min(pixel_2theta_array)
            if max_2theta is None:
                # upper boundary of 2theta for bins is the maximum 2theta angle of all the pixels
                max_2theta = np.max(pixel_2theta_array)
        # END-IF

        step_2theta = (max_2theta - min_2theta) * 1. / num_bins

        # Check inputs
        checkdatatypes.check_float_variable('Minimum 2theta', min_2theta, (-180, 180))
        checkdatatypes.check_float_variable('Maximum 2theta', max_2theta, (-180, 180))
        checkdatatypes.check_float_variable('2theta bin size', step_2theta, (0, 180))
        if min_2theta >= max_2theta:
            raise RuntimeError('2theta range ({}, {}) is invalid for generating histogram'
                               ''.format(min_2theta, max_2theta))

        # Create 2theta: these are bin edges from (min - 1/2) to (max + 1/2) with num_bins bins
        # and (num_bins + 1) data points
        vec_2theta = np.arange(num_bins + 1).astype(float) * step_2theta + (min_2theta - step_2theta)
        if vec_2theta.shape[0] != num_bins + 1:
            raise RuntimeError('Expected = {} vs {}\n2theta min max  = {}, {}\n2thetas: {}'
                               ''.format(num_bins, vec_2theta.shape, min_2theta, max_2theta,
                                         vec_2theta))

        # Sanity check
        assert vec_2theta.shape[0] == num_bins + 1, '2theta bins (boundary)\'size ({}) shall be exactly ' \
                                                    '1 larger than specified num_bins ({})' \
                                                    ''.format(vec_2theta.shape, num_bins)

        return vec_2theta

    def save_reduced_diffraction(self, session_name, output_name):
        """
        Save the reduced diffraction data to file
        :param session_name:
        :param output_name:
        :return:
        """
        checkdatatypes.check_file_name(output_name, False, True, False, 'Output reduced file')

        workspace = self._session_dict[session_name]

        # Open
        if os.path.exists(output_name):
            io_mode = HidraProjectFileMode.READWRITE
        else:
            io_mode = HidraProjectFileMode.OVERWRITE
        project_file = HidraProjectFile(output_name, io_mode)

        # Save
        workspace.save_reduced_diffraction_data(project_file)

        # Close
        project_file.save()

    def set_mantid_idf(self, idf_name):
        """
        set the IDF file to reduction engine
        :param idf_name:
        :return:
        """
        checkdatatypes.check_file_name(idf_name, True, False, False, 'Mantid IDF file')
        if not idf_name.lower().endswith('.xml'):
            raise RuntimeError('Mantid IDF {} must end with .xml'.format(idf_name))

        self._mantid_idf = idf_name

    def set_output_dir(self, output_dir):
        """
        set the directory for output data
        :param output_dir:
        :return:
        """
        checkdatatypes.check_file_name(output_dir, True, True, True, 'Output directory')

        self._output_directory = output_dir
