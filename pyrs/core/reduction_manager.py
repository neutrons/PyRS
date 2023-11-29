# Reduction engine including slicing
import os
import numpy as np
from pyrs.core import workspaces
from pyrs.core import instrument_geometry
from pyrs.core import mask_util
from pyrs.core import reduce_hb2b_pyrs
from pyrs.core.nexus_conversion import NeXusConvertingApp
from pyrs.dataobjects import HidraConstants  # type: ignore
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore
from pyrs.utilities import checkdatatypes
from pyrs.utilities.convertdatatypes import to_float, to_int
from typing import Optional


class HB2BReductionManager:
    """
    A data reduction manager of HB2B

    1. It can work with both PyHB2BReduction and MantidHB2BReduction seamlessly
    2. It can always compare the results between 2 reduction engines
    3. It shall provide an API to calibration optimization
    """

    def __init__(self):
        """ initialization
        """
        # workspace name or array vector
        self._curr_workspace = None
        self._session_dict = dict()  # [Project name/ID] = workspace / counts vector

        # Reduction engine
        self._last_reduction_engine = None

        # Vanadium
        self._van_ws = None

        # (default) number of bins
        self._num_bins = 2500

        # masks
        self._loaded_mask_files = list()
        self._loaded_mask_dict = dict()

    def get_reduced_diffraction_data(self, session_name, sub_run=None, mask_id=None):
        """ Get the reduce data

        :param str session_name: name of the session for locating workspace
        :param int sub_run: sub-run index
        :param mask_id: mask used to reduce diffraction data (default is None)
        :type mask_id: str, optional
        :return: 2-vectors: 2theta and intensity
        :rtype: numpy.ndarray
        """
        checkdatatypes.check_string_variable('Session name', session_name, list(self._session_dict.keys()))
        workspace = self._session_dict[session_name]

        data_set = workspace.get_reduced_diffraction_data(sub_run, mask_id)

        return data_set

    def get_sub_runs(self, session_name):
        """Get sub runs from a workspace belonged to a session

        :param str session_name: name of the session for locating workspace
        :return: return a list of sub-runs in the workspace
        :rtype: list
        """
        checkdatatypes.check_string_variable('Session name', session_name, list(self._session_dict.keys()))
        workspace = self._session_dict[session_name]

        return workspace.get_sub_runs()

    def get_sample_log_value(self, session_name, log_name, sub_run):
        """Get an individual sample log's value for a sub run

        :param str session_name: name of the session for locating workspace
        :param str log_name: Name of the sample log
        :param int sub_run: sub-run index
        :return: vector of sample log values
        :rtype: numpy.ndarray
        """
        workspace = self._session_dict[session_name]

        log_value = workspace.get_sample_log_value(log_name, sub_run)

        return log_value

    def get_sample_logs_names(self, session_name):
        """Get the names of all sample logs in the workspace

        :param str session_name: name of the session for locating workspace
        :return: list of sample logs
        :rtype: list
        """
        workspace = self._session_dict[session_name]

        sample_logs = workspace.sample_log_names

        return sample_logs

    def get_sub_run_2theta(self, session_name, sub_run):
        """Get the detector arm's 2theta position of a sub run

        :param str session_name: name of the session for locating workspace
        :param int sub_run: sub-run index
        :return: 2theta vector for the sub-run
        :rtype: numpy.ndarray
        """

        checkdatatypes.check_string_variable('Session name', session_name, list(self._session_dict.keys()))
        workspace = self._session_dict[session_name]

        return workspace.get_detector_2theta(sub_run)

    def get_detector_counts(self, session_name, sub_run: int):
        """ Get the raw counts from detector of the specified sub run

        :param str session_name: name of the session for locating workspace
        :param int sub_run: sub run number
        :return: array of detector counts
        :rtype: numpy.ndarray
        """

        sub_run = to_int('Sub run number', sub_run, min_value=0)
        workspace = self._session_dict[session_name]

        return workspace.get_detector_counts(sub_run)

    def init_session(self, session_name, hidra_ws=None):
        """
        Initialize a new session of reduction and thus to store data according to session name
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

        self._session_dict[session_name] = self._curr_workspace

    def load_hidra_project(self, project_file_name, load_calibrated_instrument, load_detectors_counts,
                           load_reduced_diffraction):
        """ Load hidra project file

        :param str project_file_name: filename for the Hidra project file
        :param bool load_calibrated_instrument:
        :param bool load_detectors_counts: Flag to load detector counts
        :param bool load_reduced_diffraction: Flag to reduced diffraction data
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

    def load_mask_file(self, mask_file_name):
        """ Load mask file to 1D array and auxiliary information

        :param str mask_file_name: mask filename
        :return: two_theta, note, mask_id
        """
        mask_vec, two_theta, note = mask_util.load_pyrs_mask(mask_file_name)

        # register the masks
        self._loaded_mask_files.append(mask_file_name)

        mask_id = os.path.basename(mask_file_name).split('.')[0] + '_{}'.format(hash(mask_file_name) % 100)
        self._loaded_mask_dict[mask_id] = mask_vec, two_theta, mask_file_name

        return two_theta, note, mask_id

    def load_vanadium(self, van_project_file):
        """Load vanadium from HiDRA project file or NeXus file

        :param str van_project_file: vanadium HiDRA project file or NeXus file
        :returns:
            - van_array (numpy.ndarray) - 1D array as vanadium counts
            - van_duration (numpy.ndarray) - duration of vanadium run (second)
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

    def get_mask_vector(self, mask_id):
        """
        Get the detector mask

        :param str mask_id:  String as ID
        :return: a 1D array (0: mask, 1: keep)
        :rtype: numpy.ndarray
        """
        checkdatatypes.check_string_variable('Mask ID', mask_id, list(self._loaded_mask_dict.keys()))

        return self._loaded_mask_dict[mask_id][0]

    def reduce_diffraction_data(self, session_name, apply_calibrated_geometry, num_bins, sub_run_list,
                                mask, mask_id, vanadium_counts=None, van_duration=None, normalize_by_duration=True,
                                eta_step=None, eta_min=None, eta_max=None, min_2theta=None, max_2theta=None,
                                delta_2theta=None):
        """Reduce ALL sub runs in a workspace from detector counts to diffraction data

        :param session_name: Name for the reduction session
        :type session_name: str
        :param apply_calibrated_geometry: (1) user-provided DENEXDetectorShift
            (2) True (use calibrated geometry in workspace)
            (3) False (no calibration)
        :type apply_calibrated_geometry: DENEXDetectorShift, bool
        :param num_bins: 2theta resolution/step
        :type num_bins: int
        :param mask: 1D array for masking (1 to keep, 0 to mask out)
        :type mask_vec_tuple: numpy.ndarray
        :param mask_id: ID for mask.  If mask ID is None, then it is the default universal mask applied to all data
        :type mask_id: str, optional
        :param integer sub_run: sub run number in workspace to reduce
        :param vanadium_counts: detector pixels' vanadium for efficiency and normalization.
            If vanadium duration is recorded, the vanadium counts are normalized by its duration in seconds
        :type vanadium_counts: numpy.ndarray, optional
        :param eta_step: angular step size for out-of-plane reduction
        :type eta_step: float, optional
        :param eta_min: min angle for out-of-plane reduction
        :type eta_min: float, optional
        :param eta_max: max angle for out-of-plane reduction
        :type eta_max: float, optional
        :param min_2theta: min 2theta
        :type min_2theta: float, optional
        :param max_2theta: max 2theta
        :type max_2theta: float, optional
        :param delta_2theta: 2theta increment in the reduced diffraction data
        :type delta_2theta: float, optional
        :return: None
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

        # Operate AND with default mask
        if default_mask is not None:
            mask_vec *= default_mask

        # Apply (or not) instrument geometry calibration shift
        if isinstance(apply_calibrated_geometry, instrument_geometry.DENEXDetectorShift):
            det_pos_shift = apply_calibrated_geometry
        elif apply_calibrated_geometry:
            det_pos_shift = workspace.get_detector_shift()
        else:
            det_pos_shift = None

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
                                                mask_vec_tuple=(mask_id, mask_vec),
                                                min_2theta=min_2theta,
                                                max_2theta=max_2theta,
                                                num_bins=num_bins,
                                                delta_2theta=delta_2theta,
                                                sub_run_duration=duration_i,
                                                vanadium_counts=vanadium_counts,
                                                van_duration=van_duration)
            else:
                # reduce sub run texture
                self.reduce_sub_run_texture(workspace, sub_run, det_pos_shift,
                                            mask_vec_tuple=(mask_id, mask_vec),
                                            min_2theta=min_2theta,
                                            max_2theta=max_2theta,
                                            num_bins=num_bins,
                                            sub_run_duration=duration_i,
                                            vanadium_counts=vanadium_counts,
                                            van_duration=van_duration,
                                            eta_step=eta_step,
                                            eta_min=eta_min,
                                            eta_max=eta_max,
                                            delta_2theta=delta_2theta)

    def setup_reduction_engine(self, workspace, sub_run, geometry_calibration):
        """Setup reduction engine to reduce data (workspace or vector) to 2-theta ~ I

        Builds a new 2theta pixel map if none is present or if the detector has moved

        :param HidraWorkspace workspace: workspace with detector counts and position
        :param int sub_run: sub run number in workspace to reduce
        :param DENEXDetectorShift geometry_calibration: instrument geometry to calculate diffraction pattern
        :return: None
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

        if self._last_reduction_engine is None:
            rebuild_instrument = True

        # Convert 2-theta from DAS convention to Mantid/PyRS convention
        mantid_two_theta = -two_theta

        # Set up reduction engine
        if not rebuild_instrument:
            reduction_engine = self._last_reduction_engine
            reduction_engine.set_raw_counts(raw_count_vec)
        else:
            reduction_engine = reduce_hb2b_pyrs.PyHB2BReduction(workspace.get_instrument_setup())

            # Set up reduction engine
            reduction_engine.set_experimental_data(mantid_two_theta, l2, raw_count_vec)
            reduction_engine.build_instrument(geometry_calibration)

        return reduction_engine

    # NOTE: Refer to compare_reduction_engines_tst
    def reduce_sub_run_diffraction(self, workspace, sub_run, geometry_calibration,
                                   mask_vec_tuple, min_2theta=None, max_2theta=None, num_bins=1000,
                                   sub_run_duration=None, vanadium_counts=None, van_duration=None,
                                   delta_2theta=None):
        """Reduce import data (workspace or vector) to 2-theta ~ I

        The binning of 2theta is linear in range (min, max) with given resolution

        1. 2-theta range:
            If nothing is given, then the default range is from detector arm's 2theta value
            going up and down with half of default_two_theta_range
        2. 2-theta resolution/step size:
            If 2theta resolution is not given, num_bins will be used to determine resolution with 2-theta range;
            Otherwise, use resolution

        Normalization to time/duration

        If both sub run duration and vanadium duration are given
        normalized intensity = raw histogram / vanadium histogram * vanadium duration / sub run duration

        :param HidraWorkspace workspace: workspace with detector counts and position
        :param integer sub_run: sub run number in workspace to reduce
        :param DENEXDetectorShift geometry_calibration: instrument geometry to calculate diffraction pattern
        :param mask_vec_tuple: mask ID and 1D array for masking (1 to keep, 0 to mask out)
        :type mask_vec_tuple: tuple, [str, numpy.ndarray]
        :param min_2theta: min 2theta
        :type min_2theta: float, optional
        :param max_2theta: max 2theta
        :type max_2theta: float, optional
        :param num_bins: 2theta resolution/step
        :type num_bins: float, optional
        :param int num_bins: number of bins
        :param delta_2theta: 2theta increment in the reduced diffraction data
        :type delta_2theta: float, optional
        :param sub_run_duration: If None, then no normalization to time (duration) will be done.
            Otherwise, intensity will be normalized by time (duration)
        :type sub_run_duration: float, optional
        :param vanadium_counts: detector pixels' vanadium for efficiency and normalization.
            If vanadium duration is recorded, the vanadium counts are normalized by its duration in seconds
        :type vanadium_counts: numpy.ndarray, optional
        :return: None

        """

        # Setup reduction enegine
        reduction_engine = self.setup_reduction_engine(workspace, sub_run, geometry_calibration)

        # Apply mask
        mask_id, mask_vec = mask_vec_tuple

        # Histogram
        bin_centers, hist, variances = self.convert_counts_to_diffraction(reduction_engine,
                                                                          (min_2theta, max_2theta),
                                                                          num_bins, delta_2theta, mask_vec,
                                                                          vanadium_counts)

        # record
        workspace.set_reduced_diffraction_data(sub_run, mask_id, bin_centers, hist, variances)
        self._last_reduction_engine = reduction_engine

    def generate_eta_roi_vector(self, eta_step, eta_min, eta_max):
        """Generate vector of out-of-plane angle centers

        Determines list of out-of-plane centers for texture reduction generated list is centered at eta of 0
            e.g.; 0, +/-step, +/-2step, +/-3step, ...

        :param float eta_step: angular step size for out-of-plane reduction
        :param float eta_min: min angle for out-of-plane reduction
        :param float eta_max: max angle for out-of-plane reduction
        :return: eta_roi_vector
        :rtype: numpy.ndarray

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
    def reduce_sub_run_texture(self, workspace, sub_run, geometry_calibration,
                               mask_vec_tuple, min_2theta=None, max_2theta=None, num_bins=1000,
                               sub_run_duration=None, vanadium_counts=None, van_duration=None,
                               eta_step=None, eta_min=None, eta_max=None, delta_2theta=None):

        """
        Reduce import data (workspace or vector) to 2-theta ~ I

        The binning of 2theta is linear in range (min, max) with given resolution

        1) 2-theta range:
        If nothing is given, then the default range is from detector arm's 2theta value
        going up and down with half of default_two_theta_range
        2) 2-theta resolution/step size:
        If 2theta resolution is not given, num_bins will be used to determine resolution with 2-theta range;
        Otherwise, use resolution

        Normalization to time/duration
        If both sub run duration and vanadium duration are given
        normalized intensity = raw histogram / vanadium histogram * vanadium duration / sub run duration

        :param HidraWorkspace workspace: workspace with detector counts and position
        :param integer sub_run: sub run number in workspace to reduce
        :param DENEXDetectorShift geometry_calibration: instrument geometry to calculate diffraction pattern
        :param mask_vec_tuple: mask ID and 1D array for masking (1 to keep, 0 to mask out)
        :type mask_vec_tuple: tuple, [str, numpy.ndarray]
        :param min_2theta: min 2theta
        :type min_2theta: float, optional
        :param max_2theta: max 2theta
        :type max_2theta: float, optional
        :param num_bins: 2theta resolution/step
        :type num_bins: float, optional
        :param int num_bins: number of bins
        :param delta_2theta: 2theta increment in the reduced diffraction data
        :type delta_2theta: float, optional
        :param sub_run_duration: If None, then no normalization to time (duration) will be done.
            Otherwise, intensity will be normalized by time (duration)
        :type sub_run_duration: float, optional
        :param vanadium_counts: detector pixels' vanadium for efficiency and normalization.
            If vanadium duration is recorded, the vanadium counts are normalized by its duration in seconds
        :type vanadium_counts: numpy.ndarray, optional
        :param float eta_step: angular step size for out-of-plane reduction
        :param float eta_min: min angle for out-of-plane reduction
        :param float eta_max: max angle for out-of-plane reduction
        :return: None
        """

        # Setup reduction enegine
        reduction_engine = self.setup_reduction_engine(workspace, sub_run, geometry_calibration)

        # Apply mask
        mask_id, mask_vec = mask_vec_tuple

        # Get vector of pixel eta positions
        eta_vec = reduction_engine.get_eta_value()

        # validate input
        if abs(eta_step) is not eta_step:
            eta_step = abs(eta_step)

        # Generate eta roi vector
        eta_roi_vec = self.generate_eta_roi_vector(eta_step, eta_min, eta_max)

        for eta_cent in eta_roi_vec:            # define mask to isolate narrow eta wedge
            # here a value of zero means do not use
            eta_mask = np.ones_like(eta_vec)
            eta_mask[eta_vec > (eta_cent + eta_step / 2.)] = 0
            eta_mask[eta_vec < (eta_cent - eta_step / 2.)] = 0

            if mask_vec is not None:
                eta_mask[mask_vec] = 0

            # Histogram data
            bin_centers, hist, variances = self.convert_counts_to_diffraction(reduction_engine,
                                                                              (min_2theta, max_2theta),
                                                                              num_bins, delta_2theta, eta_mask,
                                                                              vanadium_counts)

            if mask_id is None:
                eta_mask_id = 'eta_{}'.format(eta_cent)
            else:
                eta_mask_id = '{}_eta_{}'.format(mask_id, eta_cent)

            # record
            workspace.set_reduced_diffraction_data(sub_run, eta_mask_id, bin_centers, hist, variances)

        self._last_reduction_engine = reduction_engine

    def convert_counts_to_diffraction(self, reduction_engine,
                                      two_theta_range, num_bins, delta_2theta, mask_array, vanadium_array):
        """Histogram detector counts for a defined 2theta range

        :param object reduction_engine: object containing information about the instrument geometry
        :param two_theta_range: tuple with the min and max 2theta range
        :type two_theta_range: tuple, [float, optional]
        :param num_bins: number of bins in 2theta vector
        :type num_bins: int, optional
        :param delta_2theta: 2theta increment in the reduced diffraction data
        :type delta_2theta: float, optional
        :param mask_array: mask: 1 to keep, 0 to mask (exclude)
        :type mask_array: numpy.ndarray, optional
        :param vanadium_array: detector pixels' vanadium for efficiency and normalization.
            If vanadium duration is recorded, the vanadium counts are normalized by its duration in seconds
        :type vanadium_array: numpy.ndarray, optional
        :return: 2theta bins, histogram of counts, variances of counts
        :rtype: numpy.ndarray
        """

        # Default minimum and maximum 2theta are related with
        min_2theta, max_2theta = two_theta_range

        # Get the 2theta values for all pixels
        pixel_2theta_array = reduction_engine.instrument.get_pixels_2theta(1)

        bin_boundaries_2theta = self.generate_2theta_histogram_vector(min_2theta, max_2theta, num_bins,
                                                                      pixel_2theta_array, mask_array, delta_2theta)

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
    def generate_2theta_histogram_vector(min_2theta: Optional[float], max_2theta: Optional[float],
                                         num_bins: int,
                                         pixel_2theta_array, mask_array,
                                         step_2theta=None):
        """Generate a 1-D array for histogram 2theta bins

        :param min_2theta: minimum 2theta to use for generating vector
        :type min_2theta: float, optional
        :param max_2theta: maximum 2theta to use for generating vector
        :type max_2theta: float, optional
        :param int num_bins: number of bins in 2theta vector
        :param numpy.ndarray pixel_2theta_array: 2theta of each detector pixel
        :param mask_array: vector for pixel mask
        :type mask_array: numpy.ndarray, optional
        :return: 2theta values serving as bin boundaries, such its size is 1 larger than num_bins
        :rtype: numpy.ndarray
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

            if min_2theta is None:
                # lower boundary of 2theta for bins is the minimum 2theta angle of all the pixels
                min_2theta = np.min(pixel_2theta_array)
            if max_2theta is None:
                # upper boundary of 2theta for bins is the maximum 2theta angle of all the pixels
                max_2theta = np.max(pixel_2theta_array)

        if step_2theta is None:
            step_2theta = (max_2theta - min_2theta) * 1. / num_bins
        else:
            num_bins = np.ceil((max_2theta - min_2theta) / step_2theta) + 1

        # Check inputs
        min_2theta = to_float('Minimum 2theta', min_2theta, 0.0, 140)
        max_2theta = to_float('Maximum 2theta', max_2theta, 0.5, 180)
        step_2theta = to_float('2theta bin size', step_2theta, 0, 180)
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
        """Save the reduced diffraction data to file

        :param str session_name: name of the session for locating workspace
        :param str output_name: output filename
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
