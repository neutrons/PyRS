import os
import datetime
import numpy
from mantid.simpleapi import FilterEvents, LoadEventNexus, LoadInstrument, GenerateEventsFilter
from mantid.simpleapi import ConvertSpectrumAxis, ResampleX, Transpose, AddSampleLog
from mantid.simpleapi import SortXAxis, CreateWorkspace
from mantid.api import AnalysisDataService as ADS
from pyrs.utilities import checkdatatypes
from pyrs.core.instrument_geometry import AnglerCameraDetectorShift
from pyrs.core import mantid_helper


def histogram_data(raw_vec_x, raw_vec_y, target_vec_2theta):
    """
    histogram again a set of point data (this is a backup solution)
    it yields exactly the same result as numpy.histogram() except it does not work with unordered vector X
    :param raw_vec_x:
    :param raw_vec_y:
    :param target_vec_2theta:
    :return:
    """
    raw_index = 0
    raw_size = raw_vec_x.shape[0]

    # compare the first entry
    if raw_vec_x[0] < target_vec_2theta[0]:
        # target range is smaller. need to throw away first several raw bins
        raw_index = numpy.searchsorted(raw_vec_x, [target_vec_2theta[0]])[0]
        target_index = 0
    elif raw_vec_x[0] > target_vec_2theta[0]:
        raw_index = 0
        target_index = numpy.searchsorted(target_vec_2theta, [raw_vec_x[0]])[0]
    else:
        # equal.. not very likely
        raw_index = 0
        target_index = 0

    target_size = target_vec_2theta.shape[0] - 1

    target_vec_y = numpy.zeros(shape=(target_size,), dtype='float')

    for bin_i in range(target_index, target_size):
        #
        # x_i = target_vec_2theta[bin_i]
        x_f = target_vec_2theta[bin_i+1]
        while raw_vec_x[raw_index] < x_f and raw_index < raw_size:
            target_vec_y[bin_i] += raw_vec_y[raw_index]
            raw_index += 1
        # END-WHILE
    # END-FOR

    return target_vec_2theta, target_vec_y, numpy.sqrt(target_vec_y)


class MantidHB2BReduction(object):
    """ Reducing the data using Mantid algorithm
    """

    def __init__(self, idf_file_name):
        """
        initialization
        """
        self._curr_reduced_data = None   # dict[ws name]  = vec_2theta, vec_y, vec_e

        # data workspace to reduce
        self._data_ws_name = None

        # instrument file
        checkdatatypes.check_file_name(idf_file_name, True, False, False, 'Mantid IDF XML file')
        self._mantid_idf = idf_file_name

        # about experimental data
        self._detector_2theta = None
        self._detector_counts = None
        self._detector_mask = None

        # calibration: an instance of AnglerCameraDetectorShift
        self._instrument_calibration = None

        # TODO - FUTURE - Need to find out which one, resolution or number of bins, is more essential
        self._2theta_resolution = 0.1
        self._num_bins = 1800  # NUM_BINS = [1800, 2500]

        return

    @staticmethod
    def convert_from_raw_to_2theta(det_counts_ws_name, test_mode=False):
        """ Convert workspace with detector counts workspace with unit X as
        :param det_counts_ws_name: name of input workspace
        :param test_mode: test mode.... cannot give out correct result
        :return: workspace in unit 2theta and transposed to 1-spectrum workspace (handler)
        """
        print('[DB...BAT] Report: Convert raw to 2theta')

        # check input
        checkdatatypes.check_string_variable('Input raw data workspace name', det_counts_ws_name)
        if not ADS.doesExist(det_counts_ws_name):
            raise RuntimeError('Raw data workspace {} does not exist in Mantid ADS.'.format(det_counts_ws_name))

        # convert to 2theta - counts
        matrix_ws = ADS.retrieve(det_counts_ws_name)
        print('[DB...BAT] Raw workspace: number of histograms = {}, Unit = {}'
              ''.format(matrix_ws.getNumberHistograms(), matrix_ws.getAxis(0).getUnit().unitID()))

        two_theta_ws_name = '{}_2theta'.format(det_counts_ws_name)

        ConvertSpectrumAxis(InputWorkspace=det_counts_ws_name, Target='Theta', OutputWorkspace=two_theta_ws_name,
                            EnableLogging=False, OrderAxis=False)

        # convert from N-spectra-single element to 1-spectrum-N-element
        two_theta_ws = Transpose(InputWorkspace=two_theta_ws_name, OutputWorkspace=two_theta_ws_name,
                                 EnableLogging=False)
        print('[DB.....BAT.....PROBLEM] Reduced 2theta workspace {}: num histograms = {}, sum(Y) = {}\nY: {}'
              ''.format(det_counts_ws_name, two_theta_ws.getNumberHistograms(), two_theta_ws.readY(0).sum(),
                        two_theta_ws.readY(0)))

        return two_theta_ws

    def get_pixel_positions(self, is_matrix=False, corner_center=True):
        """
        Get 3D positions of each pixel
        :param is_matrix:
        :return:
        """
        import time
        import math

        workspace = ADS.retrieve(self._data_ws_name)
        num_dets = workspace.getNumberHistograms()

        if corner_center:
            # only return 5 positions: 4 corners and center
            pos_array = numpy.ndarray(shape=(5, 3), dtype='float')

            linear_size = int(math.sqrt(num_dets))
            for i_pos, pos_tuple in enumerate([(0, 0), (0, linear_size - 1),
                                               (linear_size - 1, 0), (linear_size - 1, linear_size - 1),
                                               (linear_size / 2, linear_size / 2)]):
                i_ws = pos_tuple[0] * linear_size + pos_tuple[1]
                pos_array[i_pos] = workspace.getDetector(i_ws).getPos()
            # END-FOR

        else:
            # full list of pixels' positions
            print('[L125] Number of spectra in {} is {}'.format(self._data_ws_name, num_dets))
            pos_array = numpy.ndarray(shape=(num_dets, 3), dtype='float')

            t0 = time.time()
            for iws in range(num_dets):
                pos_array[iws] = workspace.getDetector(iws).getPos()
            tf = time.time()
            print('[L134] Time to build array of all detectors positions: {}'.format(tf - t0))
        # END-IF

        return pos_array

    def reduce_to_2theta_histogram(self, two_theta_range, two_theta_bins_number,
                                   apply_mask, is_point_data, use_mantid_histogram,
                                   vanadium_counts_array=None):
        """Histogram counts on each pixels with pixel's diffraction angle

        Parameters
        ----------
        two_theta_range : tuple
            2 tuple as min and max of 2theta
        two_theta_bins_number : int
            2theta number of bins
        apply_mask : bool
            If true and self._detector_mask has been set, the apply mask to output
        is_point_data : bool
            Flag whether the output is point data (numbers of X and Y are same)
        use_mantid_histogram : bool
            Flag to use Mantid (algorithm ResampleX) to do histogram
        vanadium_counts_array : None or numpy.ndarray
            Vanadium counts array for normalization and efficiency calibration

        Returns
        -------

        """
        # Process input arguments
        matrix_ws_name = self._data_ws_name
        two_theta_min, two_theta_max = two_theta_range
        # TODO FIXME - NOW NOW TONIGHT - #72 2-theta range is a myth!!!
        num_2theta_bins = numpy.arange(two_theta_min, two_theta_max, two_theta_bins_number).shape[0] - 1
        target_vec_2theta = None

        # convert with Axis ordered
        theta_ws = self.convert_from_raw_to_2theta(matrix_ws_name, test_mode=False)  # order Axis
        print('[L158] (Half) reduced workspace (theta): {}'.format(theta_ws.name()))

        # mask if required
        mask = None   # TODO FIXME #72 - mask = self._det_mask_vec
        if apply_mask and mask is not None:
            checkdatatypes.check_numpy_arrays('Mask vector', [mask, theta_ws.readY(0)], 1, True)
            masked_vec = theta_ws.dataY(0)
            masked_vec *= mask
        # END-IF(mask)

        # set up resolution and number of bins for re-sampling/binning
        if two_theta_min is None:
            two_theta_min = theta_ws.readX(0)[0]
        else:
            checkdatatypes.check_float_variable('Mininum 2theta for binning', two_theta_min, (-180, 180))
        if two_theta_max is None:
            two_theta_max = theta_ws.readX(0)[-1]
        else:
            checkdatatypes.check_float_variable('Maximum 2theta for binning', two_theta_max, (-180, 180))
        if two_theta_min >= two_theta_max:
            raise RuntimeError('2theta range ({}, {}) is not acceptable.'.format(two_theta_min, two_theta_max))
        if num_2theta_bins is not None:
            # checkdatatypes.check_float_variable('2theta resolution', num_2theta_bins, (0.0001, 10))
            # num_bins = int((two_theta_max - two_theta_min) / num_2theta_bins)
            num_bins = num_2theta_bins
        else:
            num_bins = self._num_bins

        # rebin
        if False:
            ResampleX(InputWorkspace=theta_ws, OutputWorkspace=theta_ws.name(), XMin=two_theta_min,
                      XMax=two_theta_max,
                      NumberBins=num_bins, EnableLogging=False)
            reduced_ws = ADS.retrieve(matrix_ws_name)
            vec_2theta = reduced_ws.readX(0)
            vec_y = reduced_ws.readY(0)
            vec_e = reduced_ws.readE(0)
        elif False:
            # proved that histogram_data == numpy.histogram
            if target_vec_2theta is None:
                raise AssertionError('In this case, target vector X shall be obtained from')
            else:
                raw_data_ws = theta_ws
                vec_2theta, vec_y, vec_e = histogram_data(raw_data_ws.readX(0), raw_data_ws.readY(0),
                                                          target_vec_2theta)
        elif True:
            # experimenting to use SortXAxis, (modified) ResampleX
            import time
            t0 = time.time()

            raw_2theta = theta_ws.readX(0)
            raw_counts = theta_ws.readY(0)
            raw_error = theta_ws.readE(0)

            # create a 1-spec workspace
            CreateWorkspace(DataX=raw_2theta, DataY=raw_counts, DataE=raw_error, NSpec=1, OutputWorkspace='prototype')

            t1 = time.time()

            # Sort X-axis
            SortXAxis(InputWorkspace='prototype', OutputWorkspace='prototype_sorted', Ordering='Ascending')

            t2 = time.time()

            # Resample
            binned = ResampleX(InputWorkspace='prototype_sorted', OutputWorkspace=theta_ws.name(),
                               XMin=two_theta_min,
                               XMax=two_theta_max,
                               NumberBins=num_bins, EnableLogging=False)

            t3 = time.time()

            print('[STAT] Create workspace: {}\n\tSort: {}\n\tResampleX: {}'
                  ''.format(t1 - t0, t2 - t0, t3 - t0))

            vec_2theta = binned.readX(0)
            vec_y = binned.readY(0)
            vec_e = binned.readY(0)

        else:
            # use numpy histogram
            raw_2theta = raw_data_ws.readX(0)
            raw_counts = raw_data_ws.readY(0)
            print('bins = {}'.format(num_bins))
            vec_y, vec_2theta = numpy.histogram(raw_2theta, bins=num_bins, range=(two_theta_min, two_theta_max),
                                                weights=raw_counts)

            # TODO - NEXT - Here is where the detector efficiency (vanadium) and Lorentzian correction step in
            # blabla .. ...

            # take care of normalization
            vec_1 = numpy.zeros(raw_counts.shape) + 1
            vec_weights, v2t = numpy.histogram(raw_2theta, bins=num_bins, range=(two_theta_min, two_theta_max),
                                               weights=vec_1)
            # correct all the zero count bins
            for i, bin_weight_i in enumerate(vec_weights):
                if bin_weight_i < 1.E-2:  # practically zero
                    vec_weights[i] = 1.E5
            # END-FOR

            # calculate uncertainties before vec Y is changed
            # process the uncertainty
            vec_e = numpy.sqrt(vec_y)

            # normalize by bin weight
            vec_y = vec_y / vec_weights
            vec_e = vec_e / vec_weights  # for example: 3 measuremnts: n, n , n, then by this e = sqrt(n/3)
        # END-IF-ELSE

        # do some study on the workspace dimension
        print('[DB...BAT] 2theta range: {}, {}; 2theta-size = {}, Y-size = {}'
              ''.format(vec_2theta[0], vec_2theta[-1], len(vec_2theta), len(vec_y)))

        # GeneratePythonScript(InputWorkspace=reduced_ws, Filename='reduce_mantid.py')
        # file_util.save_mantid_nexus(workspace_name=matrix_ws_name, file_name='debugmantid.nxs')

        return vec_2theta, vec_y, vec_e

    @staticmethod
    def _reduced_to_2theta(matrix_ws_name, num_bins=100, twotheta_min=0, twotheta_max=100):
        """
        convert to 2theta data set from event workspace
        :param matrix_ws_name:
        :return: 3-tuple: vec 2theta, vec Y and vec E
        """
        # TODO - The following section for accepting geometry calibration shall be reviewed and refactored
        # # locate calibration file
        # if raw_nexus_file_name is not None:
        #     run_date = file_util.check_creation_date(raw_nexus_file_name)
        #     try:
        #         cal_ref_id = self._calibration_manager.check_load_calibration(exp_date=run_date)
        #     except RuntimeError as run_err:
        #         err_msg = 'Unable to locate calibration file for run {} due to {}\n'.format(run_date, run_err)
        #         cal_ref_id = None
        # else:
        #     cal_ref_id = None
        #
        # # load instrument
        # if cal_ref_id is not None:
        #     self._set_geometry_calibration(
        #         matrix_ws_name, self.calibration_manager.get_geometry_calibration(cal_ref_id))

        LoadInstrument(Workspace=matrix_ws_name, InstrumentName='HB2B', RewriteSpectraMap=True)

        ConvertSpectrumAxis(InputWorkspace=matrix_ws_name, Target='Theta', OutputWorkspace=matrix_ws_name,
                            EnableLogging=False)
        Transpose(InputWorkspace=matrix_ws_name, OutputWorkspace=matrix_ws_name, EnableLogging=False)

        ResampleX(InputWorkspace=matrix_ws_name, OutputWorkspace=matrix_ws_name, XMin=twotheta_min, XMax=twotheta_max,
                  NumberBins=num_bins, EnableLogging=False)

        # Get data
        reduced_ws = mantid_helper.retrieve_workspace(matrix_ws_name)
        vec_2theta = reduced_ws.readX(0)
        vec_y = reduced_ws.readY(0)
        vec_e = reduced_ws.readE(0)

        return vec_2theta, vec_y, vec_e

    @staticmethod
    def _get_nexus_file(ipts_number, run_number):
        """
        get Nexus file (name)
        :param ipts_number:
        :param run_number:
        :return:
        """
        checkdatatypes.check_int_variable('IPTS number', ipts_number, (1, None))
        checkdatatypes.check_int_variable('Run number', run_number, (1, None))

        # check IPTS
        ipts_path = os.path.join('/HFIR/HB2B/', 'IPTS-{}'.format(ipts_number))
        if not os.path.exists(ipts_path):
            return False, 'Unable to find {}'.format(ipts_path)

        # check run number
        nexus_name = os.path.join(ipts_path, 'nexus/HB2B_{}.nxs.h5'.format(run_number))
        if not os.path.exists(nexus_name):
            return False, 'Unable to find {} under {}' \
                          ''.format('nexus/HB2B_{}.nxs.h5'.format(run_number), ipts_path)

        return True, nexus_name

    @staticmethod
    def _load_event_nexus(nexus_file_name, ws_name=False):
        """
        load event Nexus file
        :param nexus_file_name:
        :return:
        """
        if ws_name is None:
            ws_name = os.path.basename(nexus_file_name).split('.nxs')[0] + '_event'

        LoadEventNexus(Filename=nexus_file_name, OutputWorkspace=ws_name, LoadLogs=True)

        return ws_name

    @staticmethod
    def _set_geometry_calibration(ws_name, calibration_dict):
        """
        set the calibrated geometry parameter to workspace such that
        :param ws_name:
        :param calibration_dict:
        :return:
        """
        workspace = mantid_helper.retrieve_workspace(ws_name)

        # set 2theta 0
        two_theta = mantid_helper.get_log_value(workspace, 'twotheta')
        two_theta += calibration_dict['2theta_0']
        mantid_helper.set_log_value(workspace, 'twotheta', two_theta, 'degree')

        # shift parameters
        mantid_helper.set_log_value(workspace, 'shiftx', calibration_dict['shiftx'])
        mantid_helper.set_log_value(workspace, 'shifty', calibration_dict['shifty'])

        # spin...
        # TODO - 20181204 - Refer to IDF for the rest of parameters

    @staticmethod
    def _slice_mapping_scan(ws_name):
        """
        slice (event filtering) workspace by mapping scans
        :param ws_name:
        :return: a list of sliced EventWorkspaces' names
        """
        event_ws = mantid_helper.retrieve_workspace(ws_name)

        # Check whether log 'scan_index' exists
        try:
            event_ws.run().getProperty('scan_index')
        except KeyError as key_err:
            raise RuntimeError('scan_index does not exist in {}.  Failed to slice for mapping run.'
                               'FYI {}'.format(ws_name, key_err))

        # generate event filter for the integer log
        splitter_name = ws_name + '_mapping_splitter'
        info_ws_name = ws_name + '_mapping_split_info'
        GenerateEventsFilter(InputWorkspace=event_ws, OutputWorkspace=splitter_name,
                             InformationWorkspace=info_ws_name,
                             LogName='scan_index', MinimumLogValue=1, LogValueInterval=1)

        # filter events
        out_base_name = ws_name + '_split_'
        result = FilterEvents(InputWorkspace=ws_name, SplitterWorkspace=splitter_name,
                              OutputWorkspaceBaseName=out_base_name, InformationWorkspace=info_ws_name,
                              GroupWorkspaces=True)

        output_ws_names = result.OutputWorkspaceNames  # contain 'split___ws_unfiltered'

        return output_ws_names

    def get_workspace(self):
        """
        Get the handler to the workspace
        :return:
        """
        checkdatatypes.check_string_variable('Data worksapce name', self._data_ws_name)

        return ADS.retrieve(self._data_ws_name)

    def build_instrument(self, geometry_shift):
        """ Load instrument with option as calibration
        :param geometry_shift: detector position shift
        :return:
        """
        from pyrs.core import instrument_geometry
        # Get required parameters
        two_theta_value, idf_name = self._detector_2theta, self._mantid_idf
        l2_value = self._l2
        if l2_value is not None:
            raise RuntimeError('It is not clear how to set L2 to workspace\'s run properties.')

        if self._data_ws_name is None or ADS.doesExist(self._data_ws_name) is False:
            raise RuntimeError('Reduction HB2B (Mantid) has no workspace set to reduce')

        # check calibration
        if geometry_shift is not None:
            checkdatatypes.check_type('Instrument geometry shift', geometry_shift,
                                      instrument_geometry.AnglerCameraDetectorShift)
        else:
            geometry_shift = instrument_geometry.AnglerCameraDetectorShift(0., 0., 0., 0., 0., 0.)
        # END-IF

        add_2theta = True
        if add_2theta:
            print('[INFO] 2theta degree = {}'.format(two_theta_value))
            AddSampleLog(Workspace=self._data_ws_name, LogName='2theta',
                         LogText='{}'.format(two_theta_value),  # arm_length-DEFAULT_ARM_LENGTH),
                         LogType='Number Series', LogUnit='meter',
                         NumberType='Double')

        # set up sample logs
        # cal::arm
        AddSampleLog(Workspace=self._data_ws_name, LogName='cal::arm',
                     LogText='{}'.format(geometry_shift.center_shift_z),  # arm_length-DEFAULT_ARM_LENGTH),
                     LogType='Number Series', LogUnit='meter',
                     NumberType='Double')

        # cal::deltax
        AddSampleLog(Workspace=self._data_ws_name, LogName='cal::deltax',
                     LogText='{}'.format(geometry_shift.center_shift_x),
                     LogType='Number Series', LogUnit='meter',
                     NumberType='Double')
        #
        # cal::deltay
        AddSampleLog(Workspace=self._data_ws_name, LogName='cal::deltay',
                     LogText='{}'.format(geometry_shift.center_shift_y),
                     LogType='Number Series', LogUnit='meter',
                     NumberType='Double')

        # cal::roty
        AddSampleLog(Workspace=self._data_ws_name, LogName='cal::roty',
                     LogText='{}'.format(geometry_shift.rotation_y),
                     LogType='Number Series', LogUnit='degree',
                     NumberType='Double')

        # cal::flip
        AddSampleLog(Workspace=self._data_ws_name, LogName='cal::flip',
                     LogText='{}'.format(geometry_shift.rotation_x),
                     LogType='Number Series', LogUnit='degree',
                     NumberType='Double')

        # cal::spin
        AddSampleLog(Workspace=self._data_ws_name, LogName='cal::spin',
                     LogText='{}'.format(geometry_shift.rotation_z),
                     LogType='Number Series', LogUnit='degree',
                     NumberType='Double')

        # load instrument
        LoadInstrument(Workspace=self._data_ws_name,
                       Filename=idf_name,
                       InstrumentName='HB2B', RewriteSpectraMap='True')

        return

    def mask_detectors(self, mask_vector):
        """
        Mask detectors
        :param mask_vector:
        :return:
        """

    def reduce_rs_nexus(self, nexus_name, auto_mapping_check, output_dir, do_calibration,
                        allow_calibration_unavailable):
        """ reduce an HB2B nexus file
        :param nexus_name:
        :param auto_mapping_check:
        :param output_dir:
        :param do_calibration: flag to calibrate the detector
        :param allow_calibration_unavailable: if True and do_calibration is True, then when calibration file cannot
                be found, reduction will be continued with a warning.  Otherwise, an exception will be thrown
        :return:
        """
        # load data with check
        event_ws_name = self._load_event_nexus(nexus_name)

        # is it a mapping run?
        if auto_mapping_check:
            is_mapping_run = self._check_mapping_run(event_ws_name)
        else:
            is_mapping_run = False

        # slice data
        if is_mapping_run:
            event_ws_list = self._slice_mapping_scan(event_ws_name)
        else:
            event_ws_list = [event_ws_name]

        # reduce
        reduced_data_dict = dict()
        err_msg = ''
        for ws_name in event_ws_list:
            try:
                corr_data = self._reduced_to_2theta(ws_name)
                # TODO - Need to review and refactor about how to do calibration
                # if do_calibration:
                #     # calibration_dict = self.calibration_manager.get_calibration(data_file=nexus_name)
                #     # if calibration_dict is None and not allow_calibration_unavailable:
                #     #     raise RuntimeError('Unable to locate calibration file for {}'.format(nexus_name))
                #     # elif calibration_dict is None and allow_calibration_unavailable:
                #     #     err_msg + 'Unable to find calibration for {}\n'.format(nexus_name)
                #     corr_data = self._reduced_to_2theta(ws_name)  # , calibration_dict)
                # else:
                #     corr_data = self._reduced_to_2theta(ws_name)  #, None)
            except RuntimeError as run_err:
                err_msg += 'Failed to convert {} to 2theta space due to {}\n'.format(ws_name, run_err)
            else:
                reduced_data_dict[ws_name, corr_data] = corr_data
        # END-FOR

        # set to the class variable
        self._curr_reduced_data = reduced_data_dict

        # save file
        # TODO - Need to review
        # out_file_name = os.path.join(os.path.basename(nexus_name).split('.')[0], '.hdf5')
        # self.save_reduced_data(reduced_data_dict, out_file_name)

        return

    def reduce_rs_run(self, ipts_number, run_number, is_mapping, do_calibration):
        """ reduce an HB2B run
        :param ipts_number:
        :param run_number:
        :param is_mapping:
        :param do_calibration: flag to do calibration (if it exists)
        :return: (dict, str): dict[ws name] = data set, error message
        """
        # get file
        status, ret_obj = self._get_nexus_file(ipts_number, run_number)
        if status:
            nxs_file_name = ret_obj
        else:
            err_msg = ret_obj
            raise RuntimeError('Unable to reduce ITPS-{} Run {} due to {}'.format(ipts_number, run_number, err_msg))

        # load data
        event_ws_name = self._load_event_nexus(nxs_file_name)

        # chop?
        if is_mapping:
            event_ws_list = self._slice_mapping_scan(event_ws_name)
        else:
            event_ws_list = [event_ws_name]

        # reduce
        reduced_data_dict = dict()
        err_msg = ''
        for ws_name in event_ws_list:
            try:
                corr_data = self._reduced_to_2theta(ws_name)
                # TODO - Requiring review and refactoring for calibration
                # if do_calibration:
                #       #  nxs_file_name)
                # else:
                #     corr_data = self._reduced_to_2theta(ws_name)  #, None)
            except RuntimeError as run_err:
                err_msg += 'Failed to convert {} to 2theta space due to {}\n'.format(ws_name, run_err)
            else:
                reduced_data_dict[ws_name, corr_data] = corr_data
        # END-FOR

        # set to the class variable
        self._curr_reduced_data = reduced_data_dict

        return reduced_data_dict, err_msg

    def set_calibration(self, calibration):
        """
        Set the instrument calibration
        :param calibration:
        :return:
        """
        assert isinstance(calibration, AnglerCameraDetectorShift),\
            'Instrument-calibration instance {} must be of AnglerCameraDetectorShift, but not an instance ' \
            'of type {}'.format(calibration, type(calibration))

        self._instrument_calibration = calibration

        return

    def set_experimental_data(self, det_2theta_pos, l2, det_counts_vec):
        """ Set experimental data to engine for data reduction
        :param det_2theta_pos: 2theta position of detector arm
        :param l2: L2 (None to use default value)
        :param det_counts_vec: 1D array for all detector counts
        :return:
        """
        checkdatatypes.check_float_variable('Detector arm 2theta', det_2theta_pos, (-180., 180.))
        checkdatatypes.check_numpy_arrays('Detector counts', [det_counts_vec], 1, False)

        # Set
        self._detector_2theta = det_2theta_pos
        if l2 is None:
            self._l2 = None
        else:
            checkdatatypes.check_float_variable('L2', l2, (1.E-2, None))
            self._l2 = l2

        # Create workspace
        num_pixels = det_counts_vec.shape[0]
        vec_x = numpy.array([0, 1])
        vec_e = numpy.sqrt(det_counts_vec)

        # workspace name
        now64 = numpy.datetime64(str(datetime.datetime.now()))
        start = numpy.datetime64('2011-08-26T11:46:52.960756')
        tag = hash((now64 - start).astype('int')) % 100000

        self._data_ws_name = 'HB2B_{}'.format(tag)

        CreateWorkspace(DataX=vec_x, DataY=det_counts_vec, DataE=vec_e, NSpec=num_pixels,
                        OutputWorkspace=self._data_ws_name)

        return

    def set_workspace(self, ws_name):
        """ Set the workspace that is ready for reduction to 2theta
        The workspace usually is created from loading a NeXus file
        This method is complementary to set_experimental_data()
        :param ws_name:
        :return:
        """
        checkdatatypes.check_string_variable('Workspace name', ws_name)

        if ADS.doesExist(ws_name):
            self._data_ws_name = ws_name
        else:
            raise RuntimeError('Workspace {} does not exist in ADS'.format(ws_name))

        return

    def set_2theta_resolution(self, delta_two_theta):
        """
        set 2theta resolution
        :param delta_two_theta: 2theta resolution
        :return:
        """
        checkdatatypes.check_float_variable('2-theta resolution', delta_two_theta, (1.E-10, 10.))

        self._2theta_resolution = delta_two_theta

        return
