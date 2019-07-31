# Reduction engine including slicing
import os
import numpy as np
import matplotlib.image
from pyrs.utilities import checkdatatypes
from pyrs.core import datamanagers
from pyrs.utilities import calibration_file_io
from pyrs.core import mask_util
from pyrs.core import reduce_hb2b_mtd
from pyrs.core import reduce_hb2b_pyrs
from pyrs.utilities import rs_scan_io
from pyrs.utilities import rs_project_file
from mantid.simpleapi import CreateWorkspace, LoadSpiceXML2DDet, Transpose, LoadEventNexus, ConvertToMatrixWorkspace

# TODO - FIXME - Issue #72 : Clean up


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
        # self._geometry_calibration = calibration_file_io.ResidualStressInstrumentCalibration()

        # workspace name or array vector
        self._curr_workspace = None
        self._session_dict = dict()  # ID = workspace, counts vector

        # number of bins
        self._num_bins = 2500

        # instrument setup (for non NeXus input)
        self._instrument = None
        self._mantid_idf = None
        # calibration

        # record
        self._last_loaded_data_id = None

        # masks
        self._loaded_mask_files = list()
        self._loaded_mask_dict = dict()

        # # reduced data
        # self._curr_vec_x = None
        # self._curr_vec_y = None

        # raw data and reduced data
        self._raw_data_dict = dict()  # [Handler][Sub-run][Mask ID] = vec_y, 2theta
        self._reduce_data_dict = dict()  # D[data ID][mask ID] = vec_x, vec_y

        # Outputs
        self._output_directory = None

        return

    @staticmethod
    def _generate_ws_name(file_name, is_nexus):
        ws_name = os.path.basename(file_name).split('.')[0]
        if is_nexus:
            # flag to show that there is no need to load instrument again
            ws_name = '{}__nexus'.format(ws_name)

        return ws_name

    @property
    def current_data_id(self):
        """
        current or last loaded data ID
        :return:
        """
        return self._last_loaded_data_id

    def get_diffraction_pattern(self, data_handler, sub_run):
        try:
            vec_x, vec_y = self._reduce_data_dict[data_handler][sub_run][None]
        except KeyError as key_err:
            vec_x = vec_y = None

        return vec_x, vec_y

    def get_sub_runs(self, exp_handler):

        # TODO - TONIGHT - Doc and check
        return self._raw_data_dict[exp_handler].keys()

    def get_sub_run_count(self, exp_handler, sub_run):
        return self._raw_data_dict[exp_handler][sub_run][0]

    def get_sub_run_2theta(self, exp_handler, sub_run):
        return self._raw_data_dict[exp_handler][sub_run][1]

    def init_session(self, session_name):
        """
        Initialize a new session of reduction and thus to store data according to session name
        :return:
        """
        # Check inputs
        checkdatatypes.check_string_variable('Reduction session name', session_name)
        if session_name == '' or session_name in self._session_dict:
            raise RuntimeError('Session name {} is either empty or previously used (not unique)'.format(session_name))

        self._curr_workspace = datamanagers.HidraWorkspace()
        self._session_dict[session_name] = self._curr_workspace

        return

    def load_hidra_project(self, project_file_name, load_calibrated_instrument):
        """
        load hidra project file
        :param project_file_name:
        :return:
        """
        # check inputs
        checkdatatypes.check_file_name(project_file_name, True, False, False, 'Project file to load')

        # Check
        if self._curr_workspace is None:
            raise RuntimeError('Call init_session to create a ReductionWorkspace')

        # PyRS HDF5
        project_h5_file = rs_project_file.HydraProjectFile(project_file_name,
                                                           mode=rs_project_file.HydraProjectFileMode.READWRITE)

        # Load
        self._curr_workspace.load_hidra_project(project_h5_file,
                                                load_raw_counts=True,
                                                load_reduced_diffraction=False)

        # # Load sample logs
        # sub_run_list = project_h5_file.get_sub_runs()
        # for sub_run in sorted(sub_run_list):
        #     # count
        #     count_vec = project_h5_file.get_scan_counts(sub_run=sub_run)
        #     self._curr_workspace.add_sub_run(sub_run, count_vec)
        # # END-FOR
        #
        # # Get sample logs
        # two_theta_log = project_h5_file.get_log_value(log_name='2Theta')
        # self._curr_workspace.add_log('2Theta', two_theta_log)
        #
        # # Instrument
        # instrument_setup = project_h5_file.get_instrument_geometry(load_calibrated_instrument)
        # self._curr_workspace.set_instrument(instrument_setup)
        #
        # project_h5_file.close()

        return

    def load_instrument_file(self, instrument_file_name):
        """
        Load instrument (setup) file
        :param instrument_file_name:
        :return:
        """
        # Check
        if self._curr_workspace is None:
            raise RuntimeError('Call init_session to create a ReductionWorkspace')

        instrument = calibration_file_io.import_instrument_setup(instrument_file_name)
        self._curr_workspace.set_instrument(instrument)

        return

    def load_mask_file(self, mask_file_name):
        """ Load mask file
        :param mask_file_name:
        :return:
        """
        mask_vec, two_theta, note = mask_util.load_pyrs_mask(mask_file_name)

        # register the masks
        self._loaded_mask_files.append(mask_file_name)

        mask_id = os.path.basename(mask_file_name).split('.')[0] + '_{}'.format(hash(mask_file_name))
        self._loaded_mask_dict[mask_id] = mask_vec, two_theta, mask_file_name

        return two_theta, note, mask_id

    def _load_nexus(self, nxs_file_name):
        """
        Load NeXus file
        :param nxs_file_name:
        :return:
        """
        out_ws_name = self._generate_ws_name(nxs_file_name)
        LoadEventNexus(Filename=nxs_file_name, OutputWorkspace=out_ws_name)

        # get vector of counts
        CovertToMatrixWorkspace(InputWorkspace=out_ws_name, OutputWorkspace='_temp')
        count_ws = Transpose(InputWorkspace='_temp', OutputWorkspace='_temp')
        count_vec = count_ws.readY(0)

        self._data_dict[out_ws_name] = [out_ws_name, count_vec]

        return out_ws_name

    def _load_pyrs_h5(self, pyrs_h5_name, sub_run, create_workspace):
        """ Load Reduced PyRS file in HDF5 format
        :param pyrs_h5_name:
        :return:
        """
        # check input
        checkdatatypes.check_string_variable('PyRS reduced file (.hdf5)', pyrs_h5_name)
        checkdatatypes.check_file_name(pyrs_h5_name, True, False, False, 'PyRS reduced file (.hdf5)')

        # using base name (without postfix) with hashed directory as ID (unique)
        data_id = os.path.basename(pyrs_h5_name).split('.h')[0] + '_{}'.format(hash(os.path.dirname(pyrs_h5_name)))

        # load file
        if pyrs_h5_name.endswith('hdf5'):
            # start file and load
            diff_file = rs_project_file.HydraProjectFile(pyrs_h5_name, mode='r')
            count_vec = diff_file.get_raw_counts(sub_run=sub_run)
            two_theta = diff_file.get_log_value(log_name='2Theta', sub_run=sub_run)
            # close file
            diff_file.close()
        else:
            diff_file = rs_scan_io.DiffractionDataFile()
            count_vec, two_theta = diff_file.load_raw_measurement_data(pyrs_h5_name)

        # create workspace for counts as an option
        if create_workspace:
            vec_x = np.zeros(count_vec.shape)
            ws = CreateWorkspace(DataX=vec_x, DataY=count_vec, DataE=np.sqrt(count_vec), NSpec=vec_x.shape[0],
                                 OutputWorkspace=data_id, UnitX='SpectraNumber')
            # ws = Transpose(data_id, OutputWorkspace=data_id)
            print ('[DB.......BAT......LOOK] Unit of workspace = {}'.format(ws.getAxis(0).getUnit().unitID()))

        self._data_dict[data_id] = [data_id, count_vec]

        return data_id, two_theta

    def _load_spice_binary(self, bin_file_name):
        """ Load SPICE binary
        :param bin_file_name:
        :return:
        """
        ws_name = self._generate_ws_name(bin_file_name, is_nexus=False)
        LoadSpiceXML2DDet(Filename=bin_file_name, OutputWorkspace=ws_name, LoadInstrument=False)

        # get vector of counts
        counts_ws = Transpose(InputWorkspace=ws_name, OutputWorkspace='_temp')
        count_vec = counts_ws.readY(0)

        self._data_dict[ws_name] = [ws_name, count_vec]

        return ws_name

    def _load_tif_image(self, raw_tiff_name, pixel_size, rotate, load_to_workspace):
        """
        Load data from TIFF
        It is same as pyrscalibration.load_data_from_tiff
        Create numpy 2D array with integer 32
        :param raw_tiff_name:
        :param pixel_size: linear pixel size (N) image is N x N
        :param rotate:
        :return:
        """
        # Load data from TIFF
        image_2d_data = matplotlib.image.imread(raw_tiff_name)
        image_2d_data.astype(np.int32)  # to an N x N data
        if rotate:
            image_2d_data = image_2d_data.transpose()

        if image_2d_data.shape[0] != 2048:
            raise RuntimeError('Current algorithm can only handle 2048 x 2048 TIFF but not of size {}'
                               ''.format(image_2d_data.shape))
        # Merge data if required
        if pixel_size == 1024:
            counts_vec = image_2d_data[::2, ::2] + image_2d_data[::2, 1::2] + image_2d_data[1::2, ::2] + image_2d_data[
                                                                                                         1::2, 1::2]
            pixel_type = '1K'
            # print (DataR.shape, type(DataR))
        else:
            # No merge
            counts_vec = image_2d_data
            pixel_type = '2K'

        counts_vec = counts_vec.reshape((pixel_size * pixel_size,))
        data_id = self._generate_ws_name(raw_tiff_name, is_nexus=False)
        self._data_dict[data_id] = [None, counts_vec]

        if load_to_workspace:
            data_ws_name = '{}_{}'.format(data_id, pixel_type)
            CreateWorkspace(DataX=np.zeros((pixel_size ** 2,)), DataY=counts_vec, DataE=np.sqrt(counts_vec),
                            NSpec=pixel_size ** 2,
                            OutputWorkspace=data_ws_name, VerticalAxisUnit='SpectraNumber')
            self._data_dict[data_id][0] = data_ws_name

        print ('[DB...BAT] Loaded TIF and return DataID = {}'.format(data_id))

        return data_id

    def get_counts(self, data_id):
        """
        Get the array as detector counts
        :param data_id:
        :return:
        """
        if data_id not in self._data_dict:
            raise RuntimeError('Data key {} does not exist in loaded data dictionary (keys are {})'
                               ''.format(data_id, self._data_dict.keys()))
        return self._data_dict[data_id][1]

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
        return self._loaded_mask_dict[mask_id][0]

    def get_raw_data(self, data_id, is_workspace):
        """
        Get the raw data
        :param data_id:
        :param is_workspace: True, workspace; False: vector
        :return:
        """
        if is_workspace:
            return self._data_dict[data_id][0]

        return self._data_dict[data_id][1]

    def set_geometry_calibration(self, geometry_calibration):
        """
        Load calibration file
        :param geometry_calibration:
        :return:
        """
        checkdatatypes.check_type('Geometry calibration', geometry_calibration,
                                  calibration_file_io.ResidualStressInstrumentCalibration)
        self._geometry_calibration = geometry_calibration

        return

    # TODO - TONIGHT 0 - This script does not work correctly! Refer to compare_reduction_engines_tst
    def reduce_to_2theta(self, data_id, sub_run, use_mantid_engine, mask, two_theta,
                         min_2theta=None, max_2theta=None, resolution_2theta=None):
        """
        Reduce import data (workspace or vector) to 2-theta ~ I
        :param data_id:
        :param use_mantid_engine:
        :param mask: mask ID or mask vector
        :param two_theta: 2theta value
        :param min_2theta: None or user specified
        :param max_2theta: None or user specified
        :param resolution_2theta: None or user specified
        :return:
        """
        # check input
        checkdatatypes.check_string_variable('Data ID', data_id)
        if sub_run is None:
            # single run .h5 case
            if data_id not in self._data_dict:
                raise RuntimeError('Data ID {} does not exist in loaded data dictionary. '
                                   'Current keys: {}'.format(data_id, self._data_dict.keys()))
        else:
            if data_id not in self._raw_data_dict or sub_run not in self._raw_data_dict[data_id]:
                raise RuntimeError('Project ID {} Sub-run {} does not exist in raw data dictionary'
                                   ''.format(data_id, sub_run))

        # about mask
        if mask is None:
            mask_vec = None
            mask_id = None
        elif isinstance(mask, str):
            # mask ID
            mask_vec = self.get_mask_vector(mask)
            mask_id = mask
        else:
            mask_vec = mask
            mask_id = hash('{}'.format(mask_vec.min())) + hash('{}'.format(mask_vec.max())) + hash('{}'.format(mask_vec.mean()))

        # process two theta
        print ('[INFO] User specified 2theta = {} is converted to Mantid 2theta = {}'
               ''.format(two_theta, -two_theta))
        two_theta = -two_theta

        if use_mantid_engine:
            # init mantid reducer and add workspace in ADS
            data_ws_name = self._data_dict[data_id][0]
            mantid_reducer = reduce_hb2b_mtd.MantidHB2BReduction()
            mantid_reducer.set_workspace(data_ws_name)

            # build instrument
            mantid_reducer.load_instrument(two_theta, self._mantid_idf, self._geometry_calibration)

            # reduce data
            r = mantid_reducer.reduce_to_2theta(data_ws_name, mask=mask_vec,
                                                two_theta_min=min_2theta, two_theta_max=max_2theta,
                                                num_2theta_bins=resolution_2theta)

            self._curr_vec_x = r[0]
            self._curr_vec_y = r[1]

        else:
            # pyrs solution: calculate instrument geometry on the fly
            python_reducer = reduce_hb2b_pyrs.PyHB2BReduction(self._instrument)

            pixel_matrix = python_reducer.build_instrument(two_theta,
                                                           arm_length_shift=self._geometry_calibration.center_shift_z,
                                                           center_shift_x=self._geometry_calibration.center_shift_x,
                                                           center_shift_y=self._geometry_calibration.center_shift_y,
                                                           rot_x_flip=self._geometry_calibration.rotation_x,
                                                           rot_y_flip=self._geometry_calibration.rotation_y,
                                                           rot_z_spin=self._geometry_calibration.rotation_z)
            # 2 different cases to access raw data
            if sub_run is None:
                counts_vec = self._data_dict[data_id][1]
            else:
                counts_vec = self._raw_data_dict[data_id][sub_run][0]

            bin_edges, hist = python_reducer.reduce_to_2theta_histogram(counts_array=counts_vec,
                                                                        mask=mask_vec,
                                                                        num_bins=self._num_bins,
                                                                        x_range=None, is_point_data=True,
                                                                        use_mantid_histogram=False)
            self._curr_vec_x = bin_edges
            self._curr_vec_y = hist
            print ('[DB...BAT] vec X shape = {}, vec Y shape = {}'.format(bin_edges.shape,
                                                                          hist.shape))
        # END-IF

        # record
        if data_id not in self._reduce_data_dict:
            self._reduce_data_dict[data_id] = dict()
        if sub_run is None:
            # single run mode
            self._reduce_data_dict[data_id][mask_id] = self._curr_vec_x, self._curr_vec_y
        else:
            # project file mode
            self._reduce_data_dict[data_id][sub_run] = dict()
            self._reduce_data_dict[data_id][sub_run][mask_id] = self._curr_vec_x, self._curr_vec_y

        return

    def save_project(self, project_id, output_project_file, mask_id=None):

        project_file = rs_project_file.HydraProjectFile(output_project_file, mode='a')

        for sub_run in sorted(self._reduce_data_dict[project_id].keys()):
            vec_x, vec_y = self._reduce_data_dict[project_id][sub_run][mask_id]
            project_file.add_diffraction_data(sub_run, vec_x, vec_y, '2theta')

        project_file.close()

        return

    # TODO - TONIGHT 0 - From here!
    def save_reduced_diffraction(self, data_id, output_name):
        checkdatatypes.check_file_name(output_name, False, True, False, 'Output reduced file')

        print ('data id: ', data_id)
        print ('masks: ', self._reduce_data_dict[data_id].keys())
        # self._reduce_data_dict[data_id][mask_id] = self._curr_vec_x, self._curr_vec_y


        return

    # TODO - TONIGHT 0 - Need to register reduced data with sub-run
    def get_reduced_data(self, data_id=None, mask_id=None):
        """
        Get the reduce data
        :param data_id:
        :param mask_id:  ID of mask
        :return:
        """
        # default (no data ID) is the currently reduced 2theta pattern
        if data_id is None:
            return self._curr_vec_x, self._curr_vec_y

        # TODO - TONIGHT 0 - ASAP: How to store previously reduced data (different masks as use cases)
        return self._reduced_data_dict[data_id][mask_id]

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

        return

    def set_instrument(self, instrument):
        """
        set the instrument configuration
        :param instrument:
        :return:
        """
        checkdatatypes.check_type('Instrument setup', instrument, calibration_file_io.InstrumentSetup)
        self._instrument = instrument

        return

    def set_output_dir(self, output_dir):
        """
        set the directory for output data
        :param output_dir:
        :return:
        """
        # FIXME - check whether the output dir exist;

        self._output_directory = output_dir

        return

# END-CLASS-DEF

def get_log_value(workspace, log_name):
    """
    get log value from workspace
    :param workspace:
    :param log_name:
    :return:
    """
    # TODO - 20181204 - Implement!

    return blabla


def set_log_value(workspace, log_name, log_value):
    """
    set a value to a workspace's sample logs
    :param workspace:
    :param log_name:
    :param log_value:
    :return:
    """
    # TODO - 20181204 - Implement!

    return


def retrieve_workspace(ws_name, must_be_event=True):
    """
    retrieve workspace
    :param ws_name:
    :param must_be_event: throw if not event workspace if this is specified
    :return:
    """
    checkdatatypes.check_string_variable('Workspace name', ws_name)
    if ws_name == '':
        raise RuntimeError('Workspace name cannot be an empty string')

    if not ADS.doesExist(ws_name):
        raise RuntimeError('Worksapce {} does not exist in Mantid ADS'.format(ws_name))

    return ADS.retrieve(ws_name)
