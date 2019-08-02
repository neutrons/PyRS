# Reduction engine including slicing
import os
import numpy as np
import matplotlib.image
from pyrs.utilities import checkdatatypes
from pyrs.core import workspaces
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
        self._curr_session_name = None
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

        self._curr_workspace = workspaces.HidraWorkspace()
        self._curr_session_name = session_name
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

        # Close
        project_h5_file.close()

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

        mask_id = os.path.basename(mask_file_name).split('.')[0] + '_{}'.format(hash(mask_file_name) % 100)
        print (mask_id)
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
        print ('L317 Mask dict: {}'.format(self._loaded_mask_dict.keys()))
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

    def reduce_diffraction_data(self, session_name, bin_size_2theta, use_pyrs_engine, mask):
        # TODO - NOW TONIGHT #72 - Doc, check and etc
        # mask:  mask ID or mask vector

        if session_name is None:
            workspace = self._curr_workspace
        else:
            workspace = self._session_dict[session_name]

        # mask
        # Process mask
        if mask is None:
            mask_vec = None
            mask_id = None
        elif isinstance(mask, str):
            # mask ID
            mask_vec = self.get_mask_vector(mask)
            mask_id = mask
        else:
            checkdatatypes.check_numpy_arrays('Mask', [mask], dimension=1, check_same_shape=False)
            mask_vec = mask
            mask_id = hash('{}'.format(mask_vec.min())) + hash('{}'.format(mask_vec.max())) + \
                      hash('{}'.format(mask_vec.mean()))
        # END-IF-ELSE

        # TODO - TONIGHT NOW #72 - How to embed mask information???
        for sub_run in workspace.get_subruns():
            self.reduce_to_2theta(workspace, sub_run,
                                  use_mantid_engine=not use_pyrs_engine,
                                  mask_vec_id=(mask_id, mask_vec),
                                  resolution_2theta=bin_size_2theta)

        return

    # NOTE: Refer to compare_reduction_engines_tst
    def reduce_to_2theta(self, workspace, sub_run, use_mantid_engine, mask_vec_id,
                         min_2theta=None, max_2theta=None, resolution_2theta=None):
        """
        Reduce import data (workspace or vector) to 2-theta ~ I
        :param workspace:
        :param use_mantid_engine:
        :param mask_vec_id: 2-tuple (String as ID, None or vector for Mask)
        :param min_2theta: None or user specified
        :param max_2theta: None or user specified
        :param resolution_2theta: None or user specified
        :return:
        """
        # Get the raw data
        raw_count_vec = workspace.get_raw_data(sub_run)

        # process two theta
        two_theta = workspace.get_2theta(sub_run)
        print ('[INFO] User specified 2theta = {} is converted to Mantid 2theta = {}'
               ''.format(two_theta, -two_theta))
        two_theta = -two_theta

        # Set up reduction engine and also
        if use_mantid_engine:
            # Mantid reduction engine
            reduction_engine = reduce_hb2b_mtd.MantidHB2BReduction()
            data_ws_name = reduction_engine.create_workspace()
            reduction_engine.set_workspace(data_ws_name)
            reduction_engine.load_instrument(two_theta, mantid_idf, test_calibration)
        else:
            # PyRS reduction engine
            reduction_engine = reduce_hb2b_pyrs.PyHB2BReduction(workspace.get_instrument_setup())
            reduction_engine.set_experimental_data(two_theta, raw_count_vec)
            reduction_engine.build_instrument(None)

            # TODO FIXME - NEXT - START OF DEBUG OUTPUT -------->
            # Debug output: self._pixel_matrix
            # check corners
            # test 5 spots (corner and center): (0, 0), (0, 1023), (1023, 0), (1023, 1023), (512, 512)
            pixel_1d_array = reduction_engine.get_pixel_positions(False)
            pixel_number = 2048
            pixel_locations = [(0, 0),
                               (0, pixel_number - 1),
                               (pixel_number - 1, 0),
                               (pixel_number - 1, pixel_number - 1),
                               (pixel_number / 2, pixel_number / 2)]
            for index_i, index_j in pixel_locations:
                index1d = index_i + pixel_number * index_j
                pos_python = pixel_1d_array[index1d]
                print (pos_python)
                for i in range(3):
                    print ('dir {}:  {:10f}'
                           ''.format(i, float(pos_python[i])))
                # END-FOR
            # END-FOR
            # TODO FIXME - NEXT - END OF DEBUG OUTPUT <------------

        # END-IF

        # Mask
        mask_id, mask_vec = mask_vec_id
        if mask_vec is not None:
            reduction_engine.set_mask(mask_vec)

        # Reduce
        num_bins = 500
        two_theta_range = (10, 60)
        bin_edges, hist = reduction_engine.reduce_to_2theta_histogram(num_bins, two_theta_range,
                                                                      is_point_data=True,
                                                                      use_mantid_histogram=False)

        print ('[DB...BAT] vec X shape = {}, vec Y shape = {}'.format(bin_edges.shape, hist.shape))

        # record
        workspace.set_reduced_diffraction_data(sub_run, mask_id, bin_edges, hist)

        return

    def save_project(self, project_id, output_project_file, mask_id=None):

        project_file = rs_project_file.HydraProjectFile(output_project_file, mode='a')

        for sub_run in sorted(self._reduce_data_dict[project_id].keys()):
            vec_x, vec_y = self._reduce_data_dict[project_id][sub_run][mask_id]
            project_file.add_diffraction_data(sub_run, vec_x, vec_y, '2theta')

        project_file.close()

        return

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
            io_mode = rs_project_file.HydraProjectFileMode.READWRITE
        else:
            io_mode = rs_project_file.HydraProjectFileMode.OVERWRITE
        project_file = rs_project_file.HydraProjectFile(output_name, io_mode)

        # Save
        workspace.save_reduced_diffraction_data(project_file)

        # Close
        project_file.save_hydra_project()

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
