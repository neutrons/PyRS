# Reduction engine including slicing
import os
import numpy as np
import matplotlib.image
from pyrs.utilities import checkdatatypes
from pyrs.core import calibration_file_io
from pyrs.core import mask_util
from pyrs.core import reduce_hb2b_mtd
from pyrs.core import reduce_hb2b_pyrs
from pyrs.utilities import rs_scan_io
from mantid.simpleapi import CreateWorkspace, LoadSpiceXML2DDet, Transpose, LoadEventNexus, ConvertToMatrixWorkspace


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
        # calibration manager
        self._calibration_manager = calibration_file_io.CalibrationManager()

        # workspace name or array vector
        self._data_dict = dict()   # ID = workspace, counts vector

        # number of bins
        self._num_bins = 2500

        # instrument setup (for non NeXus input)
        self._instrument = None
        self._mantid_idf = None
        # calibration
        self._geometry_calibration = calibration_file_io.ResidualStressInstrumentCalibration()

        # record
        self._last_loaded_data_id = None

        # masks
        self._loaded_mask_files = list()
        self._loaded_mask_dict = dict()

        # reduced data
        self._curr_vec_x = None
        self._curr_vec_y = None

        self._reduce_data_dict = dict()   # D[data ID][mask ID] = vec_x, vec_y

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

    def load_data(self, data_file_name, target_dimension=None, load_to_workspace=True):
        """
        Load data set and
        - determine the instrument size (for PyHB2BReduction and initialize the right one if not created)
        :param data_file_name:
        :param target_dimension: if TIFF, target dimension will be used to bin the data
        :param load_to_workspace: if TIFF, option to create a workspace
        :return: data ID to look up
        """
        # check inputs
        checkdatatypes.check_file_name(data_file_name, True, False, False, 'Data file to load')

        # check file type
        if data_file_name.endswith('.nxs.h5'):
            file_type = 'nxs.h5'
        else:
            file_type = data_file_name.split('.')[-1].lower()

        # load
        if file_type == 'tif' or file_type == 'tiff':
            # TIFF
            data_id = self._load_tif_image(data_file_name, target_dimension, rotate=True,
                                           load_to_workspace=load_to_workspace)

        elif file_type == 'bin':
            # SPICE binary
            data_id = self._load_spice_binary(data_file_name)

        elif file_type == 'hdf5' or file_type == 'h5':
            # PyRS HDF5
            data_id = self._load_pyrs_h5(data_file_name, True)

        elif file_type == 'nxs.h5' or file_type == 'nxs':
            # Event NeXus
            data_id = self._load_nexus(data_file_name)

        else:
            # not supported
            raise RuntimeError('File type {} from input {} is not supported.'.format(file_type, data_file_name))

        self._last_loaded_data_id = data_id

        return data_id

    # TODO - TONIGHT 4 - Better!
    def load_instrument_file(self, instrument_file_name):
        """
        Load instrument (setup) file
        :param instrument_file_name:
        :return:
        """
        instrument = calibration_file_io.import_instrument_setup(instrument_file_name)
        self.set_instrument(instrument)

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

    def _load_pyrs_h5(self, pyrs_h5_name, create_workspace):
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

        return data_id

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

    def reduce_to_2theta(self, data_id, output_name, use_mantid_engine, mask, two_theta,
                         min_2theta=None, max_2theta=None, resolution_2theta=None):
        """
        Reduce import data (workspace or vector) to 2-theta ~ I
        :param data_id:
        :param output_name:
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
        if data_id not in self._data_dict:
            raise RuntimeError('Data ID {} does not exist in loaded data dictionary. '
                               'Current keys: {}'.format(data_id, self._data_dict.keys()))
        if output_name:
            checkdatatypes.check_file_name(output_name, False, True, False, 'Output reduced file')

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
                                                two_theta_resolution=resolution_2theta)

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

            bin_edges, hist = python_reducer.reduce_to_2theta_histogram(pixel_matrix,
                                                                        counts_matrix=self._data_dict[data_id][1],
                                                                        mask=mask_vec,
                                                                        num_bins=self._num_bins)
            self._curr_vec_x = bin_edges
            self._curr_vec_y = hist
            print ('[DB...BAT] vec X shape = {}, vec Y shape = {}'.format(bin_edges.shape,
                                                                          hist.shape))
        # END-IF

        # record
        if data_id not in self._reduce_data_dict:
            self._reduce_data_dict[data_id] = dict()
        self._reduce_data_dict[data_id][mask_id] = self._curr_vec_x, self._curr_vec_y

        return

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
