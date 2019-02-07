# Reduction engine including slicing
import os
import numpy as np
import matplotlib.image
from pyrs.utilities import checkdatatypes
from pyrs.core import calibration_file_io
from pyrs.core import scandataio
from pyrs.core import reduce_hb2b_mtd
from pyrs.core import reduce_hb2b_pyrs
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
        self._data_dict = dict()

        # number of bins
        self._num_bins = 2500

        # instrument setup (for non NeXus input)
        self._instrument = None
        self._mantid_idf = None
        # calibration
        self._geometry_calibration = calibration_file_io.ResidualStressInstrumentCalibration()

        return

    @staticmethod
    def _generate_ws_name(file_name, is_nexus):
        ws_name = os.path.basename(file_name).split('.')[0]
        if is_nexus:
            # flag to show that there is no need to load instrument again
            ws_name = '{}__nexus'.format(ws_name)

        return ws_name

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
        file_type = data_file_name.split('.')[-1].lower()

        # load
        if file_type == 'tif' or file_type == 'tiff':
            # TIFF
            data_id = self._load_tif_image(data_file_name, target_dimension, rotate=True, load_to_workspace=load_to_workspace)

        elif file_type == 'bin':
            # SPICE binary
            data_id = self._load_spice_binary(data_file_name)

        elif file_type == '.h5' or file_type == '.nxs':
            # Event NeXus
            data_id = self._load_nexus(data_file_name)

        else:
            # not supported
            raise RuntimeError('File type {} from input {} is not supported.'.format(file_type, data_file_name))

        return data_id

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

        return self._data_dict[data_id][1]

    def load_calibration(self, calibration_file):
        """
        Load calibration file
        :param calibration_file:
        :return:
        """
        self._geometry_calibration = calibration_file_io.import_calibration_ascii_file(calibration_file)

        return

    def load_instrument_setup(self, instrument_setup_file):
        """
        Load ASCII instrument set up file
        :param instrument_setup_file:
        :return:
        """
        self._instrument = calibration_file_io.import_instrument_setup(instrument_setup_file)

        return

    def reduce_to_2theta(self, data_id, output_name, use_mantid_engine, mask_vector, two_theta):
        """
        Reduce import data (workspace or vector) to 2-theta ~ I
        :param data_id:
        :param output_name:
        :param use_mantid_engine:
        :param mask_vector:
        :param two_theta: 2theta value
        :return:
        """
        # check input
        checkdatatypes.check_string_variable('Data ID', data_id)
        if data_id not in self._data_dict:
            raise RuntimeError('Data ID {} does not exist in loaded data dictionary. '
                               'Current keys: {}'.format(data_id, self._data_dict.keys()))
        checkdatatypes.check_file_name(output_name, False, True, False, 'Output reduced file')

        if use_mantid_engine:
            # init mantid reducer and add workspace in ADS
            data_ws_name = self._data_dict[data_id][0]
            print ('[DB...BAT] Data Dict: {}'.format(self._data_dict[data_id]))
            mantid_reducer = reduce_hb2b_mtd.MantidHB2BReduction()
            mantid_reducer.set_workspace(data_ws_name)

            # build instrument
            mantid_reducer.load_instrument(two_theta, self._mantid_idf, self._geometry_calibration)

            # reduce data
            mantid_reducer.convert_to_2theta(data_ws_name, mask=mask_vector)

            """
            reduction_engine.reduce_rs_nexus(source_data_file, auto_mapping_check=True, output_dir=output_dir,
                                       do_calibration=True,
                                       allow_calibration_unavailable=True)
            """

        else:
            # pyrs solution: calculate instrument geometry on the fly
            python_reducer = reduce_hb2b_pyrs.PyHB2BReduction(num_rows=self._instrument.detector_rows,
                                                              num_columns=self._instrument.detector_columns,
                                                              pixel_size_x=self._instrument.pixel_size_x,
                                                              pixel_size_y=self._instrument.pixel_size_y,
                                                              arm_length=self._instrument.arm_length)

            detector_matrix = python_reducer.build_instrument(two_theta, center_shift_x='',
                                                              center_shift_y='', rot_x_flip='',
                                                              rot_y_flip='', rot_z_spin='')

            python_reducer.reduce_to_2theta_histogram(detector_matrix, counts_matrix=counts_vec,
                                                      num_bins=self._num_bins)
        # END-IF

        return

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
