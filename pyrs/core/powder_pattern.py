import os
from pyrs.core import reduction_manager
from pyrs.utilities import checkdatatypes
from pyrs.core import mask_util
from pyrs.utilities import calibration_file_io
from pyrs.utilities import rs_project_file
from matplotlib import pyplot as plt

# This is the final version of command line script to reduce HB2B data

# TODO - #84 - Overall docs & type checks


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
        self._reduction_manager.init_session(self._session)
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
        else:
            # this returns a AnglerCameraDetectorShift
            geometry_config = calibration_file_io.import_calibration_ascii_file(configuration_file)

        return geometry_config

    @property
    def use_mantid_engine(self):
        """

        :return:
        """
        return self._use_mantid_engine

    @use_mantid_engine.setter
    def use_mantid_engine(self, value):
        """ set flag to use mantid reduction engine (True) or PyRS reduction engine (False)
        :param value:
        :return:
        """
        checkdatatypes.check_bool_variable('Flag to use Mantid as reduction engine', value)

        self._use_mantid_engine = value

        return

    def load_project_file(self, data_file):
        # load data: from raw counts to reduced data
        self._hydra_ws = self._reduction_manager.load_hidra_project(data_file, True, True, True)

        self._hydra_file_name = data_file

        return

    def plot_detector_counts(self, sub_run, mask):
        """ Plot detector counts in 2D
        :param sub_run: integer for the sub run number
        :param mask: None (no mask), mask file or mask ID in file
        :return:
        """
        # Get counts (array)
        counts_vec = self.get_detector_counts(sub_run)  # TODO this method doesn't exist!

        if mask and os.path.exists(mask):
            # mask file
            counts_vec = _mask_detectors(counts_vec, mask_file=mask)
        elif mask is not None:
            # mask ID
            counts_vec = _mask_detectors(counts_vec, mask_id=mask)  # TODO mask_id is a non-existant parameter
        # pass otherwise

        # Reshape the 1D vector for plotting
        counts_matrix = counts_vec.reshape((2048, 2048))
        plt.imshow(counts_matrix)
        plt.show()

        return

    def reduce_data(self, sub_runs, instrument_file, calibration_file, mask):
        """Reduce data from HidraWorkspace
        """
        # Check inputs
        if sub_runs is None or not bool(sub_runs):  # None or empty list
            sub_runs = self._hydra_ws.get_sub_runs()
        else:
            checkdatatypes.check_list('Sub runs', sub_runs)

        # instrument file
        if instrument_file is not None:
            print('instrument file: {}'.format(instrument_file))
            # TODO - #84 - Implement

        # calibration file - WARNING the access to the calibration is radically different
        # depending on the value of this thing that is named like it is a bool
        geometry_calibration = False
        if calibration_file is not None:
            geometry_calibration =\
                calibration_file_io.import_calibration_ascii_file(geometry_file_name=calibration_file)
        # END-IF

        # mask
        if mask is not None:
            raise NotImplementedError('It has not been decided how to parse mask to auto reduction script')

        self._reduction_manager.reduce_diffraction_data(self._session, apply_calibrated_geometry=geometry_calibration,
                                                        bin_size_2theta=0.02,
                                                        use_pyrs_engine=not self._use_mantid_engine,
                                                        mask=None,
                                                        sub_run_list=sub_runs)

        return

    def plot_reduced_data(self):
        vec_x, vec_y = self._reduction_engine.get_reduced_data()  # TODO this method doesn't exist

        if vec_x.shape[0] > vec_y.shape[0]:
            print('Shape: vec x = {}, vec y = {}'.format(vec_x.shape, vec_y.shape))
            # TODO - TONIGHT 3 - shift half bin of X to point data
            plt.plot(vec_x[:-1], vec_y)
        else:
            plt.plot(vec_x, vec_y)
        plt.show()

    def save_diffraction_data(self, output_file_name=None):
        """Save reduced diffraction data to Hidra project file
        Parameters
        ----------
        output_file_name: None or str
            if None, then append result to the input file
        Returns
        -------

        """
        # Determine output file name and writing mode
        if output_file_name is None or output_file_name == self._hydra_file_name:
            file_name = self._hydra_file_name
            mode = rs_project_file.HydraProjectFileMode.READWRITE
        else:
            file_name = output_file_name
            mode = rs_project_file.HydraProjectFileMode.OVERWRITE

        # Generate project file instance
        out_file = rs_project_file.HydraProjectFile(file_name, mode)

        # Write & close
        self._hydra_ws.save_reduced_diffraction_data(out_file)
