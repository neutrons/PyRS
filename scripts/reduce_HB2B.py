#!/usr/bin/python
import sys
import os
from pyrs.core import reduction_manager
from pyrs.utilities import checkdatatypes
from pyrs.core import mask_util
from pyrs.utilities import calibration_file_io
from pyrs.utilities import rs_project_file
from matplotlib import pyplot as plt

# This is the final version of command line script to reduce HB2B data
# including
# 1.

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
        if sub_runs is None:
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
                                                        bin_size_2theta=0.05,
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

        return

######################################################################
# Command line code
######################################################################


def _main(argv):
    """
    main body
    """
    if len(argv) < 3:
        _print_help(argv[0])
        sys.exit(-1)

    # parse input & check
    project_data_file = argv[1]
    output_dir = argv[2]

    # parse the other options
    inputs_option_dict = parse_inputs(argv[3:])

    # call for reduction
    reducer = ReductionApp(bool(inputs_option_dict['engine'] == 'mantid'))

    # Load Hidra project file
    reducer.load_project_file(project_data_file)

    # sub run
    if inputs_option_dict['subrun'] is None:
        sub_run_list = None
    else:
        sub_run_list = [inputs_option_dict['subrun']]

    # Process data
    if inputs_option_dict['no reduction']:
        # interpret None to be first subrun
        if sub_run_list is None:
            sub_run_list = [0]
        # plot raw detector counts without reduction but possibly with masking
        reducer.plot_detector_counts(sub_run=sub_run_list[0], mask=inputs_option_dict['mask'])
    else:
        # reduce data
        # collect information
        user_instrument = inputs_option_dict['instrument']
        user_calibration = inputs_option_dict['calibration']
        mask = inputs_option_dict['mask']

        # reduce
        reducer.reduce_data(instrument_file=user_instrument,
                            calibration_file=user_calibration,
                            mask=mask,
                            sub_runs=sub_run_list)

        # save
        out_file_name = os.path.join(output_dir, os.path.basename(project_data_file))
        reducer.save_diffraction_data(out_file_name)


def parse_inputs(arg_list):
    """
    parse input argument
    """
    # TODO - #84 - Try to use argparser (https://docs.python.org/3/library/argparse.html) to replace
    arg_options = {'instrument': None,
                   'calibration': None,
                   'mask': None,    # no masks
                   'engine': 'pyrs',
                   'no reduction': False,
                   'subrun': None,  # all sub runs
                   '2theta': None   # auto 2theta
                   }

    for arg_i in arg_list:
        terms = arg_i.split('=')
        arg_name_i = terms[0].strip().lower()
        arg_value_i = terms[1].strip()

        if arg_name_i == '--instrument':
            arg_options['instrument'] = arg_value_i
        elif arg_name_i == '--calibration':
            arg_options['calibration'] = arg_value_i
        elif arg_name_i == '--mask':
            arg_options['mask'] = arg_value_i
        elif arg_name_i == '--viewraw':
            arg_options['no reduction'] = bool(int(arg_value_i))
        elif arg_name_i == '--subrun':
            arg_options['subrun'] = int(arg_value_i)
        else:
            raise RuntimeError('Argument {} is not recognized and not supported.'.format(arg_name_i))
    # END-FOR

    return arg_options


def _print_help(argv):
    """
    print help information
    """
    print('Auto-reducing HB2B: {} [NeXus File Name] [Target Directory] [--instrument=xray_setup.txt]'
          '[--calibration=xray_calib.txt] [--mask=mask.h5] [--engine=engine]'.format(argv[0]))
    print('--instrument:   instrument configuration file overriding embedded (arm, pixel number and size')
    print('--calibration:  instrument geometry calibration file overriding embedded')
    print('--mask:         masking file (PyRS hdf5 format) or mask name')
    print('--engine:       mantid or pyrs.  default is pyrs')
    print('--viewraw:      viewing raw data with an option to mask (NO reduction)')
    print('--')


if __name__ == '__main__':
    _main(sys.argv)
