#!/usr/bin/python
import sys
import os
from pyrs.core import reductionengine
from pyrs.utilities import checkdatatypes
from pyrs.core import calibration_file_io
from pyrs.core import mask_util
from matplotlib import pyplot as plt


# 1. implement 2theta option
# 2. add test case for .bin file
# TODO - TONIGHT - 3. synchronize calibration parameter names among XML, .cal (ascii) and .cal.h5 (hdf5) AND doc!
# 3-1: synchronize calibration parameter names among XML,
# TODO - TONIGHT - 4. add test data file of X-ray instrument


class ReductionApp(object):
    """
    Data reduction application
    """
    def __init__(self):
        """
        initialization
        """
        self._use_mantid_engine = False
        self._reduction_engine = reductionengine.HB2BReductionManager()
        self._instrument = None

        return

    @staticmethod
    def import_calibration_file(configuration_file):
        """ set up the geometry configuration file
        :param configuration_file:
        :return:
        """
        if configuration_file.lower().endswith('.h5'):
            geometry_config = calibration_file_io.import_calibration_info_file(configuration_file)
        else:
            returns = calibration_file_io.import_calibration_ascii_file(configuration_file)
            geometry_config = returns

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

    def load_raw_data(self, data_file):

        # load data
        data_id, two_theta = self._reduction_engine.load_data(data_file, target_dimension=2048,
                                                              load_to_workspace=False)

        counts_vec = self._reduction_engine.get_counts(data_id)
        print ('Counts vec:', counts_vec)
        print ('Counts shape:', counts_vec.shape)

        return counts_vec

    def mask_detectors(self, counts_vec, mask_file):

        mask_vec, two_theta, note = mask_util.load_pyrs_mask(mask_file)
        if counts_vec.shape != mask_vec.shape:
            raise RuntimeError('Counts vector and mask vector has different shpae')

        masked_counts_vec = counts_vec * mask_vec

        return masked_counts_vec

    def plot_detector_counts(self):


        return

    def reduce_proj(self, data_file, output, instrument_file, calibration_file, mask_file=None, sub_run=None):
        """
        reduce from an HB2B project file
        :param data_file:
        :param output:
        :param instrument_file:
        :param calibration_file:
        :param mask_file:
        :param sub_run:
        :return:
        """
        import shutil

        checkdatatypes.check_file_name(data_file, True, False, False, description='Input source data file')
        checkdatatypes.check_string_variable('Output file/directory', output)

        # determine output file
        base_name = os.path.basename(data_file)

        if os.path.samefile(data_file, output):
            # data file: In/Out mode
            output_file_name = data_file
        elif os.path.exists(output) and os.path.isdir(output):
            # only output directory given
            output_file_name = os.path.join(output, base_name)
            if not os.path.exists(output_file_name) or not os.path.samefile(data_file, output_file_name):
                shutil.copyfile(data_file, output_file_name)
        else:
            # given name is supposed to be the target file name
            output_file_name = output
            shutil.copyfile(data_file, output_file_name)
        # END-IF

        checkdatatypes.check_file_name(output_file_name, False, True, False, 'Output reduced hdf5 file')

        # mask file
        if mask_file is not None:
            mask_vec, mask_2theta, note = mask_util.load_pyrs_mask(mask_file)
        else:
            mask_vec = None

        # get engine first
        if instrument_file.lower().endswith('.xml'):
            use_mantid = True
        elif instrument_file.lower().endswith('.txt'):
            use_mantid = False
        else:
            raise NotImplementedError('Impossible')

        # load data
        data_id, two_theta = self._reduction_engine.load_data(data_file, sub_run=sub_run, load_to_workspace=use_mantid)

        # load instrument
        if instrument_file.lower().endswith('.xml'):
            # Mantid IDF file: use mantid engine
            self._reduction_engine.set_mantid_idf(instrument_file)
        elif instrument_file.lower().endswith('.txt'):
            # plain text instrument setup
            instrument = calibration_file_io.import_instrument_setup(instrument_file)
            self._reduction_engine.set_instrument(instrument)
        else:
            raise NotImplementedError('Impossible')

        if calibration_file:
            geom_calibration = self.import_calibration_file(calibration_file)
            self._reduction_engine.set_geometry_calibration(geom_calibration)

        # reduce
        self._reduction_engine.reduce_to_2theta(data_id=data_id,
                                                two_theta=two_theta,
                                                output_name=output_file_name,
                                                use_mantid_engine=use_mantid,
                                                mask=mask_vec)

        return

    def reduce_beta(self, data_file, output, instrument_file, two_theta, calibration_file, mask_file=None):
        """ Reduce data
        :param data_file:
        :param output:
        :param instrument_file:
        :param two_theta
        :param calibration_file:
        :param mask_file:
        :return:
        """
        checkdatatypes.check_file_name(data_file, True, False, False, description='Input source data file')
        checkdatatypes.check_string_variable('Output file/directory', output)

        if os.path.exists(output) and os.path.isdir(output):
            # user specifies a directory
            base_name = os.path.basename(data_file).split('.')[0]
            output_file_name = os.path.join(output, '{}.h5'.format(base_name))
            if output_file_name == data_file:
                raise RuntimeError('Default output file name is exactly as same as input file name {}'
                                   ''.format(data_file))
        else:
            output_file_name = output
        checkdatatypes.check_file_name(output_file_name, False, True, False, 'Output reduced hdf5 file')

        # mask file
        if mask_file is not None:
            mask_vec, mask_2theta, note = mask_util.load_pyrs_mask(mask_file)
        else:
            mask_vec = None

        # get engine first
        if instrument_file.lower().endswith('.xml'):
            use_mantid = True
        elif instrument_file.lower().endswith('.txt'):
            use_mantid = False
        else:
            raise NotImplementedError('Impossible')

        # load data
        data_id, two_th_tp = self._reduction_engine.load_data(data_file, target_dimension=2048,
                                                              load_to_workspace=use_mantid)
        print ('2theta = {} (from {}) vs {} (from user)'.format(two_th_tp, data_file, two_theta))

        # load instrument
        if instrument_file.lower().endswith('.xml'):
            # Mantid IDF file: use mantid engine
            self._reduction_engine.set_mantid_idf(instrument_file)
        elif instrument_file.lower().endswith('.txt'):
            # plain text instrument setup
            instrument = calibration_file_io.import_instrument_setup(instrument_file)
            self._reduction_engine.set_instrument(instrument)
        else:
            raise NotImplementedError('Impossible')

        if calibration_file:
            geom_calibration = self.import_calibration_file(calibration_file)
            self._reduction_engine.set_geometry_calibration(geom_calibration)

        # reduce
        self._reduction_engine.reduce_to_2theta(data_id=data_id,
                                                two_theta=two_theta,
                                                output_name=output_file_name,
                                                use_mantid_engine=use_mantid,
                                                mask=mask_vec)

        return

    def plot_reduced_data(self):

        vec_x, vec_y = self._reduction_engine.get_reduced_data()

        if vec_x.shape[0] > vec_y.shape[0]:
            print ('Shape: vec x = {}, vec y = {}'.format(vec_x.shape, vec_y.shape))
            # TODO - TONIGHT 3 - shift half bin of X to point data
            plt.plot(vec_x[:-1], vec_y)
        else:
            plt.plot(vec_x, vec_y)
        plt.show()

    def save_reduced_data(self, sub_run):

        vec_x, vec_y = self._reduction_engine.get_reduced_data()


def main(argv):
    """
    main body
    :param argv:
    :return:
    """
    print ('args: ', argv)

    if len(argv) < 3:
        print ('Auto-reducing HB2B: {} [NeXus File Name] [Target Directory] [--instrument=xray_setup.txt]'
               '[--calibration=xray_calib.txt] [--mask=mask.h5] [--engine=engine]'.format(argv[0]))
        print ('--instrument:   instrument configuration file (arm, pixel number and size')
        print ('--2theta:       2theta value if input is not an EventNeXus file')
        print ('--calibration:  instrument geometry calibration file')
        print ('--mask:         masking file (PyRS hdf5 format)')
        print ('--engine:       mantid or pyrs.  default is pyrs')
        print ('--viewraw:      viewing raw data with an option to mask (NO reduction)')
        sys.exit(-1)

    # parse input & check
    source_data_file = argv[1]
    output_dir = argv[2]

    # parse the other options
    inputs_option_dict = parse_inputs(argv[3:])

    # call for reduction
    reducer = ReductionApp()

    if inputs_option_dict['engine'] == 'mantid':
        reducer.use_mantid_engine = True
    else:
        reducer.use_mantid_engine = False

    if inputs_option_dict['no reduction']:
        counts_vec = reducer.load_raw_data(data_file=source_data_file)
        if inputs_option_dict['mask']:
            counts_vec = reducer.mask_detectors(counts_vec, mask_file=inputs_option_dict['mask'])
        counts_matrix = counts_vec.reshape((2048, 2048))
        plt.imshow(counts_matrix)
        plt.show()
    elif inputs_option_dict['instrument'] is None and source_data_file.endswith('.h5'):
        # previous
        reducer.reduce_beta(data_file=source_data_file, output=output_dir,
                            instrument_file=inputs_option_dict['instrument'],
                            two_theta=inputs_option_dict['2theta'],
                            calibration_file=inputs_option_dict['calibration'],
                            mask_file=inputs_option_dict['mask'])
        reducer.plot_reduced_data()
    elif source_data_file.endswith('.hdf5'):
        # reduce from HB2B project file
        reducer.reduce_proj(data_file=source_data_file, output=output_dir,
                            sub_run=inputs_option_dict['subrun'],
                            instrument_file=inputs_option_dict['instrument'],
                            calibration_file=inputs_option_dict['calibration'],
                            mask_file=inputs_option_dict['mask'])
        reducer.plot_reduced_data()
        reducer.save_reduced_data(sub_run=inputs_option_dict['subrun'])
    elif source_data_file.endswith('.nxs.h5'):
        # reduce from HB2B nexus file
        raise NotImplementedError('Not been implemented to reduce from NeXus file')
    else:
        print ('For non Event NeXus file {}, instrument definition must be given '
               'or file format is not supported!'.format(source_data_file))
        sys.exit(-1)
    # END-IF-ELSE

    return


def parse_inputs(arg_list):
    """
    parse input argument
    :param arg_list:
    :return:
    """
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
        elif arg_name_i == '--2theta':
            arg_options['2theta'] = float(arg_value_i)
        elif arg_name_i == '--subrun':
            arg_options['subrun'] = int(arg_value_i)
        else:
            raise RuntimeError('Argument {} is not recognized and not supported.'.format(arg_name_i))
    # END-FOR

    return arg_options


if __name__ == '__main__':
    main(sys.argv)
