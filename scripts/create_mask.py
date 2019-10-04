#!/usr/bin/python
# Requirements:
# Case 1:
#   - Input: Detector position and phi
#   - Task:  Generate masking/ROI according to out of plane angle phi
#
# Case 2:
#   - Input: Mantid XML ROI/Masking file(s)
#   - Task:  Generating masking/ROI according to input or binary operation between inputs
#
# Output:
#   - numpy array binary or hdf5
#
# Example:
#   1. create_mask --roi=roi_1.xml --operation='reverse' --output=mask_1.h5
#   2. create_mask --mask=mask_1.h5  --mask=mask2.xml  --operation='and'  --output=upper.h5  --show=1
#
import sys
from math import sqrt
from pyrs.utilities import checkdatatypes
from pyrs.core import mask_util
import os
import numpy
from matplotlib import pyplot as plt


class MaskProcessApp(object):
    """ Application class to process mask
    """

    def __init__(self, num_pixels=2048**2):
        """ Initialization as number of pixel
        :param num_pixels:
        """
        checkdatatypes.check_int_variable('Detector pixel number', num_pixels, (1024**2, 2048**2+1))

        self._num_pixels = num_pixels
        self._mask_array_dict = dict()
        self._mask_info_dict = dict()   # mask ID, original file, target_file

        self._2theta = None

        return

    @staticmethod
    def _generate_id(mantid_xml, is_roi):
        """ Generate the reference ID of input XML file
        :param mantid_xml:
        :return:
        """
        if is_roi:
            flag = 'roi'
        else:
            flag = 'mask'

        mask_id = '{}-{}_{}'.format(flag, os.path.basename(mantid_xml).split('.')[0],
                                    hash(os.path.dirname(mantid_xml)))

        return mask_id

    def export_mask(self, mask_id, out_file, note):
        """
        export mask to HDF5 (PyRS format)
        :param mask_id:
        :param out_file:
        :param note:
        :return:
        """
        checkdatatypes.check_file_name(out_file, False, True, False, 'Output hdf5 file name')
        checkdatatypes.check_string_variable('Mask note', note)

        mask_vec = self._mask_array_dict[mask_id]

        mask_util.save_mantid_mask(mask_vec, out_file, self._2theta, note)

        return

    def get_mask_ids(self):
        """
        Get all the mask IDs
        :return:
        """
        return self._mask_array_dict.keys()

    def process_mask_file(self, mantid_mask_xml):
        """

        :param mantid_mask_xml:
        :return:
        """
        mask_vec = mask_util.load_mantid_mask(self._num_pixels, mantid_mask_xml, is_mask=True)
        mask_id = self._generate_id(mantid_mask_xml, is_roi=False)
        self._mask_array_dict[mask_id] = mask_vec

        return mask_id

    def import_roi_file(self, roi_file_name):
        """

        :param roi_file_name:
        :return:
        """
        roi_id = self._generate_id(roi_file_name, is_roi=True)

        # import file
        if roi_file_name.lower().endswith('.xml'):
            # XML  file name
            roi_vec = mask_util.load_mantid_mask(self._num_pixels, roi_file_name, is_mask=False)
        elif roi_file_name.lower().endswith('.h5') or roi_file_name.lower().endswith('.hdf5'):
            roi_vec, two_theta, user_note = mask_util.load_pyrs_mask(roi_file_name)
        else:
            raise RuntimeError('ROI file of type {} is not recognized and supported'
                               ''.format(roi_file_name.split('.')[-1]))

        # set
        self._mask_array_dict[roi_id] = roi_vec
        self._mask_info_dict[roi_id] = [None] * 2
        self._mask_info_dict[roi_id] = roi_file_name

        return roi_id

    def reverse(self, mask_id):
        """ Revert
        :param mask_id:
        :return:
        """
        src_bit_array = self._mask_array_dict[mask_id]

        if mask_id.startswith('roi'):
            new_flag = 'mask'
        else:
            new_flag = 'roi'

        target_id = '{}-{}'.format(new_flag, mask_id.split('-'))[-1]
        target_bit_array = 1 - src_bit_array
        self._mask_array_dict[target_id] = target_bit_array
        self._mask_info_dict[target_id] = [None] * 2

        return target_id

    def operate_mask_binary(self, mask_id1, mask_id2, operation):
        """ Do an 'AND' operation to 2 masks
        :param mask_id1:
        :param mask_id2:
        :param operation:
        :return:
        """
        checkdatatypes.check_string_variable('Mask ID 1', mask_id1)
        checkdatatypes.check_string_variable('Mask ID 2', mask_id2)
        checkdatatypes.check_string_variable('Mask operation', operation, ['and', 'or'])

        mask_vec_1 = self._mask_array_dict[mask_id1]
        print('[INFO] {}: Dimension = {}'.format(mask_id1, mask_vec_1.shape))
        mask_vec_2 = self._mask_array_dict[mask_id2]
        print('[INFO] {}: Dimension = {}'.format(mask_id2, mask_vec_2.shape))

        # AND operation
        if operation == 'and':
            target_mask_vec = mask_vec_1 * mask_vec_2
        elif operation == 'or':
            target_mask_vec = mask_vec_1 + mask_vec_2
        else:
            raise RuntimeError('Impossible to get here!')

        # store
        binary_mask_id = 'mask_{}_{}_{}'.format(mask_id1, mask_id2, operation)
        self._mask_info_dict[binary_mask_id] = [None] * 2
        self._mask_array_dict[binary_mask_id] = target_mask_vec

        return binary_mask_id

    def set_2theta(self, two_theta):
        """
        set 2theta value
        :param two_theta:
        :return:
        """
        checkdatatypes.check_float_variable('Two theta', two_theta, (-180, 180))

        self._2theta = two_theta

        return

    def show_mask(self, mask_id):
        """
        plot mask and print out statistics information
        :param mask_id:
        :return:
        """
        mask_vec = self._mask_array_dict[mask_id]
        num_pixels = mask_vec.shape[0]

        num_masked = num_pixels - mask_vec.sum()
        print('{}: Pixel number = {}, Number of masked pixels = {}: {}% are Masked.'
              ''.format(mask_id, mask_vec.shape[0], num_masked, num_masked * 100. / num_pixels))

        # convert to 2D
        linear_size = int(sqrt(num_pixels))
        mask_2d_vec = mask_vec.reshape((linear_size, linear_size))
        plt.imshow(mask_2d_vec)

        # vec_x = numpy.arange(0, mask_vec.shape[0])
        # plt.plot(vec_x, mask_vec, label=mask_id)
        plt.show()

        return num_masked


def main(argv):
    """ Main argument
    :param argv:
    :return:
    """
    if len(argv) == 1:
        print('Generate masks (HDF5)\n> {} --help'.format(argv[0]))
        sys.exit(-1)

    # set up default init value
    num_pixels = 2048**2

    # parse inputs
    for iarg, arg_i in enumerate(argv):
        print(iarg, arg_i)

    result = parse_input_arguments(sys.argv[1:])
    if result is None:
        sys.exit(1)
    else:
        roi_file_list, mask_file_list, operation, two_theta, note, out_file = result

    # operation
    mask_processor = MaskProcessApp(num_pixels)

    # import files
    print('[DB...BAT] ROI files (flag2): {}'.format(roi_file_list))
    for roi_file in roi_file_list:
        mask_processor.import_roi_file(roi_file)
    for mask_file in mask_file_list:
        mask_processor.process_mask_file(mask_file)

    if two_theta is not None:
        mask_processor.set_2theta(two_theta)

    mask_id_list = mask_processor.get_mask_ids()
    if operation is None:
        # not defined... just convert XML to h5
        if len(mask_id_list) > 1:
            print('Convert to HDF5 can only take 1 file a time')

        mask_id = mask_id_list[0]
        if note is None:
            note = 'Converted to PyRS mask/roi from {}'.format(mask_id)
        mask_processor.export_mask(mask_id, out_file, note)   # all masks

        mask_processor.show_mask(mask_id)

    elif operation == 'reverse':
        mask_id_list = mask_processor.get_mask_ids()
        if len(mask_id_list) > 1:
            print('Reverse mask operation can only take 1 file a time')

        mask_id = mask_id_list[0]
        new_mask_id = mask_processor.reverse(mask_id)
        if note is None:
            note = 'Convert mask/ROI to ROI/mask for {}'.format(mask_id)
        mask_processor.export_mask(new_mask_id, out_file, note)

        # show result
        mask_processor.show_mask(new_mask_id)

    elif operation == 'and' or operation == 'or':
        if len(mask_id_list) < 2:
            print('[ERROR] Unable to do binary operation to a single mask/ROI')

        binary_mask_id = mask_processor.operate_mask_binary(mask_id_list[0], mask_id_list[1], operation)
        for i in range(2, len(mask_id_list)):
            binary_mask_id = mask_processor.operate_mask_binary(binary_mask_id, mask_id_list[i], operation)
        mask_processor.export_mask(binary_mask_id, out_file, note)

        # show result
        mask_processor.show_mask(binary_mask_id)

    else:
        print('[ERROR] Operation {} is not supported'.format(operation))
        sys.exit(-1)

    return


def print_help():
    """
    print helping information
    :return:
    """
    print('<executable> --roi=1.xml --mask=2.xml --operation=and --output=/tmp/newmask.ht --note=New_Mask_1_2')

    return


def parse_input_arguments(argv):
    """ parse input arguments
    :param argv:
    :return: None (if nothing to parse) or 5-tuple as ROI files, Mask files, Operation, 2theta, Note, Output
    """
    # init outputs
    roi_file_list = list()
    mask_file_list = list()
    operation = None
    two_theta = None
    note = ''
    out_file_name = None

    is_help = False
    for arg_i in argv:
        terms = arg_i.split('=')
        arg_name = terms[0].strip().lower()

        if arg_name == '--help':
            print_help()
            is_help = True
            break
        else:
            arg_value = terms[1]

        if arg_name == '--roi':
            roi_file_list.append(arg_value)
        elif arg_name == '--mask':
            mask_file_list.append(arg_value)
        elif arg_name == '--operation':
            operation = arg_value.lower()
        elif arg_name == '--note':
            note = arg_value.replace('_', ' ')
        elif arg_name == '--output':
            out_file_name = arg_value
        elif arg_name == '--2theta':
            two_theta = float(arg_value)
        else:
            print('[ERROR] Argument {} is not supported.'.format(arg_name))
            sys.exit(-1)
    # END-FOR

    if is_help:
        return None

    print('[DB...BAT] ROI files: {}'.format(roi_file_list))

    return roi_file_list, mask_file_list, operation, two_theta, note, out_file_name


if __name__ == '__main__':
    main(sys.argv)
