# TODO - NIGHT - Implement masking file generator
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
from pyrs.utilities import checkdatatypes
from pyrs.core import mask_util
import os


class MaskProcessApp(object):
    """

    """
    def __init__(self, num_pixels=2048**2):
        """

        :param num_pixels:
        """
        checkdatatypes.check_int_variable('Detector pixel number', num_pixels, (1024**2, 2048**2))

        self._num_pixels = num_pixels
        self._mask_array_dict = dict()

        return

    def _generate_id(self, mantid_xml, is_roi):
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

    def process_mask_file(self, mantid_mask_xml):
        """

        :param mantid_mask_xml:
        :return:
        """
        mask_vec = mask_util.load_mantid_mask(mantid_mask_xml, self._num_pixels, is_mask=True)
        mask_id = self._generate_id(mantid_mask_xml, is_roi=False)
        self._mask_array_dict[mask_id] = mask_vec

        return mask_id

    def process_roi_file(self, mantid_roi_xml):
        """

        :param mantid_roi_xml:
        :return:
        """
        roi_vec = mask_util.load_mantid_mask(mantid_roi_xml, self._num_pixels, is_mask=False)
        roi_id = self._generate_id(mantid_roi_xml, is_roi=True)
        self._mask_array_dict[roi_id] = roi_vec

        return roi_id

    def revert(self, mask_id):
        """ Revert
        :param mask_id:
        :return:
        """
        src_bit_array = self._mask_array_dict[mask_id]

        if src_bit_array.startswith('roi'):
            new_flag = 'mask'
        else:
            new_flag = 'roi'

        target_id = '{}-{}'.format(new_flag, mask_id.split('-'))[-1]
        target_bit_array = 1 - src_bit_array
        self._mask_array_dict[target_id] = target_bit_array

        return target_id


    def operate_mask_and(self, mask_id1, mask_id2):
        """

        :param mask_id1:
        :param mask_id2:
        :return:
        """

        return

    def show_mask(self, mask_id):
        """
        plot mask and print out statistics information
        :param mask_id:
        :return:
        """


def main(argv):
    """
    argv
    :param argv:
    :return:
    """
    # init for default
    num_pixels = 2048**2

    parse_input_arguments(sys.argv[:1])

    return


def parse_input_arguments(argv):
    """

    :param argv:
    :return:
    """
    for arg_i in argv:
        terms = arg_i.split('=')
        arg_name = terms[0].stirp().lower()

        if arg_name == '--roi':
            blabla


