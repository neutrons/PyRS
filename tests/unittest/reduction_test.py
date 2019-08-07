#!/user/bin/python
# Test class and methods implemented for reduction from raw detector counts to diffraction data
import sys
import os
import numpy
from pyrs.core import pyrscore
from pyrs.utilities import script_helper
from matplotlib import pyplot as plt


class ReductionTest(object):
    """
    Reduction Tester
    """
    def __init__(self, input_file_name):
        """
        Init
        """
        # Create calibration control
        self._reduction_controller = pyrscore.PyRsCore()

        # Load data
        self._project_name = 'calibration xray'
        self._reduction_controller.load_hidra_project(input_file_name, project_name=self._project_name)

        return

    def test_reduce_data_basic(self):
        """
        Test data without mask without instrument geometry shift
        :return:
        """
        self._reduction_controller.reduce_diffraction_data(self._project_name,
                                                           0.005, True, None)

        # get handler on reduciton engine
        reduction_engine = self._reduction_controller.reduction_manager.get_last_reduction_engine()
        # pixels
        pixel_pos_array = reduction_engine.get_pixel_positions(is_matrix=False)
        pixel_2theta_array = reduction_engine.instrument.get_pixels_2theta(dimension=1)
        # create a 1-vector
        one_count_vec = numpy.zeros(shape=(pixel_pos_array.shape[0],), dtype='float') + 1
        # output 2theta
        histogram_2theta_vec = numpy.arange(5, 65, 0.005)
        vec_x, vec_y = reduction_engine.histogram_by_numpy(pixel_2theta_array, one_count_vec, histogram_2theta_vec,
                                                           is_point_data=True, norm_bins=True)

        plt.plot(vec_x, vec_y)
        plt.show()

        return

    def test_reduce_data_calibration(self):
        # TODO - TONIGHT #72 - ASAP
        return

    def test_reduction_engines_consistent(self):
        # TODO - TONIGHT $72 - ASAP
        return

    def set_mask_files(self, masks_list_file_name):
        """
        Read an ASCII file containing a list of masks
        :param masks_list_file_name:
        :return:
        """
        temp_list = ['Chi_0_Mask.xml', 'Chi_10_Mask.xml',
                     'Chi_20_Mask.xml', 'Chi_30_Mask.xml', 'NegZ_Mask.xml']
        mask_xml_list = [os.path.join('tests/testdata/masks', xml_name) for xml_name in temp_list]

        return mask_xml_list


def main():
    """
    Main for test
    :return:
    """
    # Create data
    tester = ReductionTest('tests/testdata/Hidra_XRay_LaB6_10kev_35deg.hdf')

    # Test basic reduction
    tester.test_reduce_data_basic()

    return


if __name__ == '__main__':
    main()
