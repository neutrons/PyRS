#!/user/bin/python
# Test class and methods implemented for reduction from raw detector counts to diffraction data
import os
import numpy
from pyrs.core import pyrscore
from pyrs.core import instrument_geometry
from matplotlib import pyplot as plt
import pytest

"""
Instrument geometry test result (X-ray): 5 corners and quick!  NO SHIFT
[-0.07092737 -0.2047      0.45817835]
dir 0:   -0.070927
dir 1:   -0.204700
dir 2:    0.458178
[-0.40628822 -0.2047      0.22335615]
dir 0:   -0.406288
dir 1:   -0.204700
dir 2:    0.223356
[-0.07092737  0.2047      0.45817835]
dir 0:   -0.070927
dir 1:    0.204700
dir 2:    0.458178
[-0.40628822  0.2047      0.22335615]
dir 0:   -0.406288
dir 1:    0.204700
dir 2:    0.223356
[ -2.38689713e-01   1.00000000e-04   3.40709893e-01]
dir 0:   -0.238690
dir 1:    0.000100
dir 2:    0.340710
"""

GoldDataXrayNoShift = {0: numpy.array([-0.07092737, -0.20470000, 0.45817835]),
                       2047 * 2048: numpy.array([-0.40628822, -0.20470000, 0.22335615]),
                       2047: numpy.array([-0.07092737,  0.2047,      0.45817835]),
                       2047 * 2048 + 2047: numpy.array([-0.40628822,  0.2047,      0.22335615]),
                       2048/2*2048+2048/2: numpy.array([-2.38689713e-01,   1.00000000e-04,   3.40709893e-01])}

GoldDataXrayWithShift = {0: numpy.array([-0.05886799, -0.26111486,  0.60563883]),
                         2047 * 2048: numpy.array([-0.39546399, -0.25236546, 0.37275487]),
                         2047: numpy.array([-0.05574015,  0.14812927,  0.61649325]),
                         2047 * 2048 + 2047: numpy.array([-0.39233615, 0.15687867,  0.38360929]),
                         2048/2 * 2048 + 2048 / 2: numpy.array([-0.22568352, -0.05201599, 0.49456983])}

"""
Instrument geometry test result (X-ray): 5 corners and quick!  WITH SHIFT
Shift:    0.1,  -0.05,  0.12
Rotation: 1.0, 0.3, -1.23
[INFO] User specified 2theta = 35.0 is converted to Mantid 2theta = -35.0
[DB...L101] Build instrumnent: 2theta = -35.0, arm = 0.416
[-0.05886799 -0.26111486  0.60563883]
dir 0:   -0.058868
dir 1:   -0.261115
dir 2:    0.605639
[-0.39546399 -0.25236546  0.37275487]
dir 0:   -0.395464
dir 1:   -0.252365
dir 2:    0.372755
[-0.05574015  0.14812927  0.61649325]
dir 0:   -0.055740
dir 1:    0.148129
dir 2:    0.616493
[-0.39233615  0.15687867  0.38360929]
dir 0:   -0.392336
dir 1:    0.156879
dir 2:    0.383609
[-0.22568352 -0.05201599  0.49456983]
dir 0:   -0.225684
dir 1:   -0.052016
dir 2:    0.494570
"""


class TestReduction(object):
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

    @staticmethod
    def generate_testing_geometry_shift():
        """
        Get a slight shift from original position to simulate the geometry calibration
        :return:
        """
        testing_shift = instrument_geometry.AnglerCameraDetectorShift(0.1, -0.05, 0.12,
                                                                      1.0, 0.3, -1.23)

        return testing_shift

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

        # Test a subset of pixels' positions
        for pixel_id in GoldDataXrayNoShift.keys():
            # print ('Test pixel {} position'.format(pixel_id))
            pixel_pos = pixel_pos_array[pixel_id]
            numpy.testing.assert_allclose(pixel_pos, GoldDataXrayNoShift[pixel_id], atol=0.000001)
        # END-FOR

        # Visual report
        print("*************************\nNo Shift Reduction Passed (Golden Xray Data)\n"
              "*************************")

        plt.plot(vec_x, vec_y)

        return

    def test_reduce_data_geometry_shift(self):
        """ Test reduction (PyRS engine) classes and methods with calibration/instrument shift
        :return:
        """
        # Geometry shift
        test_shift = self.generate_testing_geometry_shift()
        self._reduction_controller.reduce_diffraction_data(self._project_name,
                                                           two_theta_step=0.005,
                                                           pyrs_engine=True,
                                                           mask_file_name=None,
                                                           geometry_calibration=test_shift)

        # Get data and plot
        data_set = self._reduction_controller.get_diffraction_data(self._project_name, sub_run=1,
                                                                   mask=None)

        # Get the detectors' position
        reduction_engine = self._reduction_controller.reduction_manager.get_last_reduction_engine()
        pixel_pos_array = reduction_engine.get_pixel_positions(is_matrix=False)

        # Test a subset of pixels' positions
        # Test a subset of pixels' positions
        for pixel_id in GoldDataXrayWithShift.keys():
            pixel_pos = pixel_pos_array[pixel_id]
            numpy.testing.assert_allclose(pixel_pos, GoldDataXrayWithShift[pixel_id], atol=0.000001)
        # END-FOR

        # Visual report
        print("***************************\nShifted Geometry Reduction Passed (Golden Xray Data)\n"
              "***************************")

        # Plot
        vec_2theta = data_set[0]
        vec_intensity = data_set[1]
        plt.plot(vec_2theta, vec_intensity, color='red')

        return

    def test_reduce_data_calibration_more_format(self):
        """
        Test reduce data with calibration in more formats: embedded in workspace, json file
        :return:
        """
        # Test with Json file
        test_shift = self.generate_testing_geometry_shift()
        temp_json_file_name = os.path.join(os.getcwd(), 'tests/testdata/temp_calibration.json')
        test_shift.to_json(temp_json_file_name)

        self._reduction_controller.reduce_diffraction_data(self._project_name,
                                                           two_theta_step=0.005,
                                                           pyrs_engine=True,
                                                           mask_file_name=None,
                                                           geometry_calibration=temp_json_file_name)

        # Test with embedded calibration
        self._reduction_controller.reduce_diffraction_data(self._project_name,
                                                           two_theta_step=0.008,
                                                           pyrs_engine=True,
                                                           mask_file_name=None,
                                                           geometry_calibration=True)

        return

    def test_reduction_engines_consistent(self):
        """ Compare detector positions and reduced diffraction data between
        Mantid and PyRS reduction engine
        :return:
        """
        # Test with mantid engine
        idf_xml = 'data/XRAY_Definition_20190521_1342.xml'
        self._reduction_controller.reduction_manager.set_mantid_idf(idf_xml)
        if False:
            test_shift = False
        else:
            test_shift = self.generate_testing_geometry_shift()
        self._reduction_controller.reduce_diffraction_data(self._project_name,
                                                           two_theta_step=0.004,
                                                           pyrs_engine=False,
                                                           mask_file_name=None,
                                                           geometry_calibration=test_shift)
        mantid_engine = self._reduction_controller.reduction_manager.get_last_reduction_engine()
        print('[TEST] Engine name: {}'.format(mantid_engine))
        mantid_pixel_positions = mantid_engine.get_pixel_positions(is_matrix=False, corner_center=True)

        # Test with pyrs engine
        self._reduction_controller.reduce_diffraction_data(self._project_name,
                                                           0.005,
                                                           pyrs_engine=True,
                                                           mask_file_name=None,
                                                           geometry_calibration=test_shift)
        reduction_engine = self._reduction_controller.reduction_manager.get_last_reduction_engine()
        # pixels
        pyrs_pixel_positions = reduction_engine.get_pixel_positions(is_matrix=False, corner_center=True)

        # compare
        err_msg = ''
        is_wrong = False
        for i_pixel in range(5):
            err_msg += '{}:   {}   vs   {}\n'.format(i_pixel, mantid_pixel_positions[i_pixel],
                                                     pyrs_pixel_positions[i_pixel])
            err_sum = 0.
            for i_dir in range(3):
                err_sum += (mantid_pixel_positions[i_pixel][i_dir] - pyrs_pixel_positions[i_pixel][i_dir])**2
                err_msg += '\tdir {}: {}  -  {}   =  {}\n' \
                           ''.format(i_dir, mantid_pixel_positions[i_pixel][i_dir],
                                     pyrs_pixel_positions[i_pixel][i_dir],
                                     mantid_pixel_positions[i_pixel][i_dir] - pyrs_pixel_positions[i_pixel][i_dir])
            # END-FOR
            err_sum = numpy.sqrt(err_sum)
            if err_sum > 1.E-10:
                is_wrong = True
        # END-FOR
        if is_wrong:
            print('***********************\nFailure: Reduction Engine Consistency Test\n'
                  '***********************')
            print(err_msg)
        else:
            print('***********************\nPassed: Reduction Engine Consistency Test\n'
                  '***********************')
        return

    @staticmethod
    def set_mask_files(masks_list_file_name):
        """
        Read an ASCII file containing a list of masks
        :param masks_list_file_name:
        :return:
        """
        if masks_list_file_name is None:
            masks_list_file_name = ['Chi_0_Mask.xml', 'Chi_10_Mask.xml',
                                    'Chi_20_Mask.xml', 'Chi_30_Mask.xml', 'NegZ_Mask.xml']
        mask_xml_list = [os.path.join('data', xml_name) for xml_name in masks_list_file_name]

        return mask_xml_list

    # END-DEF


def test_main():
    """
    Main for test
    :return:
    """
    # Create data
    tester = TestReduction('data/Hidra_XRay_LaB6_10kev_35deg.h5')

    # Test basic reduction
    tester.test_reduce_data_basic()

    # Test calibration
    tester.test_reduce_data_geometry_shift()
    tester.test_reduce_data_calibration_more_format()

    # Engine comparison
    tester.test_reduction_engines_consistent()

    plt.show()

    return


if __name__ == '__main__':
    pytest.main()
