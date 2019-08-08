#!/user/bin/python
# Test class and methods implemented for reduction from raw detector counts to diffraction data
import sys
import os
import numpy
from pyrs.core import pyrscore
from pyrs.core import instrument_geometry
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

        plt.plot(vec_x, vec_y)

        return

    def test_reduce_data_calibration(self):
        """ Test reduction (PyRS engine) classes and methods with calibration/instrument shift
        :return:
        """
        # Geometry shift
        test_shift = self.generate_testing_geometry_shift()
        print ('[DB...BAT] In Reduction-Test  Shift: {}'.format(test_shift))

        self._reduction_controller.reduce_diffraction_data(self._project_name,
                                                           two_theta_step=0.005,
                                                           pyrs_engine=True,
                                                           mask_file_name=None,
                                                           geometry_calibration=test_shift)

        # Get data and plot
        data_set = self._reduction_controller.get_diffraction_data(self._project_name, sub_run=1,
                                                                   mask=None)

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
        # Test with embedded calibration]
        idf_xml = 'tests/testdata/xray_data/XRAY_Definition_20190521_1342.xml'
        self._reduction_controller.reduction_manager.set_mantid_idf(idf_xml)
        self._reduction_controller.reduce_diffraction_data(self._project_name,
                                                           two_theta_step=0.004,
                                                           pyrs_engine=False,
                                                           mask_file_name=None,
                                                           geometry_calibration=False)
        mantid_engine = self._reduction_controller.reduction_manager.get_last_reduction_engine()
        print ('[TEST] Engine name: {}'.format(mantid_engine))
        mantid_pixel_positions = mantid_engine.get_pixel_positions(is_matrix=False)

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

    # Test calibration
    tester.test_reduce_data_calibration()
    tester.test_reduce_data_calibration_more_format()

    # Engine comparison
    tester.test_reduction_engines_consistent()

    plt.show()

    return


if __name__ == '__main__':
    main()
