# This is a prototype reduction engine for HB2B living independently from Mantid
import numpy as np
import numpy


class BuildHB2B(object):
    """ A class to build HB2B instrument
    """
    def __init__(self, num_rows, num_columns, pixel_size_x, pixel_size_y):
        """
        initialize the instrument
        :param num_rows:
        :param num_columns:
        :param pixel_size_x:
        :param pixel_size_y:
        """
        # instrument is a N x M matrix, each element has
        self._raw_hb2b = np.ndarray(shape=(num_rows, num_columns, 3), dtype='float')

        # set Y as different from each row
        start_y_pos = -(num_rows * 0.5 - 0.5) * pixel_size_y
        start_x_pos = (num_columns * 0.5 - 0.5) * pixel_size_x
        for row_index in range(num_rows):
            self._raw_hb2b[row_index, :, 1] = start_y_pos + float(row_index) * pixel_size_y
        # set X as different from each column
        for col_index in range(num_columns):
            self._raw_hb2b[:, col_index, 0] = start_x_pos - float(col_index) * pixel_size_x
        # set Z: zero at origin
        self._raw_hb2b[:, :, 2] = 0.

        # define the real HB2B
        self._hb2b = None

        return

    def build_instrument(self, arm_length, two_theta, center_shift_x, center_shift_y,
                         rot_x_flip, rot_y_flip, rot_z_spin):
        """
        build an instrument
        :param arm_length: distance from detector "center" (intercept axis) to origin
        :param two_theta: 2theta position of the detector panel
        :param center_shift_x:
        :param center_shift_y:
        :param rot_x_flip:
        :param rot_y_flip:
        :param rot_z_spin:
        :return:
        """
        # make a copy
        self._hb2b = self._raw_hb2b.copy()

        # shift center
        self._hb2b[:, :, 0] += center_shift_x
        self._hb2b[:, :, 1] += center_shift_y

        # get rotation matrix at origin (for flip, spin and vertical)
        rot_x_flip = rot_x_flip * np.pi / 180.
        rot_y_flip = rot_y_flip * np.pi / 180.
        rot_z_spin = rot_z_spin * np.pi / 180.
        calib_matrix = self.build_rotation_matrix(rot_x_flip, rot_y_flip, rot_z_spin)

        # and rotate at origin
        self._hb2b = self.rotate_instrument(self._hb2b, calib_matrix)

        # push to +Z at length of detector arm
        self._hb2b += arm_length

        # rotate 2theta
        two_theta_rad = two_theta * np.pi / 180.
        two_theta_matrix = self.build_rotation_matrix_y(two_theta_rad)
        self._hb2b = self.rotate_instrument(self._hb2b, two_theta_matrix)

        return self._hb2b

    @staticmethod
    def build_rotation_matrix_x(angle_rad):
        """
        build rotation matrix around X-axis
        :param angle_rad:
        :return:
        """
        rotate_matrix = numpy.matrix([[1., 0., 0.],
                                      [0., numpy.cos(angle_rad), -numpy.sin(angle_rad)],
                                      [0., numpy.sin(angle_rad), numpy.cos(angle_rad)]],
                                     'float')

        return rotate_matrix

    @staticmethod
    def build_rotation_matrix_y(angle_rad):
        """
        build rotation matrix around Y-axis
        :param angle_rad:
        :return:
        """
        rotate_matrix = numpy.matrix([[numpy.cos(angle_rad), 0., numpy.sin(angle_rad)],
                                      [0., 1., 0.],
                                      [-numpy.sin(angle_rad), 0., numpy.cos(angle_rad)]],
                                     'float')

        return rotate_matrix

    @staticmethod
    def build_rotation_matrix_z(angle_rad):
        """
        build rotation matrix around Z-axis
        :param angle_rad:
        :return:
        """
        rotate_matrix = numpy.matrix([[numpy.cos(angle_rad), -numpy.sin(angle_rad), 0.],
                                      [numpy.sin(angle_rad), numpy.cos(angle_rad), 0.],
                                      [0., 0., 1.]],
                                     'float')

        return rotate_matrix

    def build_rotation_matrix(self, rot_x_rad, rot_y_rad, rot_z_rad):
        """
        build a ration matrix with 3 orthoganol directions
        :param rot_x_rad: rotation about X-axis in rad (flip forward/backward)
        :param rot_y_rad: rotation about Y-axis in rad (vertical rotation)
        :param rot_z_rad: rotation about Z-axis in rad (spin)
        :return:
        """
        rotation_matrix = self.build_rotation_matrix_x(rot_x_rad) * self.build_rotation_matrix_y(rot_y_rad) * self.build_rotation_matrix_z(rot_z_rad)

        return rotation_matrix

    @staticmethod
    def rotate_instrument(detector_matrix, rotation_matrix):
        """
        rotate instrument
        :param detector_matrix:
        :param rotation_matrix:
        :return:
        """
        rotate_det = np.zeros(detector_matrix.shape, dtype='float')

        for i in range(3):
            for j in range(3):
                temp_det = rotation_matrix[i, j] * detector_matrix[:, :, j]
                rotate_det[:, :, i] += temp_det[:, :]

        return rotate_det

    @staticmethod
    def reduce_to_2theta_histogram(det_pos_matrix, counts_matrix, num_bins):
        """ convert the inputs (detector matrix and counts to 2theta histogram)
        :param det_pos_matrix:
        :param counts_matrix:
        :param num_bins:
        :return: 2-tuple (bin edges, counts in histogram)
        """
        # convert detector position matrix to 2theta
        # normalize the detector position 2D array
        det_pos_norm_matrix = np.sqrt(det_pos_matrix[:, :, 0] ** 2 + det_pos_matrix[:, :, 1] ** 2 + det_pos_matrix[:, :, 2] ** 2)
        # normalize pixel position for diffraction angle
        for i_dir in range(3):
            det_pos_matrix[:, :, i_dir] /= det_pos_norm_matrix

        # convert to  2theta in degree
        k_in_vec = [0, 0, 1]
        diff_angle_cos_matrix = det_pos_matrix[:, :, 0] * k_in_vec[0] + det_pos_matrix[:, :, 1] * k_in_vec[1] + det_pos_matrix[:, :, 2] * k_in_vec[2]
        twotheta_matrix = np.arccos(diff_angle_cos_matrix) * 180 / np.pi

        # histogram
        vecx = twotheta_matrix.flatten()
        vecy = counts_matrix.flatten()
        hist, bin_edges = np.histogram(vecx, bins=num_bins, weights=vecy)

        return bin_edges, hist
# END-CLASS


def test_main():
    """
    test main to verify algorithm
    :return:
    """