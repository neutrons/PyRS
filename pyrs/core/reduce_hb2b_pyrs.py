# This is a prototype reduction engine for HB2B living independently from Mantid
import numpy as np
import numpy
import calibration_file_io
from pyrs.utilities import checkdatatypes

# TODO FIXME - NIGHT - This shall be a constant value in PyHB2BReduction class object
DEFAULT_ARM_LENGTH = 0.416


class ResidualStressInstrument(object):
    """
    This is a class to define HB2B instrument geometry and related calculation
    """
    def __init__(self, instrument_setup):
        """
        initialization
        :param instrument_setup:
        """
        # check input
        checkdatatypes.check_type('Instrument setup', instrument_setup, calibration_file_io.InstrumentSetup)

        self._init_setup = instrument_setup

        self._raw_pixel_matrix = self._set_uncalibrated_pixels()  # never been used for external client
        self._pixel_matrix = None  # used by external after build_instrument

        return

    @staticmethod
    def _rotate_detector(detector_matrix, rotation_matrix):
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

    def _set_uncalibrated_pixels(self):
        """
        set up a matrix of pixels instrument on XY plane (Z=0)
        Note: this is not a useful geometry because arm length is not set
        :return:
        """
        assert self._init_setup is not None, 'Initial instrument setup is not set yet'

        # build raw instrument/pixel matrix
        num_rows = self._init_setup.detector_rows
        num_columns = self._init_setup.detector_columns
        pixel_size_x = self._init_setup.pixel_size_x
        pixel_size_y = self._init_setup.pixel_size_y
        # arm_length = self._init_setup.arm_length

        # instrument is a N x M matrix, each element has
        pixel_matrix = np.ndarray(shape=(num_rows, num_columns, 3), dtype='float')

        # set Y as different from each row
        start_y_pos = -(num_rows * 0.5 - 0.5) * pixel_size_y
        start_x_pos = (num_columns * 0.5 - 0.5) * pixel_size_x

        for row_index in range(num_rows):
            pixel_matrix[row_index, :, 1] = start_y_pos + float(row_index) * pixel_size_y
        # set X as different from each column
        for col_index in range(num_columns):
            pixel_matrix[:, col_index, 0] = start_x_pos - float(col_index) * pixel_size_x
        # set Z: zero at origin
        pixel_matrix[:, :, 2] = 0.

        return pixel_matrix

    def build_instrument(self, two_theta, instrument_calibration):
        """
        build instrument considering calibration
        :param two_theta
        :param instrument_calibration:
        :return:
        """
        # check input
        checkdatatypes.check_type('Instrument calibration', instrument_calibration,
                                  calibration_file_io.ResidualStressInstrumentCalibration)
        checkdatatypes.check_float_variable('2theta', two_theta, (None, None))

        # make a copy
        self._pixel_matrix = self._raw_pixel_matrix.copy()

        # shift center
        self._pixel_matrix[:, :, 0] += instrument_calibration.center_shift_x
        self._pixel_matrix[:, :, 1] += instrument_calibration.center_shift_y
        print ('HB2B Lower Left: {}'.format(self._pixel_matrix[0, 0]))

        # get rotation matrix at origin (for flip, spin and vertical): all data from calibration value
        rot_x_flip = instrument_calibration.rotation_x * np.pi / 180.
        rot_y_flip = instrument_calibration.rotation_y * np.pi / 180.
        rot_z_spin = instrument_calibration.rotation_z * np.pi / 180.
        calib_matrix = self.generate_rotation_matrix(rot_x_flip, rot_y_flip, rot_z_spin)
        print ('[DB...BAT] Calibration rotation matrix:\n{}'.format(calib_matrix))
        # and rotate at origin
        self._pixel_matrix = self._rotate_detector(self._pixel_matrix, calib_matrix)

        # push to +Z at length of detector arm
        self._pixel_matrix[:, :, 2] += self._init_setup.arm_length + instrument_calibration.center_shift_z

        # rotate 2theta
        two_theta_rad = two_theta * np.pi / 180.
        two_theta_matrix = self._cal_rotation_matrix_y(two_theta_rad)
        print ('[DB...BAT] 2-theta rotation matrix:\n{}'.format(two_theta_matrix))
        self._pixel_matrix = self._rotate_detector(self._pixel_matrix, two_theta_matrix)

        # # TODO - FIXME - AFTER TEST - Delete
        # # compare position
        # # test 5 spots (corner and center): (0, 0), (0, 1023), (1023, 0), (1023, 1023), (512, 512)
        # pixel_number = self._pixel_matrix.shape[0]
        # print ('HB2B instrument shape = '.format(self._pixel_matrix.shape))
        # pixel_locations = [(0, 0),
        #                    (0, pixel_number - 1),
        #                    (pixel_number - 1, 0),
        #                    (pixel_number - 1, pixel_number - 1),
        #                    (pixel_number / 2, pixel_number / 2)]
        # for index_i, index_j in pixel_locations:
        #     # print ('PyRS:   ', pixel_matrix[index_i, index_j])
        #     # print ('Mantid: ', workspace.getDetector(index_i + index_j * 1024).getPos())  # column major
        #     pos_python = self._pixel_matrix[index_i, index_j]
        #     index1d = index_i + pixel_number * index_j
        #     # pos_mantid = workspace.getDetector(index1d).getPos()
        #     print ('({}, {} / {}):   {:10s} '
        #            ''.format(index_i, index_j, index1d, 'PyRS'))
        #     diff_sq = 0.
        #     for i in range(3):
        #         # diff_sq += (float(pos_python[i] - pos_mantid[i]))**2
        #         print ('dir {}:  {:10f}'
        #                ''.format(i, float(pos_python[i])))  # float(pos_mantid[i])))
        #     # END-FOR
        #     if diff_sq > 1.E-6:
        #         raise RuntimeError('Mantid PyRS mismatch!')
        # # END-FOR
        # # END-TEST-OUTPUT

        return self._pixel_matrix

    def generate_rotation_matrix(self, rot_x_rad, rot_y_rad, rot_z_rad):
        """
        build a ration matrix with 3 orthoganol directions
        :param rot_x_rad: rotation about X-axis in rad (flip forward/backward)
        :param rot_y_rad: rotation about Y-axis in rad (vertical rotation)
        :param rot_z_rad: rotation about Z-axis in rad (spin)
        :return:
        """
        rot_x_matrix = self._cal_rotation_matrix_x(rot_x_rad)
        rot_y_matrix = self._cal_rotation_matrix_y(rot_y_rad)
        rot_z_matrix = self._cal_rotation_matrix_z(rot_z_rad)

        # print ('[DB...BAT] Rotation Matrix: X = {}, Y = {}, Z = {}'
        #        ''.format(rot_x_rad, rot_y_rad, rot_z_rad))
        # print (rot_x_matrix)
        # print (rot_y_matrix)
        # print (rot_z_matrix)

        rotation_matrix = rot_x_matrix * rot_y_matrix * rot_z_matrix

        return rotation_matrix

    @staticmethod
    def _cal_rotation_matrix_x(angle_rad):
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
    def _cal_rotation_matrix_y(angle_rad):
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
    def _cal_rotation_matrix_z(angle_rad):
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






class PyHB2BReduction(object):
    """ A class to reduce HB2B data in pure Python and numpy
    """
    def __init__(self, instrument):
        """
        initialize the instrument
        :param instrument
        """
        self._instrument = ResidualStressInstrument(instrument)

        return

    # TODO - TONIGHT - More checks!
    def build_instrument(self, two_theta, arm_length_shift, center_shift_x, center_shift_y,
                         rot_x_flip, rot_y_flip, rot_z_spin):
        """
        build an instrument
        :param two_theta: 2theta position of the detector panel.  It shall be negative to sample log value
        :param center_shift_x:
        :param center_shift_y:
        :param rot_x_flip:
        :param rot_y_flip:
        :param rot_z_spin:
        :return: 2D numpy array
        """





    # TODO - TONIGHT 0 - Migrate this method to ResidualStressInstrument
    def get_pixel_matrix(self, dimension=1):
        """
        return the pixel matrix of the instrument built
        :return:
        """
        print ('[DB...BAY] Detector Pixel Shape: {}'.format(self._hb2b.shape))

        if dimension == 1:
            a, a, b = self._hb2b.shape
            transposed = numpy.transpose(self._hb2b, (1, 0, 2))
            return transposed.reshape((a*a, b))

        return self._hb2b


    @staticmethod
    def convert_to_2theta(det_pos_matrix):
        """
        convert the pixel position matrix to 2theta
        :param det_pos_matrix:
        :return:
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

        return twotheta_matrix

    def reduce_to_2theta_histogram(self, det_pos_matrix, counts_matrix, mask, num_bins, x_range=None,
                                   is_point_data=True):
        """ convert the inputs (detector matrix and counts to 2theta histogram)
        :param det_pos_matrix:
        :param counts_matrix:
        :param mask: vector of masks
        :param num_bins:
        :param x_range: range of X value
        :return: 2-tuple (bin edges, counts in histogram)
        """
        if det_pos_matrix is None:
            det_pos_matrix = self._hb2b
        two_theta_matrix = self.convert_to_2theta(det_pos_matrix)

        # check inputs
        if x_range:
            checkdatatypes.check_tuple('X range', xrange, 2)
            if x_range[0] >= x_range[1]:
                raise RuntimeError('X range {} is not allowed'.format(x_range))

        # histogram
        vecx = two_theta_matrix.transpose().flatten()
        vecy = counts_matrix.flatten()   # in fact vec Y is flattern alraedy!
        vecy = vecy.astype('float64')  # change to integer 32

        print ('[DB...BAT] counts matrix: shape = {}, type = {}'.format(counts_matrix.shape, counts_matrix.dtype))
        if mask is not None:
            print ('[DB...BAT] mask vector; shape = {}, type = {}'.format(mask.shape, mask.dtype))
            checkdatatypes.check_numpy_arrays('Counts vector and mask vector', [counts_matrix, mask], 1, True)
            vecy *= mask

        # this is histogram data
        hist, bin_edges = np.histogram(vecx, bins=num_bins, range=x_range, weights=vecy)

        # convert to point data
        if is_point_data:
            delta_bin = bin_edges[1] - bin_edges[0]
            bin_edges += delta_bin * 0.5
            bin_edges = bin_edges[:-1]

        return bin_edges, hist
# END-CLASS
