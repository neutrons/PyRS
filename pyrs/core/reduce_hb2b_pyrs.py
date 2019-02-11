# This is a prototype reduction engine for HB2B living independently from Mantid
import numpy as np
import numpy
import calibration_file_io
from pyrs.utilities import checkdatatypes

# TODO FIXME - NIGHT - This shall be a constant value in PyHB2BReduction class object
DEFAULT_ARM_LENGTH = 0.416


class PyHB2BReduction(object):
    """ A class to reduce HB2B data in pure Python and numpy
    """
    def __init__(self, instrument):
        """
        initialize the instrument
        :param instrument
        """
        # check input
        checkdatatypes.check_type('Instrument setup', instrument, calibration_file_io.InstrumentSetup)

        num_rows = instrument.detector_rows
        num_columns = instrument.detector_columns
        pixel_size_x = instrument.pixel_size_x
        pixel_size_y = instrument.pixel_size_y
        self._arm_length = instrument.arm_length

        # instrument is a N x M matrix, each element has
        self._raw_hb2b = np.ndarray(shape=(num_rows, num_columns, 3), dtype='float')

        # set Y as different from each row
        start_y_pos = -(num_rows * 0.5 - 0.5) * pixel_size_y
        start_x_pos = (num_columns * 0.5 - 0.5) * pixel_size_x

        # print ('Start X Position: ', start_x_pos)
        # print ('Start Y Position: ', start_y_pos)

        for row_index in range(num_rows):
            self._raw_hb2b[row_index, :, 1] = start_y_pos + float(row_index) * pixel_size_y
        # set X as different from each column
        for col_index in range(num_columns):
            self._raw_hb2b[:, col_index, 0] = start_x_pos - float(col_index) * pixel_size_x
        # set Z: zero at origin
        self._raw_hb2b[:, :, 2] = 0.

        # print ('HB2B[0, 0] = {}'.format(self._raw_hb2b[0, 0]))

        # define the real HB2B
        self._hb2b = None

        return

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
        :return:
        """
        # make a copy
        self._hb2b = self._raw_hb2b.copy()

        # shift center
        self._hb2b[:, :, 0] += center_shift_x
        self._hb2b[:, :, 1] += center_shift_y
        print ('HB2B Lower Left: {}'.format(self._hb2b[0, 0]))

        # get rotation matrix at origin (for flip, spin and vertical)
        rot_x_flip = rot_x_flip * np.pi / 180.
        rot_y_flip = rot_y_flip * np.pi / 180.
        rot_z_spin = rot_z_spin * np.pi / 180.
        calib_matrix = self.build_rotation_matrix(rot_x_flip, rot_y_flip, rot_z_spin)
        print ('Rotation matrix:\n{}'.format(calib_matrix))

        # and rotate at origin
        self._hb2b = self.rotate_instrument(self._hb2b, calib_matrix)

        # push to +Z at length of detector arm
        self._hb2b[:, :, 2] += self._arm_length + arm_length_shift

        # rotate 2theta
        two_theta_rad = two_theta * np.pi / 180.
        two_theta_matrix = self._cal_rotation_matrix_y(two_theta_rad)
        self._hb2b = self.rotate_instrument(self._hb2b, two_theta_matrix)

        print ('Rotation 2theta matrix:\n{}'.format(two_theta_matrix))

        # TODO - FIXME - AFTER TEST - Delete
        # compare position
        # test 5 spots (corner and center): (0, 0), (0, 1023), (1023, 0), (1023, 1023), (512, 512)
        pixel_number = self._hb2b.shape[0]
        print ('HB2B instrument shape = '.format(self._hb2b.shape))
        pixel_locations = [(0, 0),
                           (0, pixel_number - 1),
                           (pixel_number - 1, 0),
                           (pixel_number - 1, pixel_number - 1),
                           (pixel_number / 2, pixel_number / 2)]
        for index_i, index_j in pixel_locations:
            # print ('PyRS:   ', pixel_matrix[index_i, index_j])
            # print ('Mantid: ', workspace.getDetector(index_i + index_j * 1024).getPos())  # column major
            pos_python = self._hb2b[index_i, index_j]
            index1d = index_i + pixel_number * index_j
            # pos_mantid = workspace.getDetector(index1d).getPos()
            print ('({}, {} / {}):   {:10s} '
                   ''.format(index_i, index_j, index1d, 'PyRS'))
            diff_sq = 0.
            for i in range(3):
                # diff_sq += (float(pos_python[i] - pos_mantid[i]))**2
                print ('dir {}:  {:10f}'
                       ''.format(i, float(pos_python[i])))  # float(pos_mantid[i])))
            # END-FOR
            if diff_sq > 1.E-6:
                raise RuntimeError('Mantid PyRS mismatch!')
        # END-FOR
        # END-TEST-OUTPUT

        return self._hb2b

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

    def build_rotation_matrix(self, rot_x_rad, rot_y_rad, rot_z_rad):
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

        print ('[DB...BAT] Rotation Matrix: X = {}, Y = {}, Z = {}'
               ''.format(rot_x_rad, rot_y_rad, rot_z_rad))
        print (rot_x_matrix)
        print (rot_y_matrix)
        print (rot_z_matrix)

        rotation_matrix = rot_x_matrix * rot_y_matrix * rot_z_matrix

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
    def reduce_to_2theta_histogram(det_pos_matrix, counts_matrix, mask, num_bins):
        """ convert the inputs (detector matrix and counts to 2theta histogram)
        :param det_pos_matrix:
        :param counts_matrix:
        :param mask: vector of masks
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
        vecx = twotheta_matrix.transpose().flatten()
        vecy = counts_matrix.flatten()   # in fact vec Y is flattern alraedy!
        vecy = vecy.astype('float64')  # change to integer 32

        print ('[DB...BAT] counts matrix: shape = {}, type = {}'.format(counts_matrix.shape, counts_matrix.dtype))
        if mask is not None:
            print ('[DB...BAT] mask vector; shape = {}, type = {}'.format(mask.shape, mask.dtype))
            checkdatatypes.check_numpy_arrays('Counts vector and mask vector', [counts_matrix, mask], 1, True)
            vecy *= mask

        # vecy = counts_matrix.transpose().flatten()   # a try!

        # for i in range(10):
        #     print ('PyRS {}: x = {}, y = {}'.format(i, vecx[i], vecy[i]))
        # for i in range(10010, 10020):
        #     print ('PyRS {}: x = {}, y = {}'.format(i, vecx[i], vecy[i]))
        # for i in range(3000000, 3000020):
        #     print ('PyRS {}: x = {}, y = {}'.format(i, vecx[i], vecy[i]))
        #
        # print (vecx.shape, vecx.sum(), vecx.dtype)
        # print (vecy.shape, vecy.sum(), vecy.dtype)
        # print (num_bins)

        hist, bin_edges = np.histogram(vecx, bins=num_bins, weights=vecy)
        # for i in range(1000, 1020):
        #     print (bin_edges[i], hist[i])

        return bin_edges, hist
# END-CLASS


def test_main():
    """
    test main to verify algorithm
    :return:
    """
    from mantid.simpleapi import LoadSpiceXML2DDet, LoadInstrument, AddSampleLog
    from mantid.api import AnalysisDataService as ADS
    import os

    def test_with_mantid(arm_length=0.95, two_theta=0., center_shift_x=0., center_shift_y=0.,
                         rot_x_flip=0., rot_y_flip=0., rot_z_spin=0.):
        """
        :param arm_length: full arm length: engineered = 0.95 m
        :param two_theta: 2theta in sample log (instrument definition). It is opposite direction to Mantid coordinate
        :return:
        """
        pixel_matrix = hb2b_builder.build_instrument(arm_length=arm_length, two_theta=-two_theta,
                                                     center_shift_x=center_shift_x, center_shift_y=center_shift_y,
                                                     rot_x_flip=rot_x_flip, rot_y_flip=rot_y_flip,
                                                     rot_z_spin=rot_z_spin)

        # set up sample logs
        # cal::arm
        AddSampleLog(Workspace='hb2b', LogName='cal::arm', LogText='{}'.format(arm_length-DEFAULT_ARM_LENGTH),
                     LogType='Number Series', LogUnit='meter',
                     NumberType='Double')
        #
        # cal::2theta
        AddSampleLog(Workspace='hb2b', LogName='cal::2theta', LogText='{}'.format(-two_theta),
                     LogType='Number Series', LogUnit='degree',
                     NumberType='Double')
        #
        # cal::deltax
        AddSampleLog(Workspace='hb2b', LogName='cal::deltax', LogText='{}'.format(center_shift_x),
                     LogType='Number Series', LogUnit='meter',
                     NumberType='Double')
        #
        # cal::deltay
        AddSampleLog(Workspace='hb2b', LogName='cal::deltay', LogText='{}'.format(center_shift_y),
                     LogType='Number Series', LogUnit='meter',
                     NumberType='Double')

        # cal::roty
        AddSampleLog(Workspace='hb2b', LogName='cal::roty', LogText='{}'.format(-two_theta - rot_y_flip),
                     LogType='Number Series', LogUnit='degree',
                     NumberType='Double')

        # cal::flip
        AddSampleLog(Workspace='hb2b', LogName='cal::flip', LogText='{}'.format(rot_x_flip),
                     LogType='Number Series', LogUnit='degree',
                     NumberType='Double')

        # cal::spin
        AddSampleLog(Workspace='hb2b', LogName='cal::spin', LogText='{}'.format(rot_z_spin),
                     LogType='Number Series', LogUnit='degree',
                     NumberType='Double')

        LoadInstrument(Workspace='hb2b',
                       Filename=os.path.join(root, 'prototypes/calibration/HB2B_Definition_v4.xml'),
                       InstrumentName='HB2B', RewriteSpectraMap='True')
        workspace = ADS.retrieve('hb2b')

        # test 5 spots: (0, 0), (0, 1023), (1023, 0), (1023, 1023), (512, 512)
        for index_i, index_j in [(0, 0), (0, 1023), (1023, 0), (1023, 1023), (512, 512)]:
            print ('PyRS:   ', pixel_matrix[index_i, index_j])
            print ('Mantid: ', workspace.getDetector(index_i + index_j*1024).getPos())   # column major
            pos_python = pixel_matrix[index_i, index_j]
            pos_mantid = workspace.getDetector(index_i + 1024 * index_j).getPos()
            for i in range(3):
                print ('dir {}:  {:10f}   -   {:10f}    =   {:10f}'
                       ''.format(i, float(pos_python[i]), float(pos_mantid[i]),
                                 float(pos_python[i] - pos_mantid[i])))
            # END-FOR
        # END-FOR

        return

    # build instrument
    num_rows = 1024
    num_columns = 1024
    pixel_size_x = 0.00029296875
    pixel_size_y = 0.00029296875
    hb2b_builder = PyHB2BReduction(num_rows, num_columns, pixel_size_x, pixel_size_y)

    # load data to Mantid
    root = os.getcwd()
    if root.startswith('/home/'):
        root = os.path.join(os.path.expanduser('~'), 'Projects/PyRS/')
    else:
        root = os.path.join(os.path.expanduser('~'), 'Projects/PyRS/')
    LoadSpiceXML2DDet(Filename=os.path.join(root, 'tests/testdata/LaB6_10kev_35deg-00004_Rotated.bin'),
                      OutputWorkspace='hb2b',
                      DetectorGeometry='0,0',
                      LoadInstrument=False)

    # small angle
    print ('\n\nJust small angle')
    test_with_mantid(arm_length=0.95, two_theta=0., center_shift_x=0., center_shift_y=0.,
                     rot_x_flip=0., rot_y_flip=0., rot_z_spin=0.)

    # some shift
    print ('\n\nShift X and Shift Y')
    test_with_mantid(arm_length=0.95, two_theta=0., center_shift_x=0.2, center_shift_y=0.3,
                     rot_x_flip=0., rot_y_flip=0., rot_z_spin=0.)

    # some rotation about 2theta
    print ('\n\n2Theta = 85.')
    test_with_mantid(arm_length=0.95, two_theta=85., center_shift_x=0., center_shift_y=0.,
                     rot_x_flip=0., rot_y_flip=0., rot_z_spin=0.)

    # some
    # TODO - NIGHT - Refer to .../prototypes/calibration/intrument_geometry_benchmark.py

if __name__ == '__main__':
    test_main()
