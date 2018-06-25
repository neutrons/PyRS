# This module is to calculate Pole Figure
import mantid_fit_peak
import peakfitengine
from pyrs.utilities import checkdatatypes
import numpy
import math


def matrix_mul_vector(matrix, vector):
    """
    matrix multiply a vector
    :param matrix:
    :param vector:
    :return:
    """
    # check input
    assert isinstance(matrix, numpy.ndarray), 'Matrix must be a numpy array'
    assert isinstance(vector, numpy.ndarray), 'Vector must be a numpy array'

    if len(matrix.shape) != 2:
        raise RuntimeError('Matrix must have 2 dimension')

    # vector must be a (1, n) vector
    if len(vector.shape) != 1:
        raise RuntimeError('Vector {0} must be 1 dimensional but not {1}'.format(vector, vector.shape))

    if matrix.shape[1] != vector.shape[0]:
        raise RuntimeError('Matrix and vector are not matched to multiply')

    out_vec = numpy.ndarray(shape=vector.shape, dtype=vector.dtype)

    for i_row in range(matrix.shape[0]):
        try:
            out_vec[i_row] = numpy.sum(matrix[i_row] * vector)
        except ValueError as val_err:
            print ('[ERROR] Row {0} from matrix: {1}; Vector : {2}'.format(i_row, matrix[i_row, :][0], vector))
            raise val_err

    return out_vec


def matrix_mul_matrix(matrix1, matrix2):
    """
    matrix multiply matrix
    :param matrix1:
    :param matrix2:
    :return:
    """
    # check input
    assert isinstance(matrix1, numpy.ndarray), 'Matrix1 must be a numpy array'
    assert isinstance(matrix2, numpy.ndarray), 'Matrix2 must be a numpy array'

    if len(matrix1.shape) != 2:
        raise RuntimeError('Matrix 1 must have 2 dimension')
    if len(matrix2.shape) != 2:
        raise RuntimeError('Matrix 2 must have 2 dimension')

    if matrix1.shape[1] != matrix2.shape[0]:
        raise RuntimeError('Matrix 1 and Matrix 2 are not matched to multiply')

    out_matrix = numpy.ndarray(shape=(matrix1.shape[0], matrix2.shape[1]), dtype=matrix1.dtype)

    for i_row in range(out_matrix.shape[0]):
        for j_col in range(out_matrix.shape[1]):
            out_matrix[i_row, j_col] = numpy.sum(matrix1[i_row, :] * matrix2[:, j_col])

    return out_matrix


def nice(matrix):
    """
    export a string for print a matrix nicely
    :param matrix:
    :return:
    """
    nice_out = ''
    for i_row in range(matrix.shape[0]):
        row = ''
        for j_col in range(matrix.shape[1]):
            row += '{0:5.5f}\t'.format(matrix[i_row, j_col])
        nice_out += row + '\n'

    return nice_out


def cal_rotation_matrix_x(angle, is_degree, use_matrix):
    """
    calculate rotation matrix X
    :param angle:
    :param is_degree: flag show that the angle is in degree but not radius
    :return:
    """
    if is_degree:
        angle = angle / 180. * numpy.pi

    if use_matrix:
        rotation_matrix = numpy.matrix([[1., 0, 0],
                                        [0, numpy.cos(angle), -numpy.sin(angle)],
                                        [0, numpy.sin(angle), numpy.cos(angle)]], dtype='float')
    else:
        rotation_matrix = numpy.array([[1., 0, 0],
                                        [0, numpy.cos(angle), -numpy.sin(angle)],
                                        [0, numpy.sin(angle), numpy.cos(angle)]])

    return rotation_matrix


def cal_rotation_matrix_y(angle, is_degree, use_matrix):
    """
    calculate rotation matrix Y
    :param angle:
    :param is_degree: flag show that the angle is in degree but not radius
    :return:
    """
    if is_degree:
        angle = angle / 180. * numpy.pi

    if use_matrix:
        rotation_matrix = numpy.matrix([[numpy.cos(angle), 0, numpy.sin(angle)],
                                        [0, 1., 0.],
                                        [-numpy.sin(angle), 0., numpy.cos(angle)]], dtype='float')
    else:
        rotation_matrix = numpy.array([[numpy.cos(angle), 0, numpy.sin(angle)],
                                        [0, 1., 0.],
                                        [-numpy.sin(angle), 0., numpy.cos(angle)]])

    return rotation_matrix


def cal_rotation_matrix_z(angle, is_degree, use_matrix):
    """
    calculate rotation matrix Z
    :param angle:
    :param is_degree: flag show that the angle is in degree but not radius
    :return:
    """
    if is_degree:
        angle = angle / 180. * numpy.pi

    if use_matrix:
        rotation_matrix = numpy.matrix([[numpy.cos(angle), -numpy.sin(angle), 0.],
                                        [numpy.sin(angle), numpy.cos(angle), 0.],
                                        [0., 0., 1.]], dtype='float')
    else:
        rotation_matrix = numpy.array([[numpy.cos(angle), -numpy.sin(angle), 0.], [numpy.sin(angle), numpy.cos(angle), 0.], [0., 0., 1.]])

    return rotation_matrix


class PoleFigureCalculator(object):
    """
    A container for a workflow to calculate Pole Figure
    """
    def __init__(self):
        """
        initialization
        """
        # initialize class instances
        self._sample_logs_dict = None

        self._cal_successful = False

        self._use_matmul = does_numpy_support_matmul()

        return

    def calculate_pole_figure(self, peak_intensity_dict):
        """ Calculate pole figures
        :return:
        """
        checkdatatypes.check_dict('(class variable) data set', self._sample_logs_dict)
        checkdatatypes.check_dict('(peak intensities', peak_intensity_dict)

        num_pts = len(self._sample_logs_dict)
        pole_figure_array = numpy.ndarray(shape=(num_pts, 3), dtype='float')

        for index, scan_index in enumerate(self._sample_logs_dict.keys()):
            # check fitting result
            if scan_index in peak_intensity_dict:
                intensity_i = peak_intensity_dict[scan_index]
            else:
                continue

            # rotate Q from instrument coordinate to sample coordinate
            two_theta_i = self._sample_logs_dict[scan_index]['2theta']
            omega_i = self._sample_logs_dict[scan_index]['omega']
            chi_i = self._sample_logs_dict[scan_index]['chi']
            phi_i = self._sample_logs_dict[scan_index]['phi']
            alpha, beta = self.rotate_project_q(two_theta_i, omega_i, chi_i, phi_i)

            pole_figure_array[index, 0] = alpha
            pole_figure_array[index, 1] = beta
            pole_figure_array[index, 2] = intensity_i
        # END-FOR

        # convert
        self._pole_figure = pole_figure_array

        return

    def execute(self):
        """
        calculate pole figure
        :return:
        """
        # fit peaks
        # choose peak fit engine
        if use_mantid:
            fit_engine = mantid_fit_peak.MantidPeakFitEngine(self._sample_logs_dict)
        else:
            fit_engine = peakfitengine.ScipyPeakFitEngine(self._sample_logs_dict)

        # fit peaks
        fit_engine.fit_peaks(peak_function_name=profile_type, background_function_name=background_type,
                             fit_range=peak_range)

        # check result
        fit_engine.mask_bad_fit(max_chi2=self._maxChi2)

        # get fitted peak parameters from engine
        peak_intensity_dict = fit_engine.get_intensities()

        # calculate pole figure
        self._cal_successful = self.calculate_pole_figure(peak_intensity_dict)

        return

    def export_pole_figure(self, detector_id_list,  file_name, file_type):
        """
        exported the calculated pole figure
        :param file_name:
        :return:
        """
        # process detecotr ID list
        if detector_id_list is None:
            detector_id_list = self.get_detector_ids()
        else:
            checkdatatypes.check_list('Detector IDs', detector_id_list)
        # check inputs
        checkdatatypes.check_file_name(file_name, check_exist=False, check_writable=True)
        checkdatatypes.check_string_variable('Output pole figure file type/format', file_type)

        if file_type.lower() == 'ascii':
            numpy.savetxt(file_name, self._pole_figure)   # x,y,z equal sized 1D arrays
        elif file_type.lower == 'mtex':
            export_to_mtex(detector_id_list, file_name, self._pole_figure)

        return

    def get_detector_ids(self):
        """
        get all the detector IDs
        :return: list of integer
        """
        # TODO FIXME! - Need to find a way to set up detector IDs
        return [1]

    def get_pole_figure(self):
        """
        return Pole figure in a numpy 2D array
        :return: numpy array with shape (n, 3).  n is the number of data points
        """

        return self._pole_figure

    def rotate_project_q(self, two_theta, omega, chi, phi):
        """
        Rotate Q from instrument coordinate to sample coordinate defined by goniometer angles
        and project rotation Q to (001) and (100)
        :param two_theta:
        :param omega:
        :param chi:
        :param phi:
        :return: 2-tuple as the projection (alpha, beta)
        """
        # check inputs
        checkdatatypes.check_float_variable('2theta', two_theta, (None, None))
        checkdatatypes.check_float_variable('Omega', omega, (None, None))
        checkdatatypes.check_float_variable('chi', chi, (None, None))
        checkdatatypes.check_float_variable('phi', phi, (None, None))

        print ('2theta = {0}\nomega = {1}\nchi = {2}\nphi = {3}'
               ''.format(two_theta, omega, chi, phi))

        # rotate Q
        vec_q = numpy.array([0., 1., 0.])
        rotation_matrix = cal_rotation_matrix_z(-(180. - two_theta) * 0.5, is_degree=True, use_matrix=self._use_matmul)

        print ('Rotation about Z-axis:\n{0}'.format(nice(rotation_matrix)))

        if self._use_matmul:
            vec_q = numpy.matmul(rotation_matrix, vec_q.transpose())
        else:
            vec_q = matrix_mul_vector(rotation_matrix, vec_q)

        # vec_q_prime = rotation_matrix_z(-omega, True) * rotation_matrix_x(chi, True) *
        #               rotation_matrix_z(phi, True) *  vec_q
        # R(omega) * R(chi)
        matrix_omega = cal_rotation_matrix_z(-omega, True, use_matrix=self._use_matmul)
        matrix_chi = cal_rotation_matrix_x(chi, True, use_matrix=self._use_matmul)
        if self._use_matmul:
            temp_matrix = numpy.matmul(matrix_omega, matrix_chi)
        else:
            temp_matrix = matrix_mul_matrix(matrix_omega, matrix_chi)

        # (R(omega) * R(chi)) * R(phi)
        matrix_phi = cal_rotation_matrix_z(phi, True, use_matrix=self._use_matmul)
        if self._use_matmul:
            temp_matrix = numpy.matmul(temp_matrix, matrix_phi)
        else:
            temp_matrix = matrix_mul_matrix(temp_matrix, matrix_phi)

        # Q = (R(omega) * R(chi)) * R(phi)
        if self._use_matmul:
            vec_q_prime = numpy.matmul(temp_matrix, vec_q.transpose())
        else:
            vec_q_prime = matrix_mul_vector(temp_matrix, vec_q)

        print ('Vector Q (rotated): {0}'.format(vec_q))
        print ('Rotation about Z-axis (-omega): A\n{0}'
               ''.format(nice(matrix_omega)))
        print ('Rotation about X-axis (chi):    B\n{0}'
               ''.format(nice(matrix_chi)))
        print ('Production 1: A x B\n{0}'.format(nice(temp_matrix)))
        print ('Rotation about Z-axis (phi):    C\n{0}'
               ''.format(nice(matrix_phi)))
        print ('Production 2: A x B x C\n{0}'.format(nice(temp_matrix)))
        print ('Vector Q\': {0}'.format(vec_q_prime))

        # project
        alpha = math.acos(numpy.dot(vec_q_prime.transpose(), numpy.array([0., 0., 1.])))
        beta = math.acos(numpy.dot(vec_q_prime.transpose(), numpy.array([1., 0., 0.])))

        print ('Alpha = {0}\tBeta = {1}'.format(alpha, beta))

        return alpha, beta

    def set_experiment_logs(self, log_dict):
        """ set experiment logs that are required by pole figure calculation
        :param log_dict:
        :return:
        """
        # check inputs
        checkdatatypes.check_dict('Log values for pole figure', log_dict)

        # go through all the values
        for log_index in log_dict:
            log_names = log_dict[log_index].keys()
            checkdatatypes.check_list('Pole figure motor names', log_names,
                                ['2theta', 'chi', 'phi', 'omega'])

        self._sample_logs_dict = log_dict


def export_to_mtex(detector_id_list, file_name, pole_figure_array, header):
    """
    export to mtex
    :param detector_id_list:
    :param file_name:
    :param pole_figure_array:
    :return:
    """
    mtex = ''

    # header
    mtex += '{0}\n'.format(header)

    # writing data
    for i_pt in range(pole_figure_array.size):
        mtex += '{0:5.5f}\t{1:5.5f}\t{2:5.5f}\n'.format(pole_figure_array[i_pt, 0], pole_figure_array[i_pt, 1],
                                                        pole_figure_array[i_pt, 2])
    # END-FOR

    p_file = open(file_name, 'w')
    p_file.write(mtex)
    p_file.close()

    return


def does_numpy_support_matmul():
    """
    matmul is supported only after numpy version is after 1.10
    :return:
    """
    np_version = numpy.version.version
    np_version_main = int(np_version.split('.')[0])
    np_version_second = int(np_version.split('.')[1])

    print (np_version_main, np_version_second)

    if np_version_main > 1 or np_version_second >= 10:
        numpy.matmul
        return True

    return False
