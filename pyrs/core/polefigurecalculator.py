# This module is to calculate Pole Figure
# import mantid_fit_peak
# import peakfitengine
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
    A calculator for Pole Figure.
    It has the memory and result for the last time it is called to calculate
    """
    def __init__(self):
        """
        initialization
        """
        # initialize class instances
        self._peak_info_dict = dict()   # key: detector ID, scan log index  (int, int)
        self._peak_intensity_dict = dict()   # key: detector ID, scan log index (int, int)
        self._peak_fit_info_dict = dict()  # key: detector ID, value: dict of  floats
        self._pole_figure_dict = dict()  # key: detector ID, value: 2-tuple.  scan log indexes (list), 2D array

        # flag
        self._cal_successful = False

        self._use_matmul = does_numpy_support_matmul()

        return

    def add_input_data_set(self, det_id, peak_intensity_dict, peak_fit_info_dict, log_dict):
        """ set peak intensity log and experiment logs that are required by pole figure calculation
        :param det_id
        :param peak_intensity_dict : dictionary (key = scan log index (int), value = peak intensity (float)
        :param peak_fit_info_dict: dictionary (key = scan log index (int), value = peak fitting information (float)
        :param log_dict: dictionary (key = scan log index (int), value = dictionary (log name, log value))
        :return:
        """
        # check inputs
        if det_id in self._peak_intensity_dict:
            raise RuntimeError('Detector ID {0} already been added.  Must be reset calculator.'
                               ''.format(det_id))
        checkdatatypes.check_int_variable('Detector ID', det_id, (0, None))
        checkdatatypes.check_dict('Peak intensities', peak_intensity_dict)
        checkdatatypes.check_dict('Peak fitting information', peak_fit_info_dict)
        checkdatatypes.check_dict('Log values for pole figure', log_dict)

        # check sample log index
        if set(peak_intensity_dict.keys()) != set(log_dict.keys()):
            raise RuntimeError('Sample log indexes from peak intensities and sample logs'
                               ' do not match.')

        # add peak intensity
        self._peak_intensity_dict[det_id] = peak_intensity_dict

        # go through all the values
        for log_index in log_dict:
            # check
            log_names = log_dict[log_index].keys()
            checkdatatypes.check_list('Pole figure motor names', log_names, ['2theta', 'chi', 'phi', 'omega'])
            # set
            self._peak_info_dict[det_id] = log_dict
            self._peak_fit_info_dict[det_id] = peak_fit_info_dict
        # END-FOR

        return

    def calculate_pole_figure(self, det_id_list):
        """ Calculate pole figures
        :param det_id_list:
        :return:
        """
        # check input
        if det_id_list is None:
            det_id_list = self.get_detector_ids()
        else:
            checkdatatypes.check_list('Detector IDs to calculate pole figure', det_id_list,
                                      self.get_detector_ids())
        # END-IF

        for det_id in det_id_list:
            # calculator by each detector
            peak_intensity_dict = self._peak_intensity_dict[det_id]
            peak_info_dict = self._peak_info_dict[det_id]

            # construct the output
            scan_log_index_list = sorted(peak_intensity_dict.keys())
            num_pts = len(scan_log_index_list)
            pole_figure_array = numpy.ndarray(shape=(num_pts, 3), dtype='float')

            for index, scan_index in enumerate(scan_log_index_list):
                # check fitting result
                intensity_i = peak_intensity_dict[scan_index]

                # rotate Q from instrument coordinate to sample coordinate
                two_theta_i = peak_info_dict[scan_index]['2theta']
                omega_i = peak_info_dict[scan_index]['omega']
                chi_i = peak_info_dict[scan_index]['chi']
                phi_i = peak_info_dict[scan_index]['phi']
                alpha, beta = self.rotate_project_q(two_theta_i, omega_i, chi_i, phi_i)

                pole_figure_array[index, 0] = alpha
                pole_figure_array[index, 1] = beta
                pole_figure_array[index, 2] = intensity_i
                print ('[DB...BAT] index: {0} scan {1} alpha = {2}, beta = {3}'.format(index, scan_index, alpha, beta))
                # END-FOR
            # END-FOR

            # convert
            self._pole_figure_dict[det_id] = scan_log_index_list, pole_figure_array

        # END-FOR

        return

    def execute(self):
        """
        calculate pole figure
        :return:
        """
        # fit peaks
        # choose peak fit engine
        if use_mantid:
            fit_engine = mantid_fit_peak.MantidPeakFitEngine(self._peak_info_dict)
        else:
            fit_engine = peakfitengine.ScipyPeakFitEngine(self._peak_info_dict)

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
        :param detector_id_list:
        :param file_name:
        :return:
        """
        # TODO - 20180711 - Clean this method and allow user to specifiy header
        # process detector ID list
        if detector_id_list is None:
            detector_id_list = self.get_detector_ids()
        else:
            checkdatatypes.check_list('Detector IDs', detector_id_list)
        # check inputs
        checkdatatypes.check_file_name(file_name, check_exist=False, check_writable=True)
        checkdatatypes.check_string_variable('Output pole figure file type/format', file_type)

        print ('[DB...BAT] Pole figure to export:  type: {0}\n{1}'.format(type(self._pole_figure_dict),
                                                                          self._pole_figure_dict))

        # it is a dictionary now
        if file_type.lower() == 'ascii':
            # export pole figure arrays as ascii column file
            export_arrays_to_ascii(self._pole_figure_dict, file_name)
        elif file_type.lower() == 'mtex':
            # export to mtex format
            export_to_mtex(self._pole_figure_dict, file_name, header='....')

        return

    def get_detector_ids(self):
        """
        get all the detector IDs
        :return: list of integer
        """
        return self._peak_intensity_dict.keys()

    def get_peak_fit_parameter_vec(self, param_name, det_id):
        """

        :param param_name:
        :return:
        """
        checkdatatypes.check_int_variable('Detector ID', det_id, (0, None))

        param_vec = numpy.ndarray(shape=(len(self._peak_fit_info_dict), ), dtype='float')
        log_index_list = sorted(self._peak_fit_info_dict.keys())
        for i, log_index in enumerate(log_index_list):
            try:
                param_vec[i] = self._peak_fit_info_dict[det_id][log_index][param_name]
            except KeyError:
                raise RuntimeError('Parameter {0} is not a key.  Candidates are {1} ... {2}'
                                   ''.format(param_name, self._peak_fit_info_dict[det_id].keys(),
                                             self._peak_fit_info_dict[det_id][log_index].keys()))

        return param_vec

    def get_pole_figure(self, det_id, max_cost):
        """ return Pole figure in a numpy 2D array
        :param det_id:
        :param max_cost:
        :return: 2-tuple: (1) an integer list (2) numpy array with shape (n, 3).  n is the number of data points
        """
        log_index_vec, pole_figure_vec = self._pole_figure_dict[det_id]
        cost_vec = self.get_peak_fit_parameter_vec('cost', det_id)

        # filter out the value by cost
        if max_cost is None:
            return log_index_vec, pole_figure_vec

        # selected, i.e., filtered by maximum
        # check input
        checkdatatypes.check_float_variable('Maximum cost', max_cost, (0, None))

        indexes = numpy.where(cost_vec < max_cost)
        indexes = indexes[0]

        selected_log_index_vec = numpy.take(log_index_vec, indexes, axis=0)
        selected_pole_figure_vec = numpy.take(pole_figure_vec, indexes, axis=0)

        return selected_log_index_vec, selected_pole_figure_vec

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
        # check inputs - FIXME 20180716 - Recover check types after debugging - TODO
        checkdatatypes.check_float_variable('2theta', two_theta, (None, None))
        checkdatatypes.check_float_variable('Omega', omega, (None, None))
        checkdatatypes.check_float_variable('chi', chi, (None, None))
        checkdatatypes.check_float_variable('phi', phi, (None, None))

        # print ('2theta = {0}\nomega = {1}\nchi = {2}\nphi = {3}'
        #        ''.format(two_theta, omega, chi, phi))

        # rotate Q about theta along z-axis
        rotation_matrix = cal_rotation_matrix_z(-two_theta * 0.5, is_degree=True, use_matrix=self._use_matmul)

        # print ('Rotation about Z-axis with {0}:\n{1}'.format(-two_theta * 0.5, nice(rotation_matrix)))

        if self._use_matmul:
            vec_q1 = numpy.matmul(rotation_matrix, numpy.array([0., 1., 0.]))
            vec_q2 = numpy.matmul(rotation_matrix, numpy.array([1., 0., 0.]))
        else:
            vec_q1 = matrix_mul_vector(rotation_matrix, numpy.array([0., 1., 0.]))
            vec_q2 = matrix_mul_vector(rotation_matrix, numpy.array([1., 0., 0.]))
        # print ('[DB...BAT] vec(Q) shape: {0}'.format(vec_q1.shape))
        # print ('Vec(q)_1: {0}'.format(vec_q1))
        # print ('Vec(q)_2: {0}'.format(vec_q2))

        # print ('[INFO] Rotation about X-axis (phi+90): A\n{0}'
        #        ''.format(nice(cal_rotation_matrix_x(phi + 90, True, True))))
        # print ('Production 1: A x B\n{0}'.format(nice(temp_matrix)))
        # print ('Rotation about Z-axis (-omega):    C\n{0}'
        #        ''.format(nice(cal_rotation_matrix_z(-omega, True, True))))

        # rotate about phi, chi and omega
        if self._use_matmul:
            temp_matrix = numpy.matmul(cal_rotation_matrix_x(phi + 90, True, self._use_matmul),
                                       cal_rotation_matrix_y(chi, True, self._use_matmul))
            temp_matrix = numpy.matmul(temp_matrix, cal_rotation_matrix_z(-omega, True, self._use_matmul))

            vec_q_prime1 = numpy.matmul(temp_matrix, numpy.array([0., 1., 0.]))
            vec_q_prime2 = numpy.matmul(temp_matrix, numpy.array([1., 0., 0.]))
        else:
            temp_matrix = matrix_mul_matrix(cal_rotation_matrix_x(phi + 90, True, self._use_matmul),
                                            cal_rotation_matrix_y(chi, True, self._use_matmul))
            temp_matrix = matrix_mul_matrix(temp_matrix, cal_rotation_matrix_z(-omega, True, self._use_matmul))

            vec_q_prime1 = matrix_mul_vector(temp_matrix, numpy.array([0., 1., 0.]))
            vec_q_prime2 = matrix_mul_vector(temp_matrix, numpy.array([1., 0., 0.]))
        # END-IF-ELSE

        # print ('Production 2: A x B x C\n{0}'.format(nice(temp_matrix)))
        # print ('Vec(q)_1\': {0}'.format(vec_q_prime1))
        # print ('Vec(q)_2\': {0}'.format(vec_q_prime2))
        # print ('[DB...BAT] vec(Q\') shape: {0}'.format(vec_q1.shape))

        # calculate projection to alpha and beta
        if len(vec_q_prime1.shape) == 2 and vec_q_prime1[0, 2] >= 0 or \
                len(vec_q_prime1.shape) == 1 and vec_q_prime1[2] >= 0:
            beta = 360 - math.acos(numpy.dot(vec_q_prime1, vec_q1.transpose())) * 180. / numpy.pi
        else:
            beta = math.acos(numpy.dot(vec_q_prime1, vec_q1.transpose())) * 180. / numpy.pi

        alpha = math.acos(numpy.dot(vec_q_prime2, vec_q2.transpose())) * 180. / numpy.pi

        if beta <= 90:
            beta = 360 + (beta - 90)
        else:
            beta -= 90.

        # print ('[INFO] Alpha = {0}\tBeta = {1}'.format(alpha, beta))

        return alpha, beta

    def reset_calculator(self):
        """

        :return:
        """
        self._peak_info_dict = dict()
        self._peak_intensity_dict = dict()
        self._pole_figure_dict = dict()

        return

# END-OF-CLASS (PoleFigureCalculator)


def test_rotate():
    pf_cal = PoleFigureCalculator()
    pf_cal._use_matmul = True
    

    # row 636: same from pyrs-gui-test
    two_theta   = 82.3940
    omega       = -48.805
    chi         = 8.992663
    phi         = 60.00
        
    a, b = pf_cal.rotate_project_q(two_theta, omega, chi, phi)
    print (a, b)


def export_arrays_to_ascii(array_dict, out_file_name):
    """
    export a dictionary of arrays to an ASCII file
    :param array_dict: 
    :param out_file_name: 
    :return: 
    """
    print ('[DB...Export Pole Figure Arrays To ASCII:\nKeys: {0}\nValues[0]: {1}'
           ''.format(array_dict.keys(), array_dict.values()[0]))

    # combine
    pole_figure_array_list = list()
    for index_vec, pole_figure_vec in array_dict.values():
        pole_figure_array_list.append(pole_figure_vec)

    combined_array = numpy.concatenate(pole_figure_array_list, axis=0)
    # sort
    combined_array = numpy.sort(combined_array, axis=0)
    # save
    numpy.savetxt(out_file_name, combined_array)   # x,y,z equal sized 1D arrays

    return


def export_to_mtex(pole_figure_array_dict, file_name, header):
    """
    export to mtex format, which includes
    line 1: header
    line 2 and on: alpha\tbeta\tintensity
    :param file_name:
    :param pole_figure_array_dict:
    :param header
    :return:
    """
    # check input types
    checkdatatypes.check_dict('Pole figure array dictionary', pole_figure_array_dict)

    # initialize output string
    mtex = ''

    # header
    mtex += '{0}\n'.format(header)

    # writing data
    pf_keys = sorted(pole_figure_array_dict.keys())
    for pf_key in pf_keys:
        pole_figure_array = pole_figure_array_dict[pf_key]
        for i_pt in range(pole_figure_array.size):
            mtex += '{0:5.5f}\t{1:5.5f}\t{2:5.5f}\n' \
                    ''.format(pole_figure_array[i_pt, 0], pole_figure_array[i_pt, 1], pole_figure_array[i_pt, 2])
        # END-FOR (i_pt)
    # END-FOR

    # write file
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



test_rotate()
