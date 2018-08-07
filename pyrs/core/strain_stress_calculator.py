import numpy
import pyrs.utilities.checkdatatypes
import scandataio


class StrainStress(object):
    """
    class to take a calculate strain
    """
    def __init__(self, peak_pos_matrix, d0, young_modulus, poisson_ratio, is_plane_train, is_plane_stress):
        """

        :param peak_pos_vec:
        :param d0:
        """
        # check input
        pyrs.utilities.checkdatatypes.check_float_variable('Peak position (d0)', d0, (0, 30))
        pyrs.utilities.checkdatatypes.check_numpy_arrays('Fitted peak positions as d11, d22, d3',
                                                         [peak_pos_matrix],
                                                         dimension=2, check_same_shape=False)
        pyrs.utilities.checkdatatypes.check_float_variable('Young modulus E', young_modulus, (None, None))
        pyrs.utilities.checkdatatypes.check_float_variable('Poisson ratio Nu', poisson_ratio, (None, None))

        self._epsilon = numpy.zeros(shape=(3, 3), dtype='float')
        self._sigma = numpy.zeros(shape=(3, 3), dtype='float')
        self._peak_position_matrix = peak_pos_matrix

        if is_plane_stress:
            self._calculate_as_plane_stress(d0, young_modulus, poisson_ratio)
        elif is_plane_train:
            self._calculate_as_plane_stress(d0, young_modulus, poisson_ratio)
        else:
            self._calculate_as_unconstrained(d0, young_modulus, poisson_ratio)

    def _calculate_as_unconstrained(self, d0, young_e, poisson_nu):
        """
        unconstrained
        :param d0:
        :param young_e:
        :param poisson_nu:
        :return:
        """
        # calculate strain
        sum_diagonal_strain = 0.
        for index in [0, 1, 2]:
            self._epsilon[index, index] = (self._peak_position_matrix[index, index] - d0)/d0
            sum_diagonal_strain += self._epsilon[index, index]

        # calculate stress
        for i in range(3):
            for j in range(3):
                self._sigma[i, j] = young_e/(1 + poisson_nu) * \
                                    (self._epsilon[i, j] + poisson_nu / (1 - 2*poisson_nu) * sum_diagonal_strain)
            # END-j
        # END-i

        return

    def _calculate_as_plane_strain(self, d0, young_e, poisson_nu):
        """
        epsilon_33 = 0
        :param d0:
        :param young_e:
        :param poisson_nu:
        :return:
        """
        # calculate strain
        sum_diagonal_strain = 0.
        for index in [0, 1]:
            self._epsilon[index, index] = (self._peak_position_matrix[index, index] - d0)/d0
            sum_diagonal_strain += self._epsilon[index, index]

        # calculate stress
        for i in range(3):
            for j in range(3):
                self._sigma[i, j] = young_e/(1 + poisson_nu) * \
                                    (self._epsilon[i, j] + poisson_nu / (1 - 2*poisson_nu) * sum_diagonal_strain)
            # END-j
        # END-i

        return

    def _calculate_as_plane_stress(self, d0, young_e, poisson_nu):
        """
        sigma(3, 3) = 0
        :param d0:
        :param young_e:
        :param poisson_nu:
        :return:
        """
        sum_diagonal_strain = 0.
        for index in [0, 1]:
            self._epsilon[index, index] = (self._peak_position_matrix[index, index] - d0)/d0
            sum_diagonal_strain += self._epsilon[index, index]

        self._epsilon[2, 2] = poisson_nu/(poisson_nu-1) * sum_diagonal_strain
        sum_diagonal_strain += self._epsilon[2, 2]

        # calculate stress
        for i in range(3):
            for j in range(3):
                self._sigma[i, j] = young_e/(1 + poisson_nu) * \
                                    (self._epsilon[i, j] + poisson_nu / (1 - 2*poisson_nu) * sum_diagonal_strain)
            # END-j
        # END-i

        if abs(self._sigma[2, 2]) > 1.E-5:
            raise RuntimeError('unable to converge epsilon(3, 3) to zero.')

        return


class StrainStressCalculator(object):
    """
    class to manage strain stress calculation
    """
    def __init__(self, session_name, plane_stress=False, plane_strain=False):
        """
        initialization
        :param session_name:
        :param plane_strain:
        :param plane_stress:
        """
        # check input
        pyrs.utilities.checkdatatypes.check_string_variable('Session name', session_name)
        pyrs.utilities.checkdatatypes.check_bool_variable('Is plane stress', plane_stress)
        pyrs.utilities.checkdatatypes.check_bool_variable('Is plane strain', plane_strain)

        if plane_stress and plane_strain:
            raise RuntimeError('An experiment cannot be both plane stress and plane stress')

        # class variable
        self._session = session_name
        self._is_plane_strain = plane_strain
        self._is_plane_stress = plane_stress
        self._direction_list = ['e11', 'e22', 'e33']
        if self._is_plane_strain or self._is_plane_stress:
            self._direction_list.pop(2)

        # data sets required
        self._data_set_dict = dict()
        for dir_i in self._direction_list:
            self._data_set_dict[dir_i] = None

        # source files
        self._source_file_dict = dict()
        for dir_i in self._direction_list:
            self._source_file_dict[dir_i] = None

        # transformed data set
        self._dir_sample_scan_dict = dict()
        for dir_i in self._direction_list:
            self._dir_sample_scan_dict[dir_i] = None  # shall be a dictionary: key = sample position

        # flag whether the measured sample points can be aligned. otherwise, more complicated algorithm is required
        # including searching and interpolation
        self._sample_points_aligned = False

        # list of sample positions for each data set
        self._sample_positions_dict = dict()
        for dir_i in self._direction_list:
            self._sample_positions_dict[dir_i] = None  # each shall be None or a list of tuples
        # status
        self._is_saved = False

        # file loader (static kind of)
        self._file_io = scandataio.DiffractionDataFile()

        return

    def align_measuring_points(self, pos_x, pos_y, pos_z, resolution=0.001):
        """
        Align the data points among e11, e22 and/or e33 with sample log positions
        :param pos_x: sample log name for x position
        :param pos_y: sample log name for y position
        :param pos_z: sample log name for z position
        :return:
        """
        # check inputs
        pyrs.utilities.checkdatatypes.check_string_variable('Sample log name for X position', pos_x)
        pyrs.utilities.checkdatatypes.check_string_variable('Sample log name for Y position', pos_y)
        pyrs.utilities.checkdatatypes.check_string_variable('Sample log name for Z position', pos_z)
        if pos_x == pos_y or pos_y == pos_z or pos_x == pos_z:
            raise RuntimeError('Position X ({}) Y ({}) and Z ({}) have duplicate sample log names.'
                               ''.format(pos_x, pos_y, pos_z))

        # get the measuring points
        for dir_i in self._direction_list:
            self._dir_sample_scan_dict[dir_i] = self.generate_xyz_scan_log_dict(dir_i, pos_x, pos_y, pos_z)
        # END-FOR

        # align: create a list of sorted tuples and compare among different data sets whether they
        # do match or not
        for dir_i in self._direction_list:
            self._sample_positions_dict[dir_i] = sorted(self._dir_sample_scan_dict[dir_i].keys())

        # set flag
        self._sample_points_aligned = False

        # check whether the sample position numbers are different
        num_dir = len(self._direction_list)
        for i_dir_i in range(num_dir):
            dir_i = self._direction_list[i_dir_i]
            for i_dir_j in range(i_dir_i, num_dir):
                dir_j = self._direction_list[i_dir_j]
                if len(self._sample_positions_dict[dir_i]) != len(self._sample_positions_dict[dir_j]):
                    raise RuntimeError('It is not considered that the number of data points among different '
                                       'direction are different.  Need to use uneven alignment algorithm.')
        # END-FOR

        # check whether all the data points matched with each other within resolution
        num_sample_points = len(self._sample_positions_dict[0])
        for ipt in range(num_sample_points):
            max_distance = self.calculate_max_distance(ipt)
            if max_distance > resolution:
                err_msg = '{}-th sample position point: '
                for dir_i in self._direction_list:
                    err_msg += '{} @ {}; '.format(dir_i, self._sample_positions_dict[dir_i][ipt])
                err_msg += ' with maximum distance {} over specified resolution {}'.format(max_distance, resolution)
                raise RuntimeError(err_msg)
        # END-FOR

        self._sample_points_aligned = True

        return

    @staticmethod
    def calculate_distance(pos_1, pos_2):
        """
        calculate distance between 2 data points in a sequence as x, y and z
        :param pos_1:
        :param pos_2:
        :return:
        """
        # transform tuple or list to vectors
        if isinstance(pos_1, tuple) or isinstance(pos_1, list):
            vec_pos_1 = numpy.array(pos_1)
        elif isinstance(pos_1, numpy.ndarray) is False:
            raise RuntimeError('Position 1 {} with type {} is not supported.'
                               ''.format(pos_1, type(pos_1)))
        else:
            vec_pos_1 = pos_1

        if isinstance(pos_2, tuple) or isinstance(pos_2, list):
            vec_pos_2 = numpy.array(pos_2)
        elif isinstance(pos_2, numpy.ndarray) is False:
            raise RuntimeError('Position 2 {} with type {} is not supported.'
                               ''.format(pos_1, type(pos_1)))
        else:
            vec_pos_2 = pos_2

        return numpy.sqrt(numpy.sum(vec_pos_1 ** 2 + vec_pos_2 ** 2))

    def calculate_max_distance(self, sample_point_index):
        """
        in the case that all 2 or 3
        :param sample_point_index:
        :return:
        """
        pyrs.utilities.checkdatatypes.check_int_variable('Sample point index', sample_point_index,
                                                         (0, len(self._sample_positions_dict['e11'])))

        num_dir = len(self._direction_list)
        max_distance = -1
        for i_dir_i in range(num_dir):
            dir_i = self._direction_list[i_dir_i]
            pos_i = self._sample_positions_dict[dir_i][sample_point_index]
            for i_dir_j in range(i_dir_i, num_dir):
                dir_j = self._direction_list[i_dir_j]
                pos_j = self._sample_positions_dict[dir_j][sample_point_index]
                distance_i_j = self.calculate_distance(pos_i, pos_j)
                max_distance = max(distance_i_j, max_distance)
            # END-FOR-j
        # END-FOR-i

        return max_distance

    def generate_xyz_scan_log_dict(self, direction, pos_x, pos_y, pos_z):
        """
        Retrieve XYZ position
        :param direction:
        :param pos_x: sample log name for x position
        :param pos_y: sample log name for y position
        :param pos_z: sample log name for z position
        :return:
        """
        # check input
        if direction not in self._direction_list:
            raise NotImplementedError('Direction {} is not an allowed direction {}'
                                      ''.format(direction, self._direction_list))

        # returned dictionary
        xyz_log_index_dict = dict()

        # retrieve data set
        data_set = self._data_set_dict[direction]
        print ('DB...BAT] Data set: type = {}'.format(type(data_set)))

        for scan_log_index in data_set.get_scan_log_indexes():
            x_i = data_set.get_sample_log_value(pos_x)
            y_i = data_set.get_sample_log_value(pos_y)
            z_i = data_set.get_sample_log_value(pos_z)
            xyz_log_index_dict[(x_i, y_i, z_i)] = scan_log_index
        # END-FOR

        return xyz_log_index_dict

    @property
    def is_sample_positions_aligned(self):
        """
        flag whether all the sample positions are aligned
        :return:
        """
        return self._sample_points_aligned

    @property
    def is_plane_strain(self):
        """
        whether this session is for plane strain
        :param self:
        :return:
        """
        return self._is_plane_strain

    @property
    def is_plane_stress(self):
        """
        whether this session is for plane stress
        :return:
        """
        return self._is_plane_stress

    @property
    def is_unconstrained_strain_stress(self):
        """
        whether this session is unconstrained strain and stress
        :return:
        """
        return not (self._is_plane_strain or self._is_plane_stress)

    @property
    def is_saved(self):
        """
        check whether current session is saved
        :return:
        """
        return self._is_saved

    def load_raw_file(self, direction, file_name):
        """
        load raw experimental file with fit result
        :param direction:
        :param file_name:
        :return:
        """
        # check input
        pyrs.utilities.checkdatatypes.check_string_variable('Strain/Stress Direction', direction,
                                                            ['e11', 'e22', 'e33'])
        if (self._is_plane_stress or self._is_plane_strain) and direction == 'e33':
            raise RuntimeError('Direction e33 is not used in plane stress or plane strain case.')

        # import data
        diff_data_dict, sample_logs = self._file_io.load_rs_file(file_name)
        print ('[DB...BAT] data dict: {}... sample logs: {}...'.format(diff_data_dict, sample_logs))

        print (type(sample_logs))
        if data_set.has_fit_parameters is False:
            raise RuntimeError('File {} does not have fitted peak parameters value for strain/stress '
                               'calculation'.format(file_name))

        # assign data file to files
        self._data_set_dict[direction] = data_set

        # record data file name
        self._source_file_dict[direction] = file_name
        # check duplicate file
        src_file_set = set(self._source_file_dict.values())
        if len(src_file_set) != len(self._source_file_dict):
            raise RuntimeError('There are duplicate files among various directions: {}'
                               ''.format(self._source_file_dict))

        return

    def save_session(self, save_file_name):
        """
        save the complete session in order to open later
        :param save_file_name:
        :return:
        """
        # TODO - 2018 - NEXT
        raise NotImplementedError('ASAP')

    @property
    def session(self):
        """
        get session name
        :param self:
        :return:
        """
        return self._session
