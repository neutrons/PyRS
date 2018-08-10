import numpy
import math
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

        return

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

    def get_strain(self):
        """

        :return:
        """
        return self._epsilon

    def get_stress(self):
        """

        :return:
        """
        return self._sigma


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

        # data sets, sample logs and peak parameters required
        self._data_set_dict = dict()
        self._peak_param_dict = dict()
        self._sample_log_dict = dict()
        for dir_i in self._direction_list:
            self._data_set_dict[dir_i] = None
            self._peak_param_dict[dir_i] = None
            self._sample_log_dict[dir_i] = None

        # source files
        self._source_file_dict = dict()
        for dir_i in self._direction_list:
            self._source_file_dict[dir_i] = None

        # transformed data set
        self._dir_sample_scan_dict = dict()
        for dir_i in self._direction_list:
            self._dir_sample_scan_dict[dir_i] = None  # (value) a dictionary: key = sample position, value =

        # flag whether the measured sample points can be aligned. otherwise, more complicated algorithm is required
        # including searching and interpolation
        self._sample_points_aligned = False

        # list of sample positions for each data set
        self._sample_positions_dict = dict()
        for dir_i in self._direction_list:
            self._sample_positions_dict[dir_i] = None  # each shall be None or a LIST of 3-tuples
        # status
        self._is_saved = False

        # file loader (static kind of)
        self._file_io = scandataio.DiffractionDataFile()

        # strain stress parameters
        self._d0 = None
        self._2theta = None
        self._lambda = None

        self._young_e = None
        self._poisson_nu = None

        return

    def align_grids(self, resolution=0.001):
        """
        align grids
        :param resolution:
        :return:
        """
        num_e11_pts = len(self._sample_positions_dict['e11'])
        self._match11_dict = {'e22': [None] * num_e11_pts,
                              'e33': [None] * num_e11_pts}
        # list for the rest
        other_dir_list = self._direction_list[:]
        other_dir_list.pop(0)
        other_dir_list.sort()
        other_dir_matched_dict = dict()
        for dir_i in other_dir_list:
            other_dir_matched_dict[dir_i] = set()

        message = ''
        for ipt_e11 in range(num_e11_pts):
            pos_11_i = self._sample_positions_dict['e11'][ipt_e11]
            for dir_i in other_dir_list:
                sorted_pos_list_i = self._sample_positions_dict[dir_i]
                index_i = self.binary_search(sorted_pos_list_i, pos_11_i, resolution)
                if index_i is None:
                    print ('[DB...BAT] E11 Pt {} @ {} no match at direction {}'.format(ipt_e11, pos_11_i, dir_i))
                else:
                    print ('[DB...BAT] E11 Pt {} @ {} finds {} @ index = {} @ {}'
                           ''.format(ipt_e11, pos_11_i, dir_i, index_i, sorted_pos_list_i[index_i]))
                if index_i is not None:
                    self._match11_dict[dir_i][ipt_e11] = index_i
                    other_dir_matched_dict[dir_i].add(index_i)
            # END-FOR
        # END-FOR

        for dir_i in other_dir_list:
            unmatched_counts_i = len(self._sample_positions_dict[dir_i]) - len(other_dir_matched_dict[dir_i])
            print ('[INFO] Numbers of grids at direction {} unmatched to E11: {}'
                   ''.format(dir_i, unmatched_counts_i))

        return

    @staticmethod
    def binary_search(sorted_positions, xyz, resolution):
        """

        :param sorted_positions:
        :param xyz:
        :return:
        """
        def search_neighborhood(sorted_list, start_index, stop_index, list_index, tuple_index, value_range):
            """
            search direction of tuple_index near list index within resolution
            :param sorted_list:
            :param list_index:
            :param tuple_index:
            :return:
            """
            idx_start = list_index
            center_value = sorted_list[list_index][tuple_index]

            # print ('\t\t[DB...BAT] Coordinate Index = {}. Center value = {}.  Resolution = {}.'
            #        ''.format(tuple_index, center_value, value_range))

            while True:
                idx_start -= 1
                if idx_start < start_index:
                    idx_start = start_index
                    break
                elif sorted_list[idx_start][tuple_index] + value_range < center_value:
                    idx_start += 1
                    break
            # END-WHILE

            idx_stop = list_index
            while True:
                idx_stop += 1
                if idx_stop > stop_index:
                    # out of boundary
                    idx_stop = stop_index
                    break
                elif center_value + value_range < sorted_list[idx_stop][tuple_index]:
                    # x/y/z value is out of range
                    idx_stop -= 1
                    break
            # END-WHILE

            return idx_start, idx_stop

        assert resolution > 0, 'resolution > 0 required'

        i_start = 0
        i_stop = len(sorted_positions) - 1

        matched_x_not_found = True
        matched_y_not_found = False
        matched_z_not_found = False
        i_middle = None
        while matched_x_not_found:
            i_middle = (i_start + i_stop) / 2
            # print ('\t\t[DB...BAT] start = {}, middle = {}, stop = {}'.format(i_start, i_middle, i_stop))
            if abs(xyz[0] - sorted_positions[i_middle][0]) < resolution:
                # equal continue to find y to match
                matched_x_not_found = False
                matched_y_not_found = True
                break
            elif xyz[0] - sorted_positions[i_middle][0] < 0:
                # data point to the left
                i_stop = i_middle - 1
            else:
                # data point to the right
                i_start = i_middle + 1
            # check
            if i_stop < i_start:
                # it is over...
                break
            elif i_start < 0 or i_stop >= len(sorted_positions):
                raise NotImplementedError('It could not happen!')
        # END-WHILE

        if matched_x_not_found:
            # not founded matched X
            print ('\t[DB...BAT] No matched X found')
            return None
        else:
            # print ('\t[DB...BAT] Matched X (1 of many) index'.format(i_middle))
            pass

        # locate the range of X within tolerance/resolution for searching with Y
        i_start, i_stop = search_neighborhood(sorted_positions, 0, len(sorted_positions)-1, i_middle, 0, resolution)
        # print ('\t[DB...BAT] New search range for Y: {}, {}'.format(i_start, i_stop))
        orig_start = i_start
        orig_stop = i_stop
        i_middle = None
        while matched_y_not_found:
            i_middle = (i_start + i_stop) / 2
            if abs(xyz[1] - sorted_positions[i_middle][1]) < resolution:
                # equal continue to find y to match
                matched_y_not_found = False
                matched_z_not_found = True
                break
            elif xyz[1] - sorted_positions[i_middle][1] < 0:
                # data point to the left
                i_stop = i_middle - 1
            else:
                # data point to the right
                i_start = i_middle + 1
            # check
            if i_stop < i_start:
                # it is over...
                break
            elif i_start < orig_start or i_stop > orig_stop:
                raise NotImplementedError('It could not happen!')
        # END-WHILE

        if matched_y_not_found:
            # not found match Y
            print ('\t[DB...BAT] No matched X found')
            return None
        else:
            # print ('\t[DB...BAT] Matched Y (1 of many) index'.format(i_middle))
            pass

        # locate the range of Y within tolerance/resolution for searching with X
        i_start, i_stop = search_neighborhood(sorted_positions, orig_start, orig_stop, i_middle, 1, resolution)
        # print ('\t[DB...BAT] New search range for Z, {}, {}'.format(i_start, i_stop))

        orig_start = i_start
        orig_stop = i_stop
        i_middle = None
        while matched_z_not_found:
            i_middle = (i_start + i_stop) / 2
            if abs(xyz[2] - sorted_positions[i_middle][2]) < resolution:
                # equal continue to find y to match
                matched_z_not_found = False
            elif xyz[2] - sorted_positions[i_middle][2] < 0:
                # data point to the left
                i_stop = i_middle - 1
            else:
                # data point to the right
                i_start = i_middle + 1
            # check
            if i_stop < i_start:
                # it is over...
                break
            elif i_start < orig_start or i_stop > orig_stop:
                raise NotImplementedError('It could not happen!')
        # END-WHILE

        if i_middle is None:
            print ('\t[DB...BAT] No matched Z found')

        return i_middle

    def check_grids_alignment(self, pos_x, pos_y, pos_z, resolution=0.001):
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
            self._sample_positions_dict[dir_i] = sorted(self._dir_sample_scan_dict[dir_i].keys()) # list
            # print self._sample_positions_dict[dir_i]
            # print '\n\n'

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
        num_sample_points = len(self._sample_positions_dict[self._direction_list[0]])
        for ipt in range(num_sample_points):
            max_distance = self.calculate_max_distance(ipt)
            if max_distance > resolution:
                err_msg = '{}-th (of total {}) sample position point: '.format(ipt, num_sample_points)
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

    def execute(self):
        """
        calculate the strain and stress for all grids by using E11 as a standard
        :return:
        """
        # it is assumed that the grids that have been aligned
        # TODO - 20180810 - add some flags to check grid alignment

        # using E11 as the standard grid set
        for ipt_e11 in range(len(self._sample_positions_dict['e11'])):
            # get matched grids from other
            print ipt_e11
            ipt_e22 = self._match11_dict['e22'][ipt_e11]
            if not (self._is_plane_strain or self._is_plane_stress):
                ipt_e33 = self._match11_dict['e33'][ipt_e11]
            else:
                ipt_e33 = True

            if ipt_e22 is None:
                print ('{}-th grid on e11 @ {} has no matched grid on e22... Find a way to interpolate'
                       ''.format(ipt_e11, self._sample_positions_dict['e11'][ipt_e11]))
            if ipt_e33 is None:
                print ('{}-th grid on e11 @ {} has no matched grid on e33.. Find a way to interpolate'
                       ''.format(ipt_e11, self._sample_positions_dict['e11'][ipt_e11]))
            if ipt_e22 is None or ipt_e33 is None:
                continue

            # convert to sample log index
            scan_log_index_dict = dict()

            grid_pos_e11 =  self._sample_positions_dict['e11'][ipt_e11]
            scan_log_index_dict['e11'] = self._dir_sample_scan_dict['e11'][grid_pos_e11]

            grid_pos_e22 =  self._sample_positions_dict['e22'][ipt_e22]
            scan_log_index_dict['e22'] = self._dir_sample_scan_dict['e22'][grid_pos_e22]

            debug_out = 'e11: scan-index = {} @ {}, e22: scan-index = {} @ {}, ' \
                        ''.format(scan_log_index_dict['e11'], grid_pos_e11,
                                  scan_log_index_dict['e22'], grid_pos_e22)
            if isinstance(ipt_e33, int):
                pos_e33 = self._sample_positions_dict['e33'][ipt_e33]
                scan_log_index_dict['e33'] = self._dir_sample_scan_dict['e33'][pos_e33]
                debug_out += 'e33: scan-index = {} @ {}, ' \
                             ''.format(scan_log_index_dict['e33'], pos_e33)
            print (debug_out)

            # calculate peak positions in d-spacing
            peak_matrix = numpy.zeros(shape=(3, 3), dtype='float')
            peak_fit_failed = False
            for mindex, dir_i in enumerate(self._direction_list):
                peak_i_2theta = self._peak_param_dict['e11']['centre'][scan_log_index_dict[dir_i]]
                print ('[DB...BAT] Direction: {}. Log index = {}; Peak center (2theta): {}  degree'
                       ''.format(dir_i, scan_log_index_dict[dir_i], peak_i_2theta))
                if abs(peak_i_2theta) < 1:
                    peak_fit_failed = True
                    continue
                peak_i_d = self._lambda * 0.5 / math.sin(peak_i_2theta * 0.5)   # self.convert_unit_to_d(peak_i_2theta)
                peak_matrix[mindex, mindex] = peak_i_d
            # END-FOR

            if peak_fit_failed:
                continue

            ss_calculator = StrainStress(peak_pos_matrix=peak_matrix,
                                         d0=self._d0, young_modulus=self._young_e,
                                         poisson_ratio=self._poisson_nu,
                                         is_plane_train=self._is_plane_strain,
                                         is_plane_stress=self._is_plane_stress)
            # ss_calculator.get_strain()
            # ss_calculator.get_stress()

        # END-FOR

        return

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

        # print ('DB...BAT] Data set: type = {}.... keys = {}'.format(type(data_set), data_set.keys()))

        print ('[DB...BAT] sample log keys: {}'.format(self._sample_log_dict[direction].keys()))

        for scan_log_index in sorted(data_set.keys()):
            x_i = self._sample_log_dict[direction][pos_x][scan_log_index]
            y_i = self._sample_log_dict[direction][pos_y][scan_log_index]
            z_i = self._sample_log_dict[direction][pos_z][scan_log_index]
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
        # check whether it has been loaded
        if file_name in self._source_file_dict.values():
            raise RuntimeError('File {} of direction {} has already been loaded. FYI: {}'
                               ''.format(file_name, direction, self._source_file_dict))

        # check input
        pyrs.utilities.checkdatatypes.check_string_variable('Strain/Stress Direction', direction,
                                                            ['e11', 'e22', 'e33'])
        if (self._is_plane_stress or self._is_plane_strain) and direction == 'e33':
            raise RuntimeError('Direction e33 is not used in plane stress or plane strain case.')

        # import data
        diff_data_dict, sample_logs = self._file_io.load_rs_file(file_name)
        # print ('[DB...BAT] data dict: {}... sample logs: {}...'.format(diff_data_dict, sample_logs))
        print ('[DB...BAT] Data dict type: {}.  Keys: {}'.format(type(diff_data_dict), diff_data_dict.keys()))
        print ('[DB...BAT] Sample log dict keys: {}'.format(sample_logs.keys()))

        if 'peak_fit' not in sample_logs:
            raise RuntimeError('File {} does not have fitted peak parameters value for strain/stress '
                               'calculation'.format(file_name))

        # assign data file to files
        self._data_set_dict[direction] = diff_data_dict
        self._peak_param_dict[direction] = sample_logs['peak_fit']
        self._sample_log_dict[direction] = sample_logs

        # record data file name
        self._source_file_dict[direction] = file_name

        return

    def save_session(self, save_file_name):
        """
        save the complete session in order to open later
        :param save_file_name:
        :return:
        """
        # TODO - 2018 - NEXT
        raise NotImplementedError('ASAP')

    def set_d0(self, d0):
        """

        :param d0:
        :return:
        """
        pyrs.utilities.checkdatatypes.check_float_variable('d0', d0, (1E-4, None))

        self._d0 = d0

        return

    def set_2theta(self, twotheta):
        """

        :param twotheta:
        :return:
        """
        pyrs.utilities.checkdatatypes.check_float_variable('Detector 2theta', twotheta, (-180, 180))

        self._2theta = twotheta

        return

    def set_wave_length(self, wave_length):
        """

        :param wave_length:
        :return:
        """
        pyrs.utilities.checkdatatypes.check_float_variable('Wave length', wave_length, (1E-10, None))

        self._lambda = wave_length

        return

    def set_youngs_modulus(self, young_e):
        """

        :param young_e:
        :return:
        """
        self._young_e = young_e

    def set_poisson_ratio(self, poisson_ratio):
        """

        :param poisson_ratio:
        :return:
        """
        self._poisson_nu = poisson_ratio

    @property
    def session(self):
        """
        get session name
        :param self:
        :return:
        """
        return self._session
