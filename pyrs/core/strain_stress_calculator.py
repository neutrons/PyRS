import numpy
import math
import pyrs.utilities.checkdatatypes
import scandataio
from scipy.interpolate import griddata


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
    # vx, vy, vz, Delta Weld, Delta Thickness, Delta Length, Cuboid: provided by HB2B team
    allowed_grid_position_sample_names = ['vx', 'vy', 'vz', 'delta weld', 'delta thickness', 'delta length',
                                          'sx', 'sy']
    allowed_grid_position_sample_names_wild = ['cuboid*', 'sz*']
    # FIXME - sx, sy, sz* are for testing data only!

    # TODO - 20180922 - Improvements
    # 1. add an integer flag for processing status
    #    0: initial state
    #    1: all file loaded
    #    2: alignment preparation data structure ready
    #    3: grid alignment is set up
    #    4: strain and stress are calculated

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

        # strain/stress workflow tracker
        #    0: initial state
        #    1: all file loaded
        #    2: alignment preparation data structure ready
        #    3: grid alignment is set up
        #    4: strain and stress are calculated
        self._workflow_tracker = 0

        # class variable
        # vector of strain and stress matrix
        self._strain_matrix_vec = None  # strain_matrix_vec
        self._stress_matrix_vec = None  # stress_matrix_vec

        # session and strain/stress type (stage 0 class variables)
        self._session = session_name
        self._is_plane_strain = plane_strain
        self._is_plane_stress = plane_stress
        self._direction_list = ['e11', 'e22', 'e33']
        if self._is_plane_strain or self._is_plane_stress:
            self._direction_list.pop(2)

        # data sets, sample logs and peak parameters required (stage 1 class variables)
        self._data_set_dict = dict()    # [dir][scan log index] = vector
        self._peak_param_dict = dict()  # [dir][scan (peak) parameter name][scan log index] =
        self._sample_log_dict = dict()  # [dir][log name][scan log index] = value
        for dir_i in self._direction_list:
            self._data_set_dict[dir_i] = None
            self._peak_param_dict[dir_i] = None   # [dir][parameter name][scan log index]
            self._sample_log_dict[dir_i] = None   # [dir][log name][scan log index] = value

        # source files
        self._source_file_dict = dict()
        for dir_i in self._direction_list:
            self._source_file_dict[dir_i] = None

        # transformed data set
        self._dir_grid_pos_scan_index_dict = dict()   # [e11/e22/e33][grid pos][scan log index]
        for dir_i in self._direction_list:
            self._dir_grid_pos_scan_index_dict[dir_i] = None  # (value) a dictionary: key = sample position (tuple),
            #                                                   value = scan log index

        # mapped/interpolated parameters
        self._aligned_param_dict = dict()   # [param name][output-grid index, ss-direction-index] = value

        # flag whether the measured sample points can be aligned. otherwise, more complicated algorithm is required
        # including searching and interpolation
        self._sample_points_aligned = False

        # list of sample positions for each data set for grids
        self._grid_pos_x_name_dict = dict()  # dict[e11/e22/e33] = 'sx'
        self._grid_pos_y_name_dict = dict()
        self._grid_pos_z_name_dict = dict()

        self._sample_positions_dict = dict()    # dict[e11/e22/e33][i] = grid_i(x, y, z)  # shape=(n, 3) sorted
        for dir_i in self._direction_list:
            self._sample_positions_dict[dir_i] = None
        self._grid_statistics_dict = None
        self._grid_output_array = None  # array (of vector) for grids used by strain/stress calculation
        self._matched_grid_scans_list = None   # list[i] = {'e11': scan log index, 'e22': scan log index, ....

        # status
        self._is_saved = False

        # file loader (static kind of)
        self._file_io = scandataio.DiffractionDataFile()

        # strain stress parameters
        self._d0 = None
        # self._2theta = None

        self._young_e = None
        self._poisson_nu = None

        # grid matching/alignment
        self._match11_dict = None  # [dir: e22/e33][e11 scan log index] = e22/33 scan log index: matched to grid e11

        return

    def _set_workflow_state(self, at_init=False, data_loaded=False, alignment_setup=False, calculated=False):
        """ Set the strain/stress calculation workflow state
        # strain/stress workflow tracker
        #    0: initial state
        #    1: all file loaded
        #    2: alignment preparation data structure ready
        #    3: grid alignment is set up
        #    4: strain and stress are calculated
        self._workflow_tracker = 0
        :param at_init:
        :param data_loaded:
        :param alignment_setup:
        :param calculated:
        :return:
        """
        # check input
        sum_bool = int(at_init) + int(data_loaded) + int(alignment_setup) + int(calculated)
        if sum_bool == 0:
            raise RuntimeError('None state is selected to be True')
        elif sum_bool > 1:
            raise RuntimeError('More than 1 state ({}, {}, {}, {}) are set to True causing '
                               'ambiguous.'.format(at_init, data_loaded, alignment_setup, calculated))

        # set up
        if at_init:
            self._workflow_tracker = 0
        elif data_loaded:
            self._workflow_tracker = 1
        elif alignment_setup:
            self._workflow_tracker = 2
        else:
            self._workflow_tracker = 3

        return

    def _copy_grids(self, direction, grids_dimension_dict):
        """
        copy grids but limited by dimension
        :param direction:
        :param grids_dimension_dict:
        :return: numpy.ndarray (n, 3)
        """
        pyrs.utilities.checkdatatypes.check_string_variable('Strain direction', direction, self._direction_list)

        # get list of positions and convert to numpy array
        grids = numpy.array(self._sample_positions_dict[direction])

        # minimum
        pos_dir_list = sorted(grids_dimension_dict['Min'])
        for i_dir, pos_dir in enumerate(pos_dir_list):
            # get min value for this direction
            min_value_i = grids_dimension_dict['Min'][pos_dir]
            print ('[DB...BAT] {}/{} = {}'.format(i_dir, pos_dir, min_value_i))
            if min_value_i is None:
                continue
            # filter
            grids = grids[grids[:, i_dir] >= min_value_i]
            print ('[DB...BAT] After filtering >= {} at {}: Shape = {}'.format(min_value_i, pos_dir, grids.shape))
        # END-FOR

        # maximum
        for i_dir, pos_dir in enumerate(pos_dir_list):
            print ('[DB...BAT] {}/{} = {}'.format(i_dir, pos_dir, grids_dimension_dict['Max'][pos_dir]))
        for i_dir, pos_dir in enumerate(pos_dir_list):
            # get min value for this direction
            max_value_i = grids_dimension_dict['Max'][pos_dir]
            print ('[DB...BAT] {}/{} = {}'.format(i_dir, pos_dir, max_value_i))
            if max_value_i is None:
                continue
            # filter
            grids = grids[grids[:, i_dir] <= max_value_i]
            print ('[DB...BAT] After filtering <= {} at {}: Shape = {}'.format(max_value_i, pos_dir, grids.shape))
        # END-FOR

        return grids

    @staticmethod
    def _generate_slice_view_grids(grids_dimension_dict):
        """ Generate a new strain/stress grid from user-specified dimension FOR plotting (NOT FOR calculating)
        :param grids_dimension_dict:
        :return: numpy.ndarray (n, 3)
        """
        # check input
        pyrs.utilities.checkdatatypes.check_dict('Strain/stress grid setup', grids_dimension_dict)

        # start from direction to find out the dimension of the
        size_dict = dict()
        num_grids = 1
        for dir_i in ['X', 'Y', 'Z']:
            min_i = grids_dimension_dict['Min'][dir_i]
            max_i = grids_dimension_dict['Max'][dir_i]
            if min_i == max_i:
                num_pt_i = 1
            elif min_i < max_i:
                res_i = grids_dimension_dict['Resolution'][dir_i]
                if res_i is None:
                    raise RuntimeError('Resolution of direction {} is None which is not allowed when min {} ({}) '
                                       '<> max {} ({})'
                                       .format(dir_i, dir_i, min_i, dir_i, max_i))
                assert res_i > 0, 'Resolution cannot be 0 or negative'
                num_pt_i = int((max_i - min_i) / res_i) + 1
            else:
                raise RuntimeError('It is not allowed to have Min({0}) {1} > Max ({0}) {2}'
                                   .format(dir_i, min_i, max_i))

            num_grids *= num_pt_i
            size_dict[dir_i] = num_pt_i
        # END-FOR

        print ('[DB...BAT] Size: {}'.format(size_dict))

        # define grids vector
        grids_vec = numpy.ndarray(shape=(num_grids, 3), dtype='float')

        # set up grids
        global_index = 0
        for index_x in range(size_dict['X']):
            x_i = grids_dimension_dict['Min']['X'] + grids_dimension_dict['Resolution']['X'] * float(index_x)
            for index_y in range(size_dict['Y']):
                print ('[DB...BAT] {} + {} * {}'.format(grids_dimension_dict['Min']['Y'],
                                                        grids_dimension_dict['Resolution']['Y'],
                                                        index_y))
                if grids_dimension_dict['Resolution']['Y'] is None:
                    y_i = grids_dimension_dict['Min']['Y']
                else:
                    y_i = grids_dimension_dict['Min']['Y'] + grids_dimension_dict['Resolution']['Y'] * float(index_y)
                for index_z in range(size_dict['Z']):
                    z_i = grids_dimension_dict['Min']['Z'] + grids_dimension_dict['Resolution']['Z'] * float(index_z)

                    grids_vec[global_index, 0] = x_i
                    grids_vec[global_index, 1] = y_i
                    grids_vec[global_index, 2] = z_i

                    global_index += 1
                # END-FOR (z)
            # END-FOR (y)
        # END-FOR (x)

        return grids_vec

    # TODO FIXME - 20181001 - Temporarily disabled in order to clean up for the new workflow
    # TODO                    Grid won't be aligned but will be mapped for matched among E11/E22/E33
    # def align_grids(self, direction, user_defined, grids_dimension_dict):
    #     """ Align the input grids against the output strain/stress output.
    #     From the user specified output grid set up (ranges and solutions),
    #     1. the output grid in 3D array will be created.
    #     2. a mapping from the existing experiment grid to output grid (for scan logs) will be generated
    #     :param direction:
    #     :param user_defined:
    #     :param grids_dimension_dict: user specified output grid dimensions
    #     :return: 2-tuple: (1) grids (vector of position: (n, 3) array) (2) scan log map (vector of scan logs: (n, 2)
    #                                                                        or (n, 3) int array)
    #     """
    #     # get the grids for strain/stress calculation
    #     if user_defined and direction is not None:
    #         raise RuntimeError('It is not allowed to have both direction {} and user_defined {}'
    #                            .format(direction, user_defined))
    #     elif user_defined:
    #         self._grid_output_array = self._generate_grids(grids_dimension_dict)
    #     elif direction is not None:
    #         self._grid_output_array = self._copy_grids(direction, grids_dimension_dict)
    #     else:
    #         raise RuntimeError('Either direction or user-defined must be specified.')
    #
    #     num_ss_dir = len(self._direction_list)
    #     num_grids = self._grid_output_array.shape[0]
    #     mapping_vector = numpy.ndarray(shape=(num_grids, num_ss_dir), dtype=int)
    #     for i_grid in range(num_grids):
    #         ss_grid_i = self._grid_output_array[i_grid]
    #         for i_dir, ss_dir in enumerate(self._direction_list):
    #             # get the sorted positions
    #             sorted_grid_pos_vector = self._sample_positions_dict[ss_dir]
    #             if ss_dir == direction:
    #                 index_i = self.binary_search(sorted_positions=sorted_grid_pos_vector, xyz=ss_grid_i,
    #                                              resolution=1.E-10)
    #                 if index_i is None:
    #                     raise NotImplementedError('Impossible')
    #
    #             else:
    #                 index_i = self.binary_search(sorted_positions=sorted_grid_pos_vector, xyz=ss_grid_i,
    #                                              resolution=0.001)
    #
    #             # END-IF-ELSE
    #
    #             # convert index in the (e11/e22/e33) grid position list to scan log index
    #             if index_i is None:
    #                 scan_log_index_i = -1
    #             else:
    #                 exact_pos = sorted_grid_pos_vector[index_i]
    #                 scan_log_index_i = self._dir_grid_pos_scan_index_dict[ss_dir][tuple(exact_pos)]
    #
    #             mapping_vector[i_grid, i_dir] = scan_log_index_i
    #         # END-FOR
    #     # END-FOR
    #
    #     return self._grid_output_array, mapping_vector

    def located_matched_grids(self, resolution=0.001):
        """ Compare the grids among 3 (or 2) strain directions in order to search matched grids across
        :param resolution:
        :return:
        """
        # define a data structure for completely matched grids among all directions
        self._matched_grid_scans_list = list()   # list[i] = {'e11': scan log index, 'e22': scan log index, ....

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
            match_grid_dict = {'e11': self._dir_grid_pos_scan_index_dict['e11'][tuple(pos_11_i)]}
            both_matched = True
            for dir_i in other_dir_list:
                sorted_pos_list_i = self._sample_positions_dict[dir_i]
                index_i = self.binary_search(sorted_pos_list_i, pos_11_i, resolution)
                if index_i is None:
                    both_matched = False
                elif both_matched:
                    pos_dir_i = sorted_pos_list_i[index_i]
                    match_grid_dict[dir_i] = self._dir_grid_pos_scan_index_dict[dir_i][tuple(pos_dir_i)]
                # END-IF-ELSE

                # message
                if index_i is None:
                    message += '[DB...BAT] E11 Pt {} @ {} no match at direction {}\n'.format(ipt_e11, pos_11_i, dir_i)
                else:
                    message += '[DB...BAT] E11 Pt {} @ {} finds {} @ index = {} @ {}\n' \
                               ''.format(ipt_e11, pos_11_i, dir_i, index_i, sorted_pos_list_i[index_i])
                if index_i is not None:
                    self._match11_dict[dir_i][ipt_e11] = index_i
                    other_dir_matched_dict[dir_i].add(index_i)
                # END-FOR
            # END-FOR

            # store
            if both_matched:
                self._matched_grid_scans_list.append(match_grid_dict)
            # END-IF
        # END-FOR

        print ('[INFO] Matched Grids Points (For Strain/Stress Calculation): {}'
               ''.format(len(self._matched_grid_scans_list)))
        for dir_i in self._direction_list:
            unmatched_counts_i = len(self._sample_positions_dict[dir_i]) - len(self._matched_grid_scans_list)
            print ('[INFO] Numbers of grids at direction {} not matched with other directions: {}'
                   ''.format(dir_i, unmatched_counts_i))

        return

    # TODO FIXME - 20181001 - Temporarily disabled in order to clean up for the new workflow
    # TODO                    This method may be modified to calculate slice view plot from certain parameters
    # def align_peak_parameter_on_grids(self, grids_vector, parameter, scan_log_map_vector):
    #     """ align the parameter's values on a given grid
    #     [3D interpolation]
    #     1. regular grid interpolation CANNOT be used.
    #        (https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.interpolate.RegularGridInterpolator.html)
    #        It requires the grid to be interpolated from, i.e., experimental data, to be on REGULAR grid.
    #        It is NOT always TRUE in the experiment.
    #     2. imaging map: I don't understand it completely
    #        (map_coordinates)
    #     :param grids_vector: vector of grids that will have the parameter to be written on
    #     :param parameter: string, such as 'center_d'
    #     :param scan_log_map_vector: for grid[i], if map[i][1] is integer, than e22 has a grid matched. otherwise,
    #                                 an interpolation is required
    #     :return: 1D vector.  shape[0] = grids_vector.shape[0]
    #     """
    #     # check input
    #     pyrs.utilities.checkdatatypes.check_string_variable('Parameter name', parameter)
    #     pyrs.utilities.checkdatatypes.check_numpy_arrays('Grids vector', [grids_vector], dimension=2,
    #                                                      check_same_shape=False)
    #
    #     # define number of grids
    #     num_grids = grids_vector.shape[0]
    #     num_ss_dir = len(self._direction_list)
    #     param_vector = numpy.ndarray(shape=(num_grids, num_ss_dir), dtype=float)
    #
    #     for i_grid in range(num_grids):
    #         # grid_pos = grids_vector[i_grid]
    #
    #         for i_dir, ss_dir in enumerate(self._direction_list):
    #             # check whether for direction e??, there is a matched experimental grid
    #             scan_log_index_i = scan_log_map_vector[i_grid][i_dir]
    #             if isinstance(scan_log_index_i, int) and scan_log_index_i >= 0:
    #                 # there is a matched experimental grid
    #                 # TODO/FIXME: this is only for peak parameter values
    #                 # print (self._peak_param_dict[ss_dir].keys())
    #                 param_value = self._peak_param_dict[ss_dir][parameter][scan_log_index_i]
    #             else:
    #                 param_value = self.interpolate3d(self._sample_positions_dict[ss_dir],
    #                                                  self._peak_param_dict[ss_dir][parameter], grids_vector[i_grid])
    #             # END-IF-ELSE
    #             param_vector[i_grid, i_dir] = param_value
    #         # END-FOR
    #     # END-FOR
    #
    #     self._aligned_param_dict[parameter] = param_vector
    #
    #     return param_vector

    @staticmethod
    def binary_search(sorted_positions, xyz, resolution):
        """ do binary search in a sorted 2D numpy array
        :param sorted_positions:
        :param xyz:
        :param resolution: resolution of distance to grid such that can be treated as a single number
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
        if isinstance(sorted_positions, list):
            i_stop = len(sorted_positions) - 1
        elif isinstance(sorted_positions, numpy.ndarray):
            i_stop = sorted_positions.shape[0] - 1
        else:
            raise RuntimeError('Sorted position of type {} is not supported.'.format(type(sorted_positions)))

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

    def set_grid_log_names(self, pos_x_sample_names, pos_y_sample_names, pos_z_sample_names):
        """ set sample log names for grids
        :param pos_x_sample_names:
        :param pos_y_sample_names:
        :param pos_z_sample_names:
        :return:
        """
        pyrs.utilities.checkdatatypes.check_dict('Sample log names for grid position X', pos_x_sample_names)
        pyrs.utilities.checkdatatypes.check_dict('Sample log names for grid position X', pos_y_sample_names)
        pyrs.utilities.checkdatatypes.check_dict('Sample log names for grid position X', pos_z_sample_names)

        # Set up the X, Y, Z position sample log name for every direction
        # go through all the data to check
        for dir_i in self._direction_list:
            self._grid_pos_x_name_dict[dir_i] = pos_x_sample_names[dir_i]
            self._grid_pos_y_name_dict[dir_i] = pos_y_sample_names[dir_i]
            self._grid_pos_z_name_dict[dir_i] = pos_z_sample_names[dir_i]
        # END-FOR

        return

    def check_grids_alignment(self, resolution=0.001):
        """ Align the data points among e11, e22 and/or e33 with sample log positions
        :param resolution:
        :return: 2-tuple: boolean (perfectly matched), mismatched information
        """
        # (1) Check whether position X, Y, Z shall be already set and
        # (2) Generate grid mapping scan log dictionary
        for dir_i in self._direction_list:
            try:
                pos_x = self._grid_pos_x_name_dict[dir_i]
                pos_y = self._grid_pos_y_name_dict[dir_i]
                pos_z = self._grid_pos_z_name_dict[dir_i]
            except KeyError as key_err:
                raise RuntimeError('Grid position dictionary has not been set up for direction {}: {}'
                                   ''.format(dir_i, key_err))
            if pos_x == pos_y or pos_y == pos_z or pos_x == pos_z:
                raise RuntimeError('Position X ({}) Y ({}) and Z ({}) have duplicate sample log names.'
                                   ''.format(pos_x, pos_y, pos_z))

            # create the dictionaries, vectors and etc for checking how matching the grids are
            self._dir_grid_pos_scan_index_dict[dir_i] = self.generate_xyz_scan_log_dict(dir_i, pos_x, pos_y, pos_z)
        # END-FOR

        # align: create a list of sorted tuples and compare among different data sets whether they
        # do match or not
        for dir_i in self._direction_list:
            # each entry shall be a sorted numpy array (but not list as before)
            self._sample_positions_dict[dir_i] = numpy.array(sorted(self._dir_grid_pos_scan_index_dict[dir_i].keys()))

        self._set_grid_statistics()

        # set flag
        self._sample_points_aligned = False

        # check whether the sample position numbers are different
        info = ''
        num_dir = len(self._direction_list)
        for i_dir_i in range(num_dir):
            dir_i = self._direction_list[i_dir_i]
            for i_dir_j in range(i_dir_i, num_dir):
                dir_j = self._direction_list[i_dir_j]
                if self._sample_positions_dict[dir_i].shape[0] != self._sample_positions_dict[dir_j].shape[0]:
                    info = 'The number of grid points among different direction are different.'
                    break

        # END-FOR

        # check whether all the data points matched with each other within resolution
        if info == '':
            num_sample_points = self._sample_positions_dict[self._direction_list[0]].shape[0]
            for ipt in range(num_sample_points):
                max_distance = self.calculate_max_distance(ipt)
                if max_distance > resolution:
                    err_msg = '{}-th (of total {}) sample position point: '.format(ipt, num_sample_points)
                    for dir_i in self._direction_list:
                        err_msg += '{} @ {}; '.format(dir_i, self._sample_positions_dict[dir_i][ipt])
                    err_msg += ' with maximum distance {} over specified resolution {}'.format(max_distance, resolution)
                    info = err_msg
                    break
            # END-FOR
        # END-IF

        return info == '', info

    # TestMe (new) - 20180815 - Tested and convert to a method to store the result for table output (***)
    def _set_grid_statistics(self):
        """ set grid statistics
        :return:
        """
        # reset the original dictionary
        self._grid_statistics_dict = dict()
        for param_name in ['min', 'max', 'num_indv_values']:
            self._grid_statistics_dict[param_name] = dict()
            for dir_i in self._direction_list:
                self._grid_statistics_dict[param_name][dir_i] = dict()
        # END-FOR

        for dir_i in self._direction_list:
            # do statistics
            grid_pos_vec = numpy.array(self._sample_positions_dict[dir_i])
            for i_coord, coord_name in enumerate(['X', 'Y', 'Z']):
                # for x, y, z
                min_i = numpy.min(grid_pos_vec[:, i_coord])
                max_i = numpy.max(grid_pos_vec[:, i_coord])
                num_individual_i = len(set(grid_pos_vec[:, i_coord]))
                self._grid_statistics_dict['min'][dir_i][coord_name] = min_i
                self._grid_statistics_dict['max'][dir_i][coord_name] = max_i
                self._grid_statistics_dict['num_indv_values'][dir_i][coord_name] = num_individual_i
                print ('[DB...BAT] Direction {}: Coordinate {}, Min = {}, Max = {}, Number of individual = {}'
                       .format(dir_i, i_coord, min_i, max_i, num_individual_i))
            # END-FOR
        # END-FOR

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

    def convert_peaks_positions(self):
        """ convert all peaks' positions in d-space.
        Note: this must be called after check_grids() is called
        The convert peak positions shall be still recorded in "self._peak_param_dict"
        example: self._peak_param_dict[dir_i]['centre'][scan_log_index_dict[dir_i]]
        :return:
        """
        for i_dir, ss_dir in enumerate(self._direction_list):
            # create the vector
            num_pts = len(self._peak_param_dict[ss_dir]['centre'])
            self._peak_param_dict[ss_dir]['center_d'] = numpy.ndarray(shape=(num_pts,), dtype='float')

            # convert peak center from 2theta to d
            for scan_log_index in range(num_pts):
                peak_i_2theta = self._peak_param_dict[ss_dir]['centre'][scan_log_index]
                lambda_i = self._sample_log_dict[ss_dir]['Wavelength'][scan_log_index]
                peak_i_d = lambda_i * 0.5 / math.sin(peak_i_2theta * 0.5 * math.pi / 180.)
                self._peak_param_dict[ss_dir]['center_d'][scan_log_index] = peak_i_d
            # END-FOR
        # END-FOR

        return

    def migrate(self, session_name, plane_stress=False, plane_strain=False):
        """ migrate current strain/stress calculator to another one with (supposed to be) different
        type of strain/stress
        :param session_name:
        :param plane_stress:
        :param plane_strain:
        :return:
        """
        new_ss_calculator = StrainStressCalculator(session_name, plane_stress,plane_stress)

        # copy file format: try not to re-load the file
        new_ss_calculator._peak_param_dict = self._peak_param_dict.copy()
        new_ss_calculator._sample_log_dict = self._sample_log_dict.copy()

        # copy file name and etc
        new_ss_calculator._file_dict = self._file_dict.copy()

        return

    def calculate_max_distance(self, sample_point_index):
        """
        in the case that all 2 or 3
        :param sample_point_index:
        :return:
        """
        pyrs.utilities.checkdatatypes.check_int_variable('Sample point index', sample_point_index,
                                                         (0, self._sample_positions_dict['e11'].shape[0]))

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
        calculate strain/stress on the final output grids
        :return:
        """
        # prepare the data structure
        num_grids = len(self._matched_grid_scans_list)
        strain_matrix_vec = numpy.ndarray(shape=(num_grids, 3, 3), dtype='float')
        stress_matrix_vec = numpy.ndarray(shape=(num_grids, 3, 3), dtype='float')
        self._grid_output_array = numpy.ndarray(shape=(num_grids, 3), dtype='float')

        # loop over all the matched grids
        for grid_index, dir_scan_dict in enumerate(self._matched_grid_scans_list):
            # create peak matrix (3X3)
            peak_matrix = numpy.zeros(shape=(3, 3), dtype='float')
            for dir_index, dir_name in enumerate(self._direction_list):
                scan_log_index = dir_scan_dict[dir_name]
                peak_d_pos = self._peak_param_dict[dir_name]['center_d'][scan_log_index]
                peak_matrix[dir_index, dir_index] = peak_d_pos
            # END-FOR

            # calculate strain/stress
            try:
                ss_calculator = StrainStress(peak_pos_matrix=peak_matrix,
                                             d0=self._d0, young_modulus=self._young_e,
                                             poisson_ratio=self._poisson_nu,
                                             is_plane_train=self._is_plane_strain,
                                             is_plane_stress=self._is_plane_stress)
            except ZeroDivisionError as err:
                err_msg = 'Strain/stress calculation parameter set up error causing zero division: {}'.format(err)
                raise RuntimeError(err_msg)

            strain_matrix_vec[grid_index] = ss_calculator.get_strain()
            stress_matrix_vec[grid_index] = ss_calculator.get_stress()

            # set up the grid positions
            grid_x = self._sample_log_dict['e11'][self._grid_pos_x_name_dict['e11']][dir_scan_dict['e11']]
            grid_y = self._sample_log_dict['e11'][self._grid_pos_y_name_dict['e11']][dir_scan_dict['e11']]
            grid_z = self._sample_log_dict['e11'][self._grid_pos_z_name_dict['e11']][dir_scan_dict['e11']]

            self._grid_output_array[grid_index] = numpy.array((grid_x, grid_y, grid_z))
        # END-FOR

        # set to the class variable
        self._strain_matrix_vec = strain_matrix_vec
        self._stress_matrix_vec = stress_matrix_vec

        return strain_matrix_vec, stress_matrix_vec

    def execute_workflow(self):
        """ Execute the strain/stress workflow
        It is usually called after copying a StrainStressCalculator to a new one as the strain/stress
        type is changed.
        :return:
        """
        # check whether the files are loaded
        files_loaded = self._check_file_loaded()
        if files_loaded is False:
            return self._workflow_tracker

        # check alignment
        try:
            self.check_grids_alignment(resolution=0.001)
        except RuntimeError:
            self._set_workflow_state(files_loaded)
            return self._workflow_tracker

        # calculate strain and stress
        try:
            self.execute()
        except RuntimeError:
            self._set_workflow_state(alignment_setup=True)
            return self._workflow_tracker

        return self._workflow_tracker

    def execute_old(self):
        """
        calculate the strain and stress for all grids by using E11 as a standard
        taken out because the workflow is changed!
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

            grid_pos_e11 = self._sample_positions_dict['e11'][ipt_e11]
            scan_log_index_dict['e11'] = self._dir_grid_pos_scan_index_dict['e11'][grid_pos_e11]

            grid_pos_e22 = self._sample_positions_dict['e22'][ipt_e22]
            scan_log_index_dict['e22'] = self._dir_grid_pos_scan_index_dict['e22'][grid_pos_e22]

            debug_out = 'e11: scan-index = {} @ {}, e22: scan-index = {} @ {}, ' \
                        ''.format(scan_log_index_dict['e11'], grid_pos_e11,
                                  scan_log_index_dict['e22'], grid_pos_e22)
            if isinstance(ipt_e33, int):
                pos_e33 = self._sample_positions_dict['e33'][ipt_e33]
                scan_log_index_dict['e33'] = self._dir_grid_pos_scan_index_dict['e33'][pos_e33]
                debug_out += 'e33: scan-index = {} @ {}, ' \
                             ''.format(scan_log_index_dict['e33'], pos_e33)
            print (debug_out)

            # calculate peak positions in d-spacing
            peak_matrix = numpy.zeros(shape=(3, 3), dtype='float')
            peak_fit_failed = False
            for mindex, dir_i in enumerate(self._direction_list):
                peak_i_2theta = self._peak_param_dict[dir_i]['centre'][scan_log_index_dict[dir_i]]
                lambda_i = self._sample_log_dict[dir_i]['Wavelength'][scan_log_index_dict[dir_i]]

                print ('[DB...BAT] Direction: {}. Log index = {}; Peak center (2theta): {}  degree, wavelength: {}'
                       ''.format(dir_i, scan_log_index_dict[dir_i], peak_i_2theta, lambda_i))
                if abs(peak_i_2theta) < 1:
                    peak_fit_failed = True
                    continue
                peak_i_d = lambda_i * 0.5 / math.sin(peak_i_2theta * 0.5)   # self.convert_unit_to_d(peak_i_2theta)
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

    def export_2d_slice(self, param_name, is_grid_raw, ss_direction, slice_dir, slice_pos, slice_resolution, file_name):
        """
        export a 2D slice from either raw grids from experiments or grids aligned
        :param param_name:
        :param is_grid_raw:
        :param ss_direction:
        :param slice_dir:
        :param slice_pos:
        :param slice_resolution:
        :param file_name:
        :return:
        """
        print (param_name, is_grid_raw, ss_direction, slice_dir, slice_pos, slice_resolution, file_name)
        # check inputs
        pyrs.utilities.checkdatatypes.check_bool_variable('Flag to show whether the grids are raw experimental ones',
                                                          is_grid_raw)

        # convert the girds and parameter values to a shape=(n, 4) array
        if is_grid_raw:
            param_grid_array = self.convert_raw_grid_value_to_matrix(param_name, ss_direction)
        else:
            param_grid_array = self.convert_user_grid_value_to_matrix(param_name, ss_direction)

        # slice the (n, 4) array at 1 direction
        sliced_grid_array = self.slice_md_data(param_grid_array, slice_dir, slice_pos, slice_resolution)

        # export result to hdf5
        scandataio.export_md_array_hdf5(sliced_grid_array, [slice_dir], file_name)

        return

    def convert_raw_grid_value_to_matrix(self, param_name, ss_direction):
        """
        convert the grid position and specified parameter value to a (n, 4) 2D array
        :param param_name:
        :param ss_direction:
        :return:
        """
        # check inputs
        pyrs.utilities.checkdatatypes.check_string_variable('Strain/stress direction', ss_direction,
                                                            self._direction_list)
        pyrs.utilities.checkdatatypes.check_string_variable('Parameter name', param_name,
                                                            allowed_values=self._peak_param_dict[ss_direction].keys())

        num_grids = len(self._dir_grid_pos_scan_index_dict[ss_direction])

        # create the 2D array
        param_grid_array = numpy.ndarray(shape=(num_grids, 4), dtype='float')
        grids_list = sorted(self._dir_grid_pos_scan_index_dict[ss_direction].keys())
        for i_grid, grid_pos in enumerate(grids_list):
            # position of grid on sample
            for i_coord in range(3):
                param_grid_array[i_grid][i_coord] = grid_pos[i_coord]
            # parameter value
            scan_log_index = self._dir_grid_pos_scan_index_dict[ss_direction][grid_pos]
            param_value = self._peak_param_dict[ss_direction][param_name][scan_log_index]
            param_grid_array[i_grid][3] = param_value
        # END-FOR

        return param_grid_array

    def convert_user_grid_value_to_matrix(self, param_name, ss_direction):
        """
        convert user-defined grid value to matrix
        :param param_name:
        :param ss_direction:
        :return:
        """
        # check inputs
        pyrs.utilities.checkdatatypes.check_string_variable('Strain/stress direction', ss_direction,
                                                            self._direction_list)
        pyrs.utilities.checkdatatypes.check_string_variable('Parameter name', param_name,
                                                            allowed_values=self._peak_param_dict[ss_direction].keys())

        # TODO - 20180823 - the non-existing data structure is to be corrected later with method to create user grids
        num_grids = len(self._grid_output_array[ss_direction])

        # create the 2D array
        param_grid_array = numpy.ndarray(shape=(num_grids, 4), dtype='float')
        grids_list = sorted(self._grid_output_array[ss_direction].keys())
        for i_grid, grid_pos in enumerate(grids_list):
            # position of grid on sample
            for i_coord in range(3):
                param_grid_array[i_grid][i_coord] = grid_pos[i_coord]
            # parameter value
            scan_log_index = self._grid_output_array[ss_direction][grid_pos]
            param_value = self._grid_output_array
            param_grid_array[i_grid][3] = param_value
        # END-FOR

        return param_grid_array

    @staticmethod
    def slice_md_data(param_grid_array, slice_dir, slice_pos, slice_resolution):
        """ slice multi-dimensional data from a (n x m) matrix
        :param param_grid_array:
        :param slice_dir:
        :param slice_pos:
        :param slice_resolution:
        :return:
        """
        pyrs.utilities.checkdatatypes.check_numpy_arrays('Parameter value on grids', [param_grid_array], 2, False)
        pyrs.utilities.checkdatatypes.check_int_variable('Slicing direction', slice_dir, (0, 3))
        pyrs.utilities.checkdatatypes.check_float_variable('Slicing position', slice_pos, (None, None))
        pyrs.utilities.checkdatatypes.check_float_variable('Slicing resolution', slice_resolution, (0, None))

        min_value = slice_pos - slice_resolution
        max_value = slice_pos + slice_resolution

        print ('[DB...INFO Before slicing: size = {}'.format(param_grid_array.shape[0]))
        slice_array = param_grid_array[min_value <= param_grid_array[:, slice_dir]]
        slice_array = slice_array[slice_array[:, slice_dir] <= max_value]
        print ('[DB...INFO After  slicing: size = {}'.format(slice_array.shape[0]))

        return slice_array

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
            try:
                z_i = self._sample_log_dict[direction][pos_z][scan_log_index]
            except KeyError as key_err:
                err_msg = 'Direction {} Scan log index {}: Z-position sample log {} is not found. ' \
                          'Available logs include {}\nFYI (Error Message): {}' \
                          ''.format(direction, scan_log_index, pos_z, sorted(self._sample_log_dict[direction].keys()),
                                    key_err)
                raise KeyError(err_msg)
            xyz_log_index_dict[(x_i, y_i, z_i)] = scan_log_index
        # END-FOR

        return xyz_log_index_dict

    # TODO - 20180815 - Implement after (***)
    def get_finest_direction(self):
        """ Go through the grid statistics and find out the direction (e11/e22/e33) with the finest grid
        :return:
        """
        if self._grid_statistics_dict is None:
            raise RuntimeError('Grid statistics has not been calculated yet.')

        finest_score = 1E20
        finest_dir = None
        for i_dir, ss_dir in enumerate(self._direction_list):
            score_i = 0
            for coord_dir in range(3):
                min_i = self._grid_statistics_dict['min'][ss_dir][coord_dir]
                max_i = self._grid_statistics_dict['max'][ss_dir][coord_dir]
                num_pt = self._grid_statistics_dict['num_indv_values'][ss_dir][coord_dir]
                res_i = (max_i - min_i) /float(num_pt)
                score_i += res_i
            # END-FOR
            if score_i < finest_score:
                finest_score = score_i
                finest_dir = ss_dir
        # END-FOR

        return finest_dir

    def get_grids_information(self):
        """
        [note for output]
        level 1: type: min, max, num_indv_values
          level 2: direction: e11, e22(, e33)
            level 3: coordinate_dir: x, y, z
        get the imported grids' statistics information
        :return: dictionary (3-levels)
        """
        return self._grid_statistics_dict

    def get_peak_parameter_names(self):
        """
        get the name of peak parameters.  It shouldn't be any difference among directions
        :return:
        """
        arb_dir = self._direction_list[0]

        return self._peak_param_dict[arb_dir].keys()

    def get_sample_logs_names(self, direction, to_set):
        """
        get the sames of all the sample logs for one direction (e11/e22/e33)
        :param direction:
        :param to_set: if True, return in set, otherwise, return as a list
        :return:
        """
        # check input
        if direction not in self._direction_list:
            raise NotImplementedError('Direction {} is not an allowed direction {}'
                                      ''.format(direction, self._direction_list))

        log_names = self._sample_log_dict[direction].keys()

        if to_set:
            log_names = set(log_names)
        else:
            log_names.sort()

        return log_names

    def get_strain_stress_direction(self):
        """
        get the direction for strain and stress calculation
        :return: either [e11, e22] or [e11, e22, e33]
        """
        return self._direction_list[:]

    def get_strain_stress_values(self):
        """ get the corresponding grid output (matched grids), strain and stress list
        :return:
        """
        return self._grid_output_array, self._strain_matrix_vec, self._stress_matrix_vec

    def get_raw_grid_param_values(self, ss_direction, param_name):
        """ Get the parameter's value on raw experimental grid
        :param ss_direction:
        :param param_name:
        :return: dict: [grid position] = value
        """
        # check input
        pyrs.utilities.checkdatatypes.check_string_variable('Strain/stress direction', ss_direction,
                                                            self._direction_list)
        pyrs.utilities.checkdatatypes.check_string_variable('Parameter name', param_name,
                                                            allowed_values=self._peak_param_dict[ss_direction].keys())

        grid_param_dict = dict()
        # check parameter name
        for grid_pos in self._dir_grid_pos_scan_index_dict[ss_direction].keys():
            scan_log_index = self._dir_grid_pos_scan_index_dict[ss_direction][grid_pos]
            param_value = self._peak_param_dict[ss_direction][param_name][scan_log_index]
            grid_val_dict = {'scan-index': scan_log_index, 'value': param_value, 'dir': ss_direction}
            grid_param_dict[grid_pos] = grid_val_dict
        # END-FOR

        return grid_param_dict

    def get_user_grid_param_values(self, ss_direction, param_name):
        """ Get the mapped value to user defined grid
        :param ss_direction:
        :param param_name:
        :return: vector
        """
        # check input
        pyrs.utilities.checkdatatypes.check_string_variable('Strain/stress direction', ss_direction,
                                                            self._direction_list)
        pyrs.utilities.checkdatatypes.check_string_variable('Parameter name', param_name,
                                                            allowed_values=self._peak_param_dict[ss_direction].keys())

        # get the
        if param_name not in self._aligned_param_dict:
            raise RuntimeError('Parameter {} is not aligned to output/strain/stress grid yet. Available incude {}'
                               .format(param_name, self._aligned_param_dict.keys()))

        ss_dir_index = self._direction_list.index(ss_direction)

        param_value_array = self._aligned_param_dict[param_name][:, ss_dir_index]

        # for whatever in []:
        #     grid_val_dict[grid_pos] = {'scan-index': None, 'value': None, 'dir': ss_direction}
        #     grid_val_dict[grid_pos] = {'scan-index': None, 'e11': None, 'e22': None, 'e33': None}
        #
        #     return_list.append(grid_val_dict)
        # # END-FOR

        return param_value_array

    # TODO - 20180820 - Write a single unit test to find out how good or bad this algorithm is
    @staticmethod
    def interpolate3d(exp_grid_pos_vector, param_value_vector, target_position):
        """ interpolate a value in a 3D system
        :param exp_grid_pos_vector: 
        :param param_value_vector: 
        :param target_position: 
        :return: 
        """
        # check inputs
        # print ('[DB...BAT] Experimental grid position vector: type = {}'.format(type(exp_grid_pos_vector)))
        # print ('[DB...BAT] Experimental grid position vector: {}'.format(exp_grid_pos_vector))
        # print ('[DB...BAT] Experimental grid position vector: shape = {}'.format(exp_grid_pos_vector.shape))
        # print ('[DB...BAT] Parameter value vector: {}'.format(exp_grid_pos_vector))
        # print ('[DB...BAT] Parameter value shape: {}'.format(exp_grid_pos_vector.shape))

        num_exp_grids = exp_grid_pos_vector.shape[0]
        if num_exp_grids != len(param_value_vector):
            raise RuntimeError('Experimental grid positions have different number to parameter value vector '
                               '({} vs {})'.format(num_exp_grids, len(param_value_vector)))

        assert isinstance(target_position, numpy.ndarray) and target_position.shape == (3,),\
            'Target position must be a (3, 0) ndarray but not {} of type {}' \
            ''.format(target_position, type(target_position))

        # grid_x, grid_y, grid_z = (grid_x, grid_y, grid_z)
        target_position_input = numpy.array([target_position])
        print (param_value_vector.shape)
        print (target_position_input.shape)
        interp_value = griddata(exp_grid_pos_vector, param_value_vector, target_position_input, method='nearest')

        return interp_value

    @staticmethod
    def is_allowed_grid_position_sample_log(log_name):
        """ check whether the input are allowed sample grid position's log name in file
        :param log_name:
        :return:
        """
        print ('[DB...BAT] Log name: {} ... Comparing {}'
               ''.format(log_name, StrainStressCalculator.allowed_grid_position_sample_names))
        log_name = log_name.lower()
        if log_name in StrainStressCalculator.allowed_grid_position_sample_names:
            return True

        for wild_name in StrainStressCalculator.allowed_grid_position_sample_names_wild:
            if wild_name.endswith('*'):
                wild_name = wild_name.split('*')[0]
                if log_name.startswith(wild_name):
                    return True
            elif wild_name.startswith('*'):
                wild_name = wild_name.split('*')[1]
                if log_name.endswith(wild_name):
                    return True
            else:
                raise RuntimeError('Case {} not supported'.format(wild_name))
        # END-FOR

        return False

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

    def load_reduced_file(self, direction, file_name):
        """
        load previously reduced experimental file with peak fit result and sample logs
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
        # print ('[DB...BAT] Data dict type: {}.  Keys: {}'.format(type(diff_data_dict), diff_data_dict.keys()))
        # print ('[DB...BAT] Sample log dict keys: {}'.format(sample_logs.keys()))

        if 'peak_fit' not in sample_logs:
            raise RuntimeError('File {} does not have fitted peak parameters value for strain/stress '
                               'calculation'.format(file_name))
        if sample_logs['peak_fit'] is None:
            raise RuntimeError('File {} has empty "peak fit" in sample logs.'
                               ''.format(file_name))

        # assign data file to files
        self._data_set_dict[direction] = diff_data_dict
        self._peak_param_dict[direction] = sample_logs['peak_fit']
        self._sample_log_dict[direction] = sample_logs

        # record data file name
        self._source_file_dict[direction] = file_name

        return

    def rename(self, new_session_name):
        """
        rename session name
        :param new_session_name:
        :return:
        """
        pyrs.utilities.checkdatatypes.check_string_variable('New strain/stress calculator session name',
                                                            new_session_name)

        self._session = new_session_name

        return

    def save_session(self, save_file_name):
        """
        save the complete session in order to open later
        :param save_file_name:
        :return:
        """
        # TODO - 2018 - NEXT
        raise NotImplementedError('ASAP')

    def save_strain_stress(self, file_name):
        """

        :param file_name:
        :return:
        """
        if self._strain_matrix_vec is None or self._stress_matrix_vec is None:
            raise RuntimeError('Strain and stress have not been calculated.')

        csv_buffer = ''
        csv_buffer += '# {:8s}{:10s}{:10s}{:10s}{:10s}{:10s}{:10s}{:10s}{:10s}' \
                      ''.format('X', 'Y', 'Z', 'e11', 'e22', 'e33', 's11', 's22', 's33')

        num_strain_stress_grid = self._strain_matrix_vec.shape[0]
        for i_grid in range(num_strain_stress_grid):
            # append a line for i-th grid
            line_i = ''

            # positions
            for dir_i in range(3):
                line_i += '{:10s}'.format('{:.5f}'.format(self._grid_output_array[i_grid][dir_i]))

            # write strain
            for ii in range(3):
                line_i += '{:10s}'.format('{:.5f}'.format(self._strain_matrix_vec[i_grid][ii, ii]))

            # write stress
            for ii in range(3):
                line_i += '{:10s}'.format('{:.5f}'.format(self._stress_matrix_vec[i_grid][ii, ii]))

            line_i += '\n'
        # END-FOR
        csv_file = open(file_name, 'w')
        csv_file.write(csv_buffer)
        csv_file.close()

        return

    def set_d0(self, d0):
        """

        :param d0:
        :return:
        """
        pyrs.utilities.checkdatatypes.check_float_variable('d0', d0, (1E-4, None))

        self._d0 = d0

        return

    # def set_2theta(self, twotheta):
    #     """
    #
    #     :param twotheta:
    #     :return:
    #     """
    #     pyrs.utilities.checkdatatypes.check_float_variable('Detector 2theta', twotheta, (-180, 180))
    #
    #     self._2theta = twotheta
    #
    #     return

    # def set_wave_length(self, wave_length):
    #     """
    #
    #     :param wave_length:
    #     :return:
    #     """
    #     pyrs.utilities.checkdatatypes.check_float_variable('Wave length', wave_length, (1E-10, None))
    #
    #     self._lambda = wave_length
    #
    #     return

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
