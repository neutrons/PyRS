# Module containing extended TableWidgets for PyRS project
import NTableWidget
from pyrs.utilities import checkdatatypes


class FitResultTable(NTableWidget.NTableWidget):
    """
    A table tailored to peak fit result
    """
    # # TODO - The setup of this table shall be flexible to the peak type
    # #        considering base/advanced peak parameter for fitted value, uncertainties or both A+/-E
    # TableSetupList = [('Index', 'int'),
    #                   ('Center', 'float'),
    #                   ('Height', 'float'),
    #                   ('FWHM', 'float'),
    #                   ('Intensity', 'float'),
    #                   ('Chi^2', 'float'),
    #                   ('C.O.M', 'float'),  # center of mass
    #                   ('Profile', 'string')]

    def __init__(self, parent):
        """ Initialization
        """
        super(FitResultTable, self).__init__(parent)

        self._colIndexIndex = None
        self._colIndexCenter = None
        self._colIndexHeight = None
        self._colIndexWidth = None
        self._colIndexChi2 = None
        self._colIndexCoM = None
        self._colIndexProfile = None
        self._colIndexIntensity = None

        return

    def init_exp(self, index_list):
        """
        init the table for an experiment with a given list of scan indexes
        :param index_list:
        :return:
        """
        # TODO - Shall create a new module named as pyrs.utilities for utility methods used by both core and interface
        assert isinstance(index_list, list), 'blabla'

        for index in index_list:
            self.append_row([index, None, None, None, None, None, None, ''])

        return

    def reset_table(self, peak_param_names):
        """

        :param peak_param_names:
        :return:
        """


        self.clear()
        self.init_setup(table_setup_list)

        return

    def setup(self, peak_param_names):
        """
        Init setup
        :return:
        """
        # TODO - 20181205 - ASAP (0) - Replacing reset_column - Review commented out!

        # create table columns dynamically
        # table_setup_list = [('Index', 'int')]
        # for param_name in peak_param_names:
        #     table_setup_list.append((param_name, 'float'))
        # table_setup_list.append(('C.O.M', 'float'))
        # table_setup_list.append(('Profile', 'string'))
        #
        # total_columns = len(table_setup_list)
        # self._colIndexIndex = 0
        # self._colIndexProfile = total_columns - 1
        # self._colIndexCoM = total_columns - 2
        #
        # self.init_setup(self.TableSetupList)
        #
        # # Set up column width
        # self.setColumnWidth(0, 60)
        # self.setColumnWidth(1, 80)
        # self.setColumnWidth(2, 80)
        # self.setColumnWidth(3, 80)
        #
        # # Set up the column index for start, stop and select
        # self._colIndexIndex = self.TableSetupList.index(('Index', 'int'))
        # self._colIndexCenter = self.TableSetupList.index(('Center', 'float'))
        # self._colIndexHeight = self.TableSetupList.index(('Height', 'float'))
        # self._colIndexWidth = self.TableSetupList.index(('FWHM', 'float'))
        # self._colIndexIntensity = self.TableSetupList.index(('Intensity', 'float'))
        # self._colIndexChi2 = self.TableSetupList.index(('Chi^2', 'float'))
        # self._colIndexCoM = self.TableSetupList.index(('C.O.M', 'float'))
        # self._colIndexProfile = self.TableSetupList.index(('Profile', 'string'))

        return

    def set_peak_center_of_mass(self, row_number, com):
        """
        set the center of mass of a peak
        :param row_number:
        :param com:
        :return:
        """
        self.update_cell_value(row_number, self._colIndexCoM, com)

        return

    def set_peak_params(self, row_number, center, height, fwhm, intensity, chi2, profile):
        """
        set fitted peak parameters
        :param row_number:
        :param center:
        :param height:
        :param fwhm:
        :param intensity:
        :param chi2:
        :param profile:
        :return:
        """
        self.update_cell_value(row_number, self._colIndexCenter, center)
        self.update_cell_value(row_number, self._colIndexHeight, height)
        self.update_cell_value(row_number, self._colIndexWidth, fwhm)
        self.update_cell_value(row_number, self._colIndexChi2, chi2)
        self.update_cell_value(row_number, self._colIndexIntensity, intensity)
        self.update_cell_value(row_number, self._colIndexProfile, profile)

        return


class GridsStatisticsTable(NTableWidget.NTableWidget):
    """ Table for grid statistics
    """
    TableSetupList = [('Item', 'str'),   # include min x, max x, num x, avg resolution x, ... (for y) .. (for z)... # data points
                      ('e11', 'float'),
                      ('e22', 'float'),
                      ('e33', 'float')]

    def __init__(self, parent):
        """
        initialization
        :param parent:
        """
        super(GridsStatisticsTable, self).__init__(parent)

        self._indexItemName = 0
        self._indexDirDict = dict()
        for i_dir, dir_name in enumerate(['e11', 'e22', 'e33']):
            self._indexDirDict[dir_name] = i_dir + 1

        return

    def set_statistics(self, stat_dict, row_item_list):
        """

        :param stat_dict:
        :param row_item_list: list of item names for each row in order to maitain the order
        :return:
        """
        checkdatatypes.check_dict('Statistic dictionary', stat_dict)
        checkdatatypes.check_list('Row item names', row_item_list)

        # add rows to fill the table
        num_diff = len(row_item_list) - self.rowCount()
        if num_diff > 0:
            for i_row in range(num_diff):
                self.append_row(['', 0., 0., 0.])

        # fill the table
        for i_row, item_name in enumerate(row_item_list):
            self.update_cell_value(i_row, self._indexItemName, item_name)
            for dir_i in stat_dict.keys():
                self.update_cell_value(i_row, self._indexDirDict[dir_i], stat_dict[dir_i][item_name])
            # END-FOR (dir)
        # END-FOR (row)

        return

    def setup(self):
        """
        Init setup
        :return:
        """
        self.init_setup(self.TableSetupList)

        return


class GridAlignmentTable(NTableWidget.NTableWidget):
    """ Table to show how the grids from e11/e22/e33 are aligned
    The grids shown in the table are those for final strain/stress calculation.
    So they are not necessary same as any from e11/e22/e33
    """
    TableSetupList = [('x', 'float'),
                      ('y', 'float'),
                      ('z', 'float'),
                      ('e11', 'int'),  # scan log index of e11 direction data, -1 for not found
                      ('e22', 'int'),
                      ('e33', 'int')]

    def __init__(self, parent):
        """ Initialization
        :param parent:
        """
        super(GridAlignmentTable, self).__init__(parent)

        # indexes
        self._index_x = 0
        self._index_y = 1
        self._index_z = 2

        self._index_dir_dict = dict()
        for i_dir, dir_name in enumerate(['e11', 'e22', 'e33']):
            self._index_dir_dict[dir_name] = i_dir + 3

        return

    def add_grid(self, grid_pos_x, grid_pos_y, grid_pos_z, scan_index_e11, scan_index_e22, scan_index_e33):
        """
        add to grid
        :param grid_pos_x:
        :param grid_pos_y:
        :param grid_pos_z:
        :param scan_index_e11:
        :param scan_index_e22:
        :param scan_index_e33:
        :return:
        """
        if scan_index_e11 < 0:
            scan_index_e11 = None
        if scan_index_e22 < 0:
            scan_index_e22 = None
        if scan_index_e33 < 0:
            scan_index_e33 = None

        self.append_row([grid_pos_x, grid_pos_y, grid_pos_z, scan_index_e11, scan_index_e22, scan_index_e33])

        return

    def set_grids_values(self, grid_vector, grids_value_vec_dict):
        """

        :param grids_value_list:
        :return:
        """
        e11_values = grids_value_vec_dict['e11']

        if grid_vector.shape[0] != e11_values.shape[0]:
            raise RuntimeError('Grid vector size is not equal to parameter value size')

        for i_grid in range(grid_vector.shape[0]):
            grid_pos = grid_vector[i_grid]
            param_value = grids_value_vec_dict['e11'][i_grid]
            self.append_row([grid_pos[0], grid_pos[1], grid_pos[2], param_value, None, None])
        return

    def setup(self):
        """
        Init setup
        :return:
        """
        self.init_setup(self.TableSetupList)

        return


class MatchedGridsTable(NTableWidget.NTableWidget):
    """ Table for matched grids, i.e., across e11/e22 (plane stress/strain) or e11/e22/e33 (unconstrained)
    """
    TableSetupList = [('x', 'float'),
                      ('y', 'float'),
                      ('z', 'float'),
                      ('e11', 'int'),  # scan log index of e11 direction data
                      ('e22', 'int'),
                      ('e33', 'int')]

    def __init__(self, parent):
        """ initialization
        :param parent:
        """
        super(MatchedGridsTable, self).__init__(parent)

        return

    def add_matched_grid(self, grid_pos_x, grid_pos_y, grid_pos_z, scan_index_e11, scan_index_e22, scan_index_e33):
        """
        add an all-direction matched grid
        :param grid_pos_x:
        :param grid_pos_y:
        :param grid_pos_z:
        :param scan_index_e11:
        :param scan_index_e22:
        :param scan_index_e33:
        :return:
        """
        checkdatatypes.check_int_variable('Scan log index for e11', scan_index_e11, (1, None))
        checkdatatypes.check_int_variable('Scan log index for e22', scan_index_e22, (1, None))
        if scan_index_e33 is not None:
            checkdatatypes.check_int_variable('Scan log index for e33', scan_index_e33, (1, None))

        self.append_row([grid_pos_x, grid_pos_y, grid_pos_z, scan_index_e11, scan_index_e22, scan_index_e33])

        return

    def setup(self):
        """
        Init setup
        :return:
        """
        self.init_setup(self.TableSetupList)

        return


class MismatchedGridsTable(NTableWidget.NTableWidget):
    """ Table for completed misaligned/mismatched grids
    """
    TableSetupList = [('Direction', 'str'),
                      ('Scan Index', 'int'),
                      ('x', 'float'),
                      ('y', 'float'),
                      ('z', 'float')]

    def __init__(self, parent):
        """
        initialization
        :param parent:
        """
        super(MismatchedGridsTable, self).__init__(parent)

        return

    def add_mismatched_grid(self, direction, scan_index, grid_pos_x, grid_pos_y, grid_pos_z):
        """
        add a line for mismatched grid
        :param direction:
        :param scan_index:
        :param grid_pos_x:
        :param grid_pos_y:
        :param grid_pos_z:
        :return:
        """
        if direction not in ['e11', 'e22', 'e33']:
            raise RuntimeError('blabla')

        checkdatatypes.check_int_variable('Scan log index for {}'.format(direction), scan_index, (1, None))

        self.append_row([direction, scan_index, grid_pos_x, grid_pos_y, grid_pos_z])

        return

    def setup(self):
        """
        Init setup
        :return:
        """
        self.init_setup(self.TableSetupList)

        return
# END-CLASS


class ParamValueGridTable(NTableWidget.NTableWidget):
    """
    Parameter values on strain/stress grids
    """
    TableSetupList = [('x', 'float'),
                      ('y', 'float'),
                      ('z', 'float'),
                      ('e11', 'float'),  # scan log index of e11 direction data
                      ('e22', 'float'),
                      ('e33', 'float')]

    def __init__(self, parent):
        """ initialization
        :param parent:
        """
        super(ParamValueGridTable, self).__init__(parent)

        return

    def add_matched_grid(self, grid_pos_x, grid_pos_y, grid_pos_z, param_value_11, param_value_22, param_value_33):
        """
        add an all-direction matched grid
        :param grid_pos_x:
        :param grid_pos_y:
        :param grid_pos_z:
        :param param_value_11:
        :param param_value_22:
        :param param_value_33:
        :return:
        """
        checkdatatypes.check_float_variable('Parameter value for e11', param_value_11, (None, None))
        checkdatatypes.check_float_variable('Parameter value for e22', param_value_22, (None, None))
        if param_value_33 is not None:
            checkdatatypes.check_float_variable('Parameter value for e33', param_value_33, (None, None))

        self.append_row([grid_pos_x, grid_pos_y, grid_pos_z, param_value_11, param_value_22, param_value_33])

        return

    def setup(self):
        """
        Init setup
        :return:
        """
        self.init_setup(self.TableSetupList)

        return

    def set_user_grid_parameter_values(self, user_grid_value_dict):
        """ set the parameter values on user defined grid
        Note: each grid's value is given by a dict with keys (2) value (3) dir (4) scan-index
        :param user_grid_value_dict: key = position, value is described as note
        :return:
        """
        checkdatatypes.check_dict('Parameter values on user defined grid', user_grid_value_dict)

        grid_positions = user_grid_value_dict.keys()
        grid_positions.sort()

        for i_grid, grid_pos in enumerate(grid_positions):
            grid_i = user_grid_value_dict[grid_pos]
            self.add_matched_grid(grid_pos[0], grid_pos[1], grid_pos[2], grid_i['e11'], grid_i['e22'], grid_i['e33'])
        # END-FOR

        return

# END-CLASS-DEF


class ParamValueMapAnalysisTable(NTableWidget.NTableWidget):
    """
    Table for parameter mapping on grids
    """
    TableSetupList = [('Scan Index', 'int'),   # main direction scan log index]
                      ('x', 'float'),
                      ('y', 'float'),
                      ('z', 'float'),
                      ('Parameter', 'float'),  # parameter value
                      ('Direction', 'str')     # e11, e22 or e33
                      ]

    def __init__(self, parent):
        """ initialization
        :param parent:
        """
        super(ParamValueMapAnalysisTable, self).__init__(parent)

        return

    def set_user_grid_parameter_values(self, grid_vec, mapped_param_value_array, direction):
        """ set the parameter values on user defined grid
        Note: each grid's value is given by a dict with keys (2) value (3) dir (4) scan-index
        :param grid_vec:
        :param mapped_param_value_array: key = position, value is described as note
        :param direction:
        :return:
        """
        checkdatatypes.check_numpy_arrays('Grid position vector', [grid_vec], 2, False)
        checkdatatypes.check_numpy_arrays('Parameter value mapped onto grid', mapped_param_value_array, 1, False)
        assert grid_vec.shape[0] == mapped_param_value_array.shape[0], 'Number of grids shall be same'

        for i_grid in range(grid_vec.shape[0]):
            self.append_row([None, grid_vec[i_grid][0], grid_vec[i_grid][1], grid_vec[i_grid][2],
                             mapped_param_value_array[i_grid], direction])

        return

    def set_raw_grid_parameter_values(self, raw_grid_value_dict):
        """
        set the parameter values on raw defined grid
        Note: each grid's value is given by a dict with keys (2) value (3) dir (4) scan-index
        :param raw_grid_value_dict: key = position, value is described as above note
        :return:
        """
        checkdatatypes.check_dict('Parameter values on raw experimental grid', raw_grid_value_dict)

        grid_positions = raw_grid_value_dict.keys()
        grid_positions.sort()

        for i_grid, grid_pos in enumerate(grid_positions):
            grid_i = raw_grid_value_dict[grid_pos]
            self.append_row([grid_i['scan-index'], grid_pos[0], grid_pos[1], grid_pos[2],
                             grid_i['value'], grid_i['dir']])
        # END-FOR

    def reset_table(self):
        """
        reset table
        :return:
        """
        self.remove_all_rows()

        return

    def setup(self):
        """
        Init setup
        :return:
        """
        self.init_setup(self.TableSetupList)

# END-DEF-CLASS


class PartialMatchedGrids(NTableWidget.NTableWidget):
    """ Table for partially matched grids
    """
    TableSetupList = [('Direction', 'str'),
                      ('Scan Index', 'int'),  # main direction scan log index]
                      ('x', 'float'),
                      ('y', 'float'),
                      ('z', 'float'),
                      ('Direction', 'str'),  # other direction
                      ('Scan Index', 'int')]

    def __init__(self, parent):
        """ initialization
        :param parent:
        """
        super(PartialMatchedGrids, self).__init__(parent)

        return

    def setup(self):
        """
        Init setup
        :return:
        """
        self.init_setup(self.TableSetupList)

        return


class PoleFigureTable(NTableWidget.NTableWidget):
    """
    A table tailored to pole figure
    """
    TableSetupList = [('alpha', 'float'),
                      ('beta', 'float'),
                      ('intensity', 'float'),
                      ('detector', 'int'),
                      ('log #', 'int'),
                      ('2theta', 'float'),
                      ('omega', 'float'),
                      ('phi', 'float'),
                      ('chi', 'float'),
                      ('cost', 'float')]

    def __init__(self, parent):
        """
        initialization
        parent
        """
        super(PoleFigureTable, self).__init__(parent)

        # declare class instance
        self._col_index_alpha = None
        self._col_index_beta = None
        self._col_index_intensity = None

        self._col_index_scan_index = None
        self._col_det_id = None

        self._col_index_2theta = None
        self._col_index_omega = None
        self._col_index_phi = None
        self._col_index_chi = None

        self._col_index_goodness = None

        return

    def get_detector_log_index(self, row_number):
        """
        get detector ID and scan log index of a row
        :param row_number:
        :return:
        """
        # check
        checkdatatypes.check_int_variable('Row number', row_number, (0, self.rowCount()))

        # get values
        det_id = self.get_cell_value(row_number, self._col_det_id)
        log_number = self.get_cell_value(row_number, self._col_index_scan_index)

        return det_id, log_number

    def init_exp(self, scan_log_indexes_dict):
        """
        init the table for an experiment with a given list of scan indexes
        :param scan_log_indexes_dict:
        :return:
        """
        # TODO - Shall create a new module named as pyrs.utilities for utility methods used by both core and interface
        assert isinstance(scan_log_indexes_dict, dict), 'blabla'

        for det_id in sorted(scan_log_indexes_dict.keys()):
            for scan_index in sorted(scan_log_indexes_dict[det_id]):
                self.append_row([None, None, 0., det_id, scan_index, None, None, None, None, None])

        return

    def set_intensity(self, row_number, intensity, chi2):
        # TODO - DOC & CHECK
        self.update_cell_value(row_number, self._col_index_intensity, intensity)
        self.update_cell_value(row_number, self._col_index_goodness, chi2)

        return

    def set_pole_figure_motors_position(self, row_number, motor_pos_dict):
        # TODO - DOC & CHECK
        self.update_cell_value(row_number, self._col_index_2theta, motor_pos_dict['2theta'])
        self.update_cell_value(row_number, self._col_index_phi, motor_pos_dict['phi'])
        self.update_cell_value(row_number, self._col_index_omega, motor_pos_dict['omega'])
        self.update_cell_value(row_number, self._col_index_chi, motor_pos_dict['chi'])

    def set_pole_figure_projection(self, row_number, alpha, beta):
        self.update_cell_value(row_number, self._col_index_alpha, alpha)
        self.update_cell_value(row_number, self._col_index_beta, beta)

    def setup(self):
        """
        Init setup
        :return:
        """
        self.init_setup(self.TableSetupList)

        # Set up column width
        self.setColumnWidth(0, 80)
        self.setColumnWidth(1, 80)
        self.setColumnWidth(2, 80)
        self.setColumnWidth(3, 60)  # integer can be narrower
        self.setColumnWidth(4, 60)  # integer can be narrower
        self.setColumnWidth(5, 80)
        self.setColumnWidth(6, 80)
        self.setColumnWidth(7, 80)
        self.setColumnWidth(8, 80)

        # Set up the column index for start, stop and select
        self._col_index_alpha = self.TableSetupList.index(('alpha', 'float'))
        self._col_index_beta = self.TableSetupList.index(('beta', 'float'))
        self._col_index_intensity = self.TableSetupList.index(('intensity', 'float'))

        self._col_index_scan_index = self.TableSetupList.index(('log #', 'int'))
        self._col_det_id = self.TableSetupList.index(('detector', 'int'))

        self._col_index_2theta = self.TableSetupList.index(('2theta', 'float'))
        self._col_index_omega = self.TableSetupList.index(('omega', 'float'))
        self._col_index_phi = self.TableSetupList.index(('phi', 'float'))
        self._col_index_chi = self.TableSetupList.index(('chi', 'float'))

        self._col_index_goodness = self.TableSetupList.index(('cost', 'float'))

        return
# END-DEF-CLASS()


class StrainStressValueTable(NTableWidget.NTableWidget):
    """
    A table for strain and stress value
    """
    TableSetupList = [('x', 'float'),
                      ('y', 'float'),
                      ('z', 'float'),
                      ('e11', 'float'),  # e is short for epsilon as strain
                      ('e22', 'float'),
                      ('e33', 'float'),
                      ('s11', 'float'),  # s is short for sigma as stress
                      ('s22', 'float'),
                      ('s33', 'float')]

    def __init__(self, parent):
        """
        initialization
        :param parent:
        """
        super(StrainStressValueTable, self).__init__(parent)

        self._col_index_strain_dict = dict()
        self._col_index_stress_dict = dict()

        return

    def add_grid_strain_stress(self, grid_pos, strain_matrix, stress_matrix):
        """
        add a grid with strain and
        :param grid_pos:
        :param strain_matrix:
        :param stress_matrix:
        :return:
        """
        # check inputs
        checkdatatypes.check_numpy_arrays('Grid position', [grid_pos], dimension=1, check_same_shape=False)
        checkdatatypes.check_numpy_arrays('Strain and stress matrix', [strain_matrix, stress_matrix],
                                          dimension=2, check_same_shape=True)

        line_list = list()
        line_list.extend([pos for pos in grid_pos])
        line_list.extend([strain_matrix[i, i] for i in range(3)])
        line_list.extend([stress_matrix[i, i] for i in range(3)])

        self.append_row(line_list)

    def setup(self):
        """
        Init setup
        :return:
        """
        self.init_setup(self.TableSetupList)

        for index, element_name in enumerate(['epsilon[11]', 'epsilon[22]', 'epsilon[33]']):
            self._col_index_strain_dict['e11'] = 3 + index

        for index, element_name in enumerate(['nu11', 'nu22', 'nu33']):
            self._col_index_strain_dict['s11'] = 6 + index

        return
