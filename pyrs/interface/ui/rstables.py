# Module containing extended TableWidgets for PyRS project
import NTableWidget
from pyrs.utilities import checkdatatypes


class FitResultTable(NTableWidget.NTableWidget):
    """
    A table tailored to peak fit result
    """
    TableSetupList = [('Index', 'int'),
                      ('Center', 'float'),
                      ('Height', 'float'),
                      ('FWHM', 'float'),
                      ('Intensity', 'float'),
                      ('Chi^2', 'float'),
                      ('C.O.M', 'float'),  # center of mass
                      ('Profile', 'string')]

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

    def setup(self):
        """
        Init setup
        :return:
        """
        self.init_setup(self.TableSetupList)

        # Set up column width
        self.setColumnWidth(0, 60)
        self.setColumnWidth(1, 80)
        self.setColumnWidth(2, 80)
        self.setColumnWidth(3, 80)

        # Set up the column index for start, stop and select
        self._colIndexIndex = self.TableSetupList.index(('Index', 'int'))
        self._colIndexCenter = self.TableSetupList.index(('Center', 'float'))
        self._colIndexHeight = self.TableSetupList.index(('Height', 'float'))
        self._colIndexWidth = self.TableSetupList.index(('FWHM', 'float'))
        self._colIndexIntensity = self.TableSetupList.index(('Intensity', 'float'))
        self._colIndexChi2 = self.TableSetupList.index(('Chi^2', 'float'))
        self._colIndexCoM = self.TableSetupList.index(('C.O.M', 'float'))
        self._colIndexProfile = self.TableSetupList.index(('Profile', 'string'))

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


# TODO - 20180814 - Clean up and fill
class GridsStatisticsTable(NTableWidget.NTableWidget):
    """

    """
    TableSetupList = [('Item', 'str'),   # include min x, max x, num x, avg resolution x, ... (for y) .. (for z)... # data points
                      ('e11', 'float'),
                      ('e22', 'float'),
                      ('e33', 'float')]

    def __init__(self, parent):
        return


# TODO - 20180814 - Clean up and fill
class GridAlignmentTable(NTableWidget.NTableWidget):
    """

    """
    TableSetupList = [('x', 'float'),
                      ('y', 'float'),
                      ('z', 'float'),
                      ('e11', 'int'),  # scan log index of e11 direction data, -1 for not found
                      ('e22', 'int'),
                      ('e33', 'int')]

    def __init__(self, parent):
        """

        :param parent:
        """
        return

# TODO - 20180814 - Clean up and fill
class MatchedGridsTable(NTableWidget.NTableWidget):
    """

    """
    TableSetupList = [('x', 'float'),
                      ('y', 'float'),
                      ('z', 'float'),
                      ('e11', 'int'),  # scan log index of e11 direction data
                      ('e22', 'int'),
                      ('e33', 'int')]
    def __init__(self, parent):
        return

# TODO - 20180814 - Clean up and fill
class MismatchedGridsTable(NTableWidget.NTableWidget):
    """

    """
    TableSetupList = [('Direction', 'str'),
                      ('Scan Index', 'int'),
                      ('x', 'float'),
                      ('y', 'float'),
                      ('z', 'float')]
    def __init__(self, parent):
        return



# TODO - 20180814 - Clean up and fill
class PartialMatchedGrids(NTableWidget.NTableWidget):
    """

    """
    TableSetupList = [('Direction', 'str'),
                      ('Scan Index', 'int'),  # main direction scan log index]
                      ('x', 'float'),
                      ('y', 'float'),
                      ('z', 'float'),
                      ('Direction', 'str'),  # other direction
                      ('Scan Index', 'int')]
    def __init__(self, parent):
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
