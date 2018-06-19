# Module containing extended TableWidgets for PyRS project
import NTableWidget


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
                      ('chi', 'float')]

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

        return

    def get_detector_log_index(self, row_number):
        # TODO DOC/CHECK

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
                self.append_row([None, None, 0., det_id, scan_index, None, None, None, None])

        return

    def set_intensity(self, row_number, intensity):
        # TODO - DOC & CHECK
        self.update_cell_value(row_number, self._col_index_intensity, intensity)

        return

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

        return
