# Module containing extended TableWidgets for PyRS project
import NTableWidget


class FitResultTable(NTableWidget.NTableWidget):
    """

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
        """

        """
        super(FitResultTable, self).__init__(parent)

        return

    def init_exp(self, index_list):
        # TODO
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
        self.update_cell_value(row_number, self._colIndexCoM, com)

    def set_peak_params(self, row_number, center, height, fwhm, intensity, chi2, profile):
        # TODO
        self.update_cell_value(row_number, self._colIndexCenter, center)
        self.update_cell_value(row_number, self._colIndexHeight, height)
        self.update_cell_value(row_number, self._colIndexWidth, fwhm)
        self.update_cell_value(row_number, self._colIndexChi2, chi2)
        self.update_cell_value(row_number, self._colIndexIntensity, intensity)
        self.update_cell_value(row_number, self._colIndexProfile, profile)

        return
