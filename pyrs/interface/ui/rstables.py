# Module containing extended TableWidgets for PyRS project
import NTableWidget


class FitResultTable(NTableWidget.NTableWidget):
    """

    """
    TableSetupList = [('Index', 'int'),
                      ('Center', 'float'),
                      ('Height', 'float'),
                      ('FWHM', 'float'),
                      ('Chi^2', 'float')]

    def __init__(self, parent):
        """

        """
        super(FitResultTable, self).__init__(parent)

        return

    def init_exp(self, index_list):
        # TODO
        for index in index_list:
            self.append_row([index, None, None, None, None])

    def setup(self):
        """
        Init setup
        :return:
        """
        self.init_setup(self.TableSetupList)

        # Set up column width
        self.setColumnWidth(0, 200)
        self.setColumnWidth(1, 200)
        self.setColumnWidth(2, 100)
        self.setColumnWidth(3, 100)

        # Set up the column index for start, stop and select
        self._colIndexIndex = self.TableSetupList.index(('Index', 'int'))
        self._colIndexCenter = self.TableSetupList.index(('Center', 'float'))
        self._colIndexHeight = self.TableSetupList.index(('Height', 'float'))
        self._colIndexWidth = self.TableSetupList.index(('FWHM', 'float'))
        self._colIndexChi2 = self.TableSetupList.index(('Chi^2', 'float'))

        return

    def set_peak_params(self, row_number, center, height, fwhm, chi2):
        # TODO
        self.update_cell_value(row_number, self._colIndexCenter, center)
        self.update_cell_value(row_number, self._colIndexHeight, height)
        self.update_cell_value(row_number, self._colIndexWidth, fwhm)
        self.update_cell_value(row_number, self._colIndexChi2, chi2)

        return
