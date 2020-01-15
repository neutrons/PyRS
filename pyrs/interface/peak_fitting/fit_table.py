import numpy as np
from qtpy.QtWidgets import QTableWidgetItem

class FitTable:

    COL_INDEX_TO_ESCAPE = [1]

    def __init__(self, parent=None, fit_result=None):
        self.parent = parent
        self.fit_result = fit_result

    def initialize_fit_result_widgets(self):
        self._initialize_list_of_peaks()
        self._initialize_table()

    def populate_fit_result_table(self):
        _peak_selected = self.parent.ui.spinBox_peak_index.value()
        _peak_collection = self.fit_result.peakcollections[_peak_selected-1]  # peak 1 is at 0 index

        _value = self._get_value_to_display(_peak_collection)

        for _row, _row_value in enumerate(_value):
            self.parent.ui.tableView_fitSummary.insertRow(_row)
            _global_col_index = 0

            print(_row_value)

            for _local_col_index, _col_value in enumerate(_row_value):
                if _local_col_index in self.COL_INDEX_TO_ESCAPE:
                    continue
                _item = QTableWidgetItem(str(_col_value))
                self.parent.ui.tableView_fitSummary.setItem(_row, _global_col_index, _item)
                _global_col_index += 1

    def _get_value_to_display(self, peak_collection):
        if self.parent.ui.radioButton_fit_value.isChecked():
            return peak_collection.parameters_values
        else:
            return peak_collection.parameters_errors

    def fit_value_error_changed(self):
        self._clear_rows()
        self.populate_fit_result_table()

    def _initialize_list_of_peaks(self):
        nbr_peaks = len(self.fit_result.peakcollections)
        self.parent.ui.spinBox_peak_index.setRange(1, nbr_peaks)

    def _initialize_table(self):
        self._clear_table()
        columns_names = self._get_list_of_columns()
        for _column in np.arange(len(columns_names)):
            self.parent.ui.tableView_fitSummary.insertColumn(_column)
        self.parent.ui.tableView_fitSummary.setHorizontalHeaderLabels(columns_names)

    def _clear_rows(self):
        _nbr_row = self.parent.ui.tableView_fitSummary.rowCount()
        for _ in np.arange(_nbr_row):
            self.parent.ui.tableView_fitSummary.removeRow(0)

    def _clear_table(self):
        _nbr_column = self.parent.ui.tableView_fitSummary.columnCount()
        for _ in np.arange(_nbr_column):
            self.parent.ui.tableView_fitSummary.removeColumn(0)

    def _get_list_of_columns(self):
        # we want to go from
        # ('wsindex', 'peakindex', 'Mixing', 'Intensity', 'PeakCentre', 'FWHM', 'A0', 'A1', 'chi2')
        # to
        # ('Sub-run #', 'Mixing', ....)

        _peak_collection = self.fit_result.peakcollections[0]
        column_names = _peak_collection.parameters_values.dtype.names
        # remove second column
        clean_column_names = []
        for _col_index, _col_value in enumerate(column_names):
            if _col_index in self.COL_INDEX_TO_ESCAPE:
                continue
            if _col_index == 0:
                _col_value = 'Sub-run #'
            clean_column_names.append(_col_value)
        return clean_column_names


