import numpy as np

from pyrs.core.peak_profile_utility import PeakShape


class FitTable:

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

        import pprint
        pprint.pprint("populate_fit_result_table")
        pprint.pprint(_value.dtype)

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
        _peak_collection = self.fit_result.peakcollections[0]
        column_names = _peak_collection.parameters_values.dtype.names
        return column_names


