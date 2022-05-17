import numpy as np
from qtpy.QtWidgets import QTableWidgetItem, QTableWidgetSelectionRange # type:ignore
from qtpy.QtGui import QColor # type:ignore

MICROSTRAIN = u"\u00b5strain"
SUCCESS = "success"
COLOR_FAILED_FITTING = QColor(247, 173, 13)  # orange


class FitTable:

    COL_SIZE = 100
    STATUS_COL_SIZE = 500  # last column

    def __init__(self, parent=None, fit_result=None):
        self.parent = parent
        self.fit_result = fit_result

    def initialize_fit_result_widgets(self):
        self._initialize_list_of_peaks()
        self.initialize_table()
        self.initialize_table_column_size()

    def populate_fit_result_table(self):
        _peak_selected = self.parent.ui.spinBox_peak_index.value()
        _peak_collection = self.fit_result.peakcollections[_peak_selected-1]  # peak 1 is at 0 index

        _value = self._get_value_to_display(peak_collection=_peak_collection)
        _chisq = _peak_collection.fitting_costs
        _status = _peak_collection.get_fit_status()

        _d_spacing = self._get_d_spacing_to_display(peak_selected=_peak_selected,
                                                    peak_collection=_peak_collection)

        _microstrain_mapping = self._get_microstrain_mapping_to_display(peak_collection=_peak_collection)

        def set_item(value='', fitting_worked=True):
            _item = QTableWidgetItem(value)
            if not fitting_worked:
                _item.setBackground(COLOR_FAILED_FITTING)
            return _item

        for _row, _row_value in enumerate(_value):
            self.parent.ui.tableView_fitSummary.insertRow(_row)
            _global_col_index = 0

            _fitting_worked = True if _status[_row] == SUCCESS else False

            for _local_col_index, _col_value in enumerate(_row_value):
                _item = set_item(value=str(_col_value), fitting_worked=_fitting_worked)
                self.parent.ui.tableView_fitSummary.setItem(_row, _global_col_index, _item)
                _global_col_index += 1

            # add chisq values (but forget when error is selected
            if self.parent.ui.radioButton_fit_value.isChecked():
                _item = set_item(value=str(_chisq[_row]), fitting_worked=_fitting_worked)
                self.parent.ui.tableView_fitSummary.setItem(_row, _global_col_index, _item)
                _global_col_index += 1

            # add d-spacing
            _item = set_item(value=str(_d_spacing[_row]), fitting_worked=_fitting_worked)
            self.parent.ui.tableView_fitSummary.setItem(_row, _global_col_index, _item)
            _global_col_index += 1

            # add strain calculation
            _microstrain = _microstrain_mapping[_row]
            if np.isnan(_microstrain):
                str_strain_value = "nan"
            else:
                str_strain_value = str(np.int(_microstrain))
            _item = set_item(value=str_strain_value, fitting_worked=_fitting_worked)
            self.parent.ui.tableView_fitSummary.setItem(_row, _global_col_index, _item)
            _global_col_index += 1

            # add status message
            _item = set_item(value=_status[_row], fitting_worked=_fitting_worked)
            self.parent.ui.tableView_fitSummary.setItem(_row, _global_col_index, _item)
            _global_col_index += 1

    def _get_d_spacing_to_display(self, peak_selected=1, peak_collection=None):
        _d_reference = np.float(str(self.parent.ui.peak_range_table.item(peak_selected-1, 3).text()))
        peak_collection.set_d_reference(values=_d_reference)
        values, error = peak_collection.get_dspacing_center()
        if self.parent.ui.radioButton_fit_value.isChecked():
            return values
        else:
            return error

    def _get_microstrain_mapping_to_display(self, peak_collection=None):
        values, error = peak_collection.get_strain(units='microstrain')
        if self.parent.ui.radioButton_fit_value.isChecked():
            return values
        else:
            return error

    def _get_value_to_display(self, peak_collection):
        values, error = peak_collection.get_effective_params()
        if self.parent.ui.radioButton_fit_value.isChecked():
            return values
        else:
            return error

    def fit_value_error_changed(self):
        self._clear_rows()
        self.populate_fit_result_table()

    def _initialize_list_of_peaks(self):
        nbr_peaks = len(self.fit_result.peakcollections)
        self.parent.ui.spinBox_peak_index.setRange(1, nbr_peaks)

    def initialize_table(self):
        self._clear_table()
        columns_names = self._get_list_of_columns()
        for _column in np.arange(len(columns_names)):
            self.parent.ui.tableView_fitSummary.insertColumn(_column)
        self.parent.ui.tableView_fitSummary.setHorizontalHeaderLabels(columns_names)

    def initialize_table_column_size(self):
        nbr_column = self.parent.ui.tableView_fitSummary.columnCount()
        for _col in np.arange(nbr_column):
            if _col < (nbr_column - 1):
                _col_size = self.COL_SIZE
            else:
                _col_size = self.STATUS_COL_SIZE
        self.parent.ui.tableView_fitSummary.setColumnWidth(_col, _col_size)

    def _clear_rows(self):
        _nbr_row = self.parent.ui.tableView_fitSummary.rowCount()
        for _ in np.arange(_nbr_row):
            self.parent.ui.tableView_fitSummary.removeRow(0)

    def _clear_columns(self):
        _nbr_column = self.get_number_of_columns()
        for _ in np.arange(_nbr_column):
            self.parent.ui.tableView_fitSummary.removeColumn(0)

    def get_number_of_columns(self):
        _nbr_column = self.parent.ui.tableView_fitSummary.columnCount()
        return _nbr_column

    def _clear_table(self):
        self._clear_rows()
        self._clear_columns()

    def _get_list_of_columns(self):
        _peak_collection = self.fit_result.peakcollections[0]
        values, _ = _peak_collection.get_effective_params()
        column_names = values.dtype.names
        clean_column_names = []
        for _col_index, _col_value in enumerate(column_names):
            if _col_index == 0:
                # _col_value = 'Sub-run #'
                _col_value = 'Peak Center'
            clean_column_names.append(_col_value)

        if self.parent.ui.radioButton_fit_value.isChecked():
            # also add chisq
            clean_column_names.append('chisq')

        # add d-spacing column
        clean_column_names.append("d spacing")

        # add strain-mapping column
        clean_column_names.append("strain mapping (" + MICROSTRAIN + ")")

        # add a status column
        clean_column_names.append("Status message")
        return clean_column_names

    def select_first_row(self):
        _nbr_column = self.get_number_of_columns()
        selection_first_row = QTableWidgetSelectionRange(0, 0, 0, _nbr_column-1)
        self.parent.ui.tableView_fitSummary.setRangeSelected(selection_first_row, True)
