# pylint: disable=C0103,R0904
# N(DAV)TableWidget
import csv

from qtpy import QtCore  # type:ignore
from qtpy.QtWidgets import QTableWidget, QTableWidgetItem, QCheckBox  # type:ignore

try:
    _fromUtf8 = QtCore.QString.fromUtf8  # type:ignore
except AttributeError:
    def _fromUtf8(s):
        return s


class NTableWidget(QTableWidget):
    """
    NdavTableWidget inherits from QTableWidget by extending the features
    for easy application.
    """

    def __init__(self, parent):
        """

        :param parent:
        :return:
        """
        QTableWidget.__init__(self, parent)

        self._myParent = parent

        self._myColumnNameList = None
        self._myColumnTypeList = None

    def append_row(self, row_value_list, type_list=None, num_decimal=7):
        """
        append a row to the table
        :param row_value_list:
        :param type_list:
        :param num_decimal: number of decimal points for floating
        :return: 2-tuple as (boolean, message)
        """
        # Check input
        assert isinstance(row_value_list, list), 'Row values {0} must be given by a list but ' \
                                                 'not a {1}'.format(row_value_list, type(row_value_list))
        if type_list is not None:
            assert isinstance(type_list, list), 'Value types {0} must be given by a list but ' \
                                                'not a {1}'.format(type_list, type(type_list))
            if len(row_value_list) != len(type_list):
                raise RuntimeError('If value types are given, then they must have the same '
                                   'numbers ({0}) and values ({1})'.format(len(row_value_list),
                                                                           len(type_list)))
        else:
            type_list = self._myColumnTypeList

        if len(row_value_list) != self.columnCount():
            ret_msg = 'Input number of values (%d) is different from ' \
                      'column number (%d).' % (len(row_value_list), self.columnCount())
            return False, ret_msg
        else:
            ret_msg = ''

        # Insert new row
        row_number = self.rowCount()
        self.insertRow(row_number)

        # Set values
        for i_col in range(min(len(row_value_list), self.columnCount())):
            if type_list[i_col] == 'checkbox':
                # special case: check box
                self.set_check_box(row_number, i_col, row_value_list[i_col])
            else:
                # regular items
                item_value = row_value_list[i_col]
                if isinstance(item_value, float):
                    value_str = ('{0:.%df}' % num_decimal).format(item_value)
                else:
                    value_str = str(item_value)

                item = QTableWidgetItem()
                item.setText(_fromUtf8(value_str))
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                # Set editable flag! item.setFlags(item.flags() | ~QtCore.Qt.ItemIsEditable)
                self.setItem(row_number, i_col, item)
        # END-FOR(i_col)

        return True, ret_msg

    def export_table_csv(self, csv_file_name):
        """ Export table to a CSV fie
        :param csv_file_name: csv file name
        :return:
        """
        # check input
        assert isinstance(csv_file_name, str), 'CSV file name {0} to export table must be a string but not a {1}' \
                                               ''.format(csv_file_name, type(csv_file_name))

        # get title as header
        col_names = self._myColumnNameList[:]
        num_columns = self.columnCount()
        num_rows = self.rowCount()
        content_line_list = list()

        for i_row in range(num_rows):
            line_items = list()
            for j_col in range(num_columns):
                item_value = self.get_cell_value(i_row, j_col)
                if isinstance(item_value, str):
                    # remove tab because tab will be used as delimiter
                    item_value = item_value.replace('\t', '')
                elif item_value is None:
                    item_value = ''
                line_items.append(item_value)
            # END-FOR
            content_line_list.append(line_items)
        # END-FOR (row)

        with open(csv_file_name, 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
            # write header
            csv_writer.writerow(col_names)
            # write content
            for line_items in content_line_list:
                csv_writer.writerow(line_items)
            # END-FOR
        # END-WITH

    def get_cell_value(self, row_index, col_index, allow_blank=False):
        """
        Purpose: Get cell value
        Requirements: row index and column index are integer and within range.
        Guarantees: the cell value with correct type is returned
        :param row_index:
        :param col_index:
        :param allow_blank:
        :return: int/float/string or None if allow_blank
        """
        # check
        assert isinstance(row_index, int), 'Row index {0} must be an integer'.format(row_index)
        assert isinstance(col_index, int), 'Column index {0} must be an integer'.format(col_index)
        if not 0 <= row_index < self.rowCount():
            raise RuntimeError('Row index {0} is out of range [0, {1})'
                               ''.format(row_index, self.rowCount()))
        if not 0 <= col_index < self.columnCount():
            raise RuntimeError('Column index {0} is out of range [0, {1})'
                               ''.format(col_index, self.columnCount()))

        # get cell type
        cell_data_type = self._myColumnTypeList[col_index]

        if cell_data_type == 'checkbox':
            # Check box
            cell_i_j = self.cellWidget(row_index, col_index)
            assert isinstance(cell_i_j, QCheckBox), 'Cell {0} {1} must be of type QCheckBox but not a {2}' \
                                                    ''.format(row_index, col_index, type(cell_i_j))

            return_value = cell_i_j.isChecked()
        else:
            # Regular cell for int, float or string
            item_i_j = self.item(row_index, col_index)
            assert isinstance(item_i_j, QTableWidgetItem), 'Cell {0} {1} must be of type QTableWidgetItem but not a ' \
                                                           '{2}'.format(row_index, col_index, type(item_i_j))
            # get the string of the cell
            return_value = str(item_i_j.text()).strip()

            # check empty input
            if cell_data_type == 'int':
                if len(return_value) == 0 and allow_blank:
                    return_value = None
                else:
                    return_value = int(return_value)
            elif cell_data_type == 'float' or cell_data_type == 'double':
                if len(return_value) == 0 and allow_blank:
                    return_value = None
                else:
                    return_value = float(return_value)
            # END-IF-ELSE (cell_data_type)
        # END-IF-ELSE (cell_type)

        return return_value

    def init_setup(self, column_tup_list):
        """ Initial setup
        :param column_tup_list: list of 2-tuple as string (column name) and string (data type)
        :return:
        """
        # Check requirements
        assert isinstance(column_tup_list, list)
        assert len(column_tup_list) > 0

        # Define column headings
        num_cols = len(column_tup_list)

        # Class variables
        self._myColumnNameList = list()
        self._myColumnTypeList = list()

        for c_tup in column_tup_list:
            c_name = c_tup[0]
            c_type = c_tup[1]
            self._myColumnNameList.append(c_name)
            self._myColumnTypeList.append(c_type)

        self.setColumnCount(num_cols)
        self.setHorizontalHeaderLabels(self._myColumnNameList)

    def remove_all_rows(self):
        """
        Remove all rows
        :return:
        """
        num_rows = self.rowCount()
        for i_row in range(1, num_rows + 1):
            self.removeRow(num_rows - i_row)

    def set_check_box(self, row, col, state):
        """ function to add a new select checkbox to a cell in a table row
        won't add a new checkbox if one already exists
        """
        # Check input
        assert isinstance(state, bool)

        # Check if cellWidget exists
        if self.cellWidget(row, col):
            # existing: just set the value
            self.cellWidget(row, col).setChecked(state)
        else:
            # case to add checkbox
            checkbox = QCheckBox()
            checkbox.setText('')
            checkbox.setChecked(state)

            # Adding a widget which will be inserted into the table cell
            # then centering the checkbox within this widget which in turn,
            # centers it within the table column :-)
            self.setCellWidget(row, col, checkbox)
        # END-IF-ELSE

    def set_value_cell(self, row, col, value=''):
        """
        Set value to a cell with integer, float or string
        :param row:
        :param col:
        :param value:
        :return:
        """
        # Check
        assert not isinstance(value, bool), 'Boolean is not accepted for set_value_cell()'

        if row < 0 or row >= self.rowCount() or col < 0 or col >= self.columnCount():
            raise IndexError('Input row number or column number is out of range.')

        # Init cell
        cell_item = QTableWidgetItem()
        cell_item.setText(_fromUtf8(str(value)))
        cell_item.setFlags(cell_item.flags() & ~QtCore.Qt.ItemIsEditable)

        self.setItem(row, col, cell_item)

    def update_cell_value(self, row, col, value, number_decimal=7):
        """
        Update (NOT reset) the value of a cell
        :param row:
        :param col:
        :param value:
        :param number_decimal: significant digit for float
        :return: None
        """
        # Check
        assert isinstance(row, int) and 0 <= row < self.rowCount(), \
            'Row %s (%s) must be an integer between 0 and %d.' % (str(row), type(row), self.rowCount())
        assert isinstance(col, int) and 0 <= col < self.columnCount()
        assert isinstance(number_decimal, int) and number_decimal > 0

        cell_item = self.item(row, col)
        cell_widget = self.cellWidget(row, col)

        if cell_widget is None:
            # TableWidgetItem
            if cell_item is None:
                self.set_value_cell(row, col, value)
            elif isinstance(cell_item, QTableWidgetItem):
                if isinstance(value, float):
                    # apply significant digit dynamically
                    # use 'g' for significant float_str = ('{0:.%dg}' % significant_digits).format(value)
                    float_str = ('{0:.%df}' % number_decimal).format(value)  # decimal
                    cell_item.setText(_fromUtf8(float_str))
                    # cell_item.setText(_fromUtf8('%.7f' % value))
                    # ('{0:.%dg}'%(2)).format(d)
                else:
                    cell_item.setText(_fromUtf8(str(value)))
            else:
                raise RuntimeError('Cell widget type error!')
        elif cell_item is None and cell_widget is not None:
            # TableCellWidget
            if isinstance(cell_widget, QCheckBox) is True:
                cell_widget.setChecked(value)
            else:
                raise TypeError('Cell of type %s is not supported.' % str(type(cell_item)))
        else:
            raise TypeError('Table cell (%d, %d) is in an unsupported situation!' % (row, col))
