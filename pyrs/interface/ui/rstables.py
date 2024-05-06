# Module containing extended TableWidgets for PyRS project
from . import NTableWidget
import numpy
from qtpy import QtWidgets
from typing import List, Tuple


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
    TableSetupList: List[Tuple[str, str]] = list()

    def __init__(self, parent):
        """ Initialization
        """
        super(FitResultTable, self).__init__(parent)

        self._column_names = None

        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)

    def init_exp(self, sub_run_number_list):
        """
        init the table for an experiment with a given list of scan indexes
        :param sub_run_number_list: list of sub runs
        :return:
        """
        # checkdatatypes.check_list('Index list', list(sub_run_number_list))
        #
        # # sort
        # sub_run_number_list.sort()

        if isinstance(sub_run_number_list, list) or isinstance(sub_run_number_list, numpy.ndarray):
            # sort
            sub_run_number_list.sort()
        else:
            raise RuntimeError('Sub runs in {} are not supported'.format(type(sub_run_number_list)))

        # clean the table
        if self.rowCount() > 0:
            self.remove_all_rows()

        # append new rows
        for index in sub_run_number_list:
            self.append_row([index, None, None, None, None, None, None, '', 'checkbox'])

        return

    def reset_table(self, peak_param_names):
        """ Reset table. Parameters other than peak fitting will be handled by setup()
        :param peak_param_names: List of peak parameters names
        :return:
        """
        # Completely clear the previously written table
        self.clear()

        # Add the new column names
        if peak_param_names is None:
            self.init_setup(self.TableSetupList)
        else:
            self.setup(peak_param_names)

        return

    def setup(self, peak_param_names):
        """
        Init setup
        :return:
        """
        # create table columns dynamically
        self.TableSetupList = list()

        # self.TableSetupList.append(('sub-run', 'int'))
        for param_name in peak_param_names:
            self.TableSetupList.append((param_name, 'float'))
        # self.TableSetupList.append(('C.O.M', 'float'))
        self.TableSetupList.append(('Profile', 'string'))

        self._column_names = [item[0] for item in self.TableSetupList]

        # reset table
        self.init_setup(self.TableSetupList)
        print('[DB...BAT] Init setup table columns: {}'.format(self.TableSetupList))
        print('[DB...BAT] Init setup table columns: {}'.format(self._column_names))

        # # Set up column width
        self.setColumnWidth(0, 60)
        for col_index in range(1, len(self._column_names) - 1):
            self.setColumnWidth(col_index, 80)
        self.setColumnWidth(len(self._column_names) - 1, 120)

        # Set up the column index for start, stop and select
        # self._colIndexIndex = self.TableSetupList.index(('sub-runs', 'int'))
        # self._colIndexCoM = self.TableSetupList.index(('C.O.M', 'float'))
        # self._colIndexProfile = self.TableSetupList.index(('Profile', 'string'))

        return

    def set_fit_summary(self, row_number, ordered_param_list, param_dict, write_error=False,
                        peak_profile='not set'):
        """

        Parameters
        ----------
        row_number: int
            row number
        ordered_param_list:
        param_dict
        write_error
        peak_profile

        Returns
        -------

        """
        """
        Set the fitting result, i.e., peak parameters' value to a row
        :param row_number:
        :param ordered_param_list: parameters names list with the same order as table columns
        :param param_dict: dictionary containing peak parameter values
        :param write_error: Flag to write out error or value
        """
        # Init list to append
        this_value_list = list()

        # Set values
        for param_name in ordered_param_list:
            # Get numpy array of this parameter
            param_value_vec = param_dict[param_name]
            assert isinstance(param_value_vec, numpy.ndarray), 'Parameter value must be given by array'
            # Get value
            value_i = param_dict[param_name][row_number]
            # value_i can be float or numpy array
            if isinstance(value_i, numpy.ndarray):
                if write_error and value_i.shape[0] > 1:
                    # Output is the error
                    value_i = value_i[1]
                else:
                    # Output is the value
                    value_i = value_i[0]

            this_value_list.append(value_i)

        # Last: peak profile
        this_value_list.append(peak_profile)

        if row_number < self.rowCount():
            for col_num, item_value in enumerate(this_value_list):
                if item_value is not None:
                    try:
                        self.update_cell_value(row_number, col_num, item_value)
                    except TypeError:
                        print('Cell @ {}, {} of value {} cannot be updated'.format(row_number, col_num, item_value))
        else:
            status, err_msg = self.append_row(row_value_list=this_value_list)
            if not status:
                print('[ERROR] {}'.format(err_msg))
