import os
import json
import numpy as np
from qtpy.QtWidgets import QTableWidgetItem, QMenu
from qtpy.QtGui import QCursor
from shutil import copyfile

from pyrs.interface.gui_helper import pop_message
from pyrs.interface.gui_helper import browse_file
from pyrs.interface.gui_helper import parse_integers
from pyrs.interface.peak_fitting.load import Load
from pyrs.interface.peak_fitting.plot import Plot
from pyrs.interface.peak_fitting.fit import Fit
from pyrs.interface.peak_fitting.gui_utilities import GuiUtilities
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore
from pyrs.utilities import get_input_project_file  # type: ignore


class EventHandler:

    def __init__(self, parent=None):
        self.parent = parent

    def save_as(self):
        out_file_name = browse_file(self.parent,
                                    caption='Choose a file to save fitted peaks to',
                                    default_dir=self.parent._core.working_dir,
                                    file_filter='H5 (*.h5);;HDF (*.hdf5)',
                                    save_file=True)

        if not (out_file_name.endswith('.hdf5') or out_file_name.endswith('.h5')):
            out_file_name += '.h5'

        self.save_fit_result(out_file_name)

    def save(self):
        self.save_fit_result(self.parent.current_hidra_file_name)

    def save_fit_result(self, out_file_name=''):
        """Save the fit result, including a copy of the rest of the file if it does not exist at the specified path.

        If out_file_name is empty or if it matches the parent's current file, this updates the file.

        Otherwise, the parent's file is copied to out_file_name and
        then the updated peak fit data is written to the copy.

        :param out_file_name: string absolute fill path for the place to save the file

        """

        fit_result = self.parent.fit_result
        if fit_result is None:
            return

        if out_file_name is not None and self.parent._curr_file_name != out_file_name:
            copyfile(self.parent._curr_file_name, out_file_name)
            current_project_file = out_file_name
        else:
            current_project_file = self.parent._curr_file_name

        project_h5_file = HidraProjectFile(current_project_file, mode=HidraProjectFileMode.READWRITE)
        peakcollections = fit_result.peakcollections
        for peak in peakcollections:
            project_h5_file.write_peak_parameters(peak)
        project_h5_file.save(False)
        project_h5_file.close()

        # self.parent._core.save_peak_fit_result(self.parent._curr_data_key,
        #                                        self.parent._curr_file_name,
        #                                        out_file_name)

    def load_run_number_plot(self):
        try:
            project_dir = get_input_project_file(int(self.parent.ui.lineEdit_expNumber.text()),
                                                 preferredType=self.parent.ui.comboBox_reduction.currentText().lower())
        except (FileNotFoundError, RuntimeError) as run_err:
            pop_message(self, f'Failed to find run {self.parent.ui.lineEdit_expNumber.text()}',
                        str(run_err), 'error')
            return

        hidra_file_name = os.path.join(project_dir, f'HB2B_{self.parent.ui.lineEdit_expNumber.text()}.h5')
        self.parent.current_hidra_file_name = hidra_file_name
        self.load_and_plot(hidra_file_name)

    def browse_load_plot_hdf(self):
        if self.parent._core is None:
            raise RuntimeError('Not set up yet!')

        hidra_file_name = None
        if hidra_file_name is None:
            # No default Hidra file: browse the file
            file_filter = 'HDF (*.hdf);H5 (*.h5)'
            hidra_file_name = browse_file(self.parent,
                                          'HIDRA Project File',
                                          os.getcwd(),
                                          file_filter,
                                          file_list=False,
                                          save_file=False)

            if hidra_file_name is None:
                return  # user clicked cancel

        self.parent.current_hidra_file_name = hidra_file_name
        self.load_and_plot(hidra_file_name)

    def load_and_plot(self, hidra_file_name):
        try:
            o_load = Load(parent=self.parent)
            o_load.load(project_file=hidra_file_name)

        except RuntimeError as run_err:
            pop_message(self, 'Failed to load {}'.format(hidra_file_name),
                        str(run_err), 'error')
        except KeyError as key_err:
            pop_message(self, 'Failed to load {}'.format(hidra_file_name),
                        str(key_err), 'error')

        self.parent.current_root_statusbar_message = "Working with: {} " \
                                                     "\t\t\t\t Project Name: {}" \
                                                     "".format(hidra_file_name,
                                                               self.parent._project_name)
        self.parent.ui.statusbar.showMessage(self.parent.current_root_statusbar_message)

        try:
            o_plot = Plot(parent=self.parent)
            o_plot.plot_diff_data(plot_model=False)
            o_plot.reset_fitting_plot()

        except RuntimeError as run_err:
            pop_message(self, 'Failed to plot {}'.format(hidra_file_name),
                        str(run_err), 'error')

        try:
            o_fit = Fit(parent=self.parent)
            o_fit.initialize_fitting_table()

            # enabled all fitting widgets and main plot
            o_gui = GuiUtilities(parent=self.parent)
            o_gui.check_if_fitting_widgets_can_be_enabled()
            o_gui.enabled_sub_runs_interation_widgets(True)
            # o_gui.enabled_fitting_widgets(True)
            o_gui.enabled_data_fit_plot(True)
            o_gui.enabled_peak_ranges_widgets(True)
            o_gui.enabled_1dplot_widgets(True)

        except RuntimeError as run_err:
            pop_message(self, 'Failed to initialize widgets for {}'.format(hidra_file_name),
                        str(run_err), 'error')

    def list_subruns_2dplot(self):
        raw_input = str(self.parent.ui.lineEdit_subruns_2dplot.text())
        o_gui = GuiUtilities(parent=self.parent)

        try:
            parse_input = parse_integers(raw_input)
            o_gui.make_visible_listsubruns_warning(False)
        except RuntimeError:
            parse_input = []
            o_gui.make_visible_listsubruns_warning(True)

        return parse_input

    def list_subruns_2dplot_changed(self):
        self.list_subruns_2dplot()

    def list_subruns_2dplot_returned(self):
        return self.list_subruns_2dplot()

    def update_fit_peak_ranges_table(self, **kwargs):

        self.parent.ui.peak_range_table.blockSignals(True)

        def __get_kwargs_value(key='', data_type='boolean'):
            if data_type == 'boolean':
                _default = False
            elif data_type == 'array':
                _default = []
            return kwargs[key] if key in kwargs.keys() else _default

        # click = __get_kwargs_value('click', data_type='boolean')
        # move = __get_kwargs_value('move', data_type='boolean')
        # release = __get_kwargs_value('release', data_type='boolean')

        list_fit_peak_ranges = __get_kwargs_value('list_fit_peak_ranges',
                                                  data_type='array')
        # list_fit_peak_ranges_matplotlib_id = __get_kwargs_value('list_fit_peak_ranges_matplotlib_id',
        #                                                         data_type='array')
        list_fit_peak_labels = __get_kwargs_value('list_fit_peak_labels',
                                                  data_type='array')
        list_fit_peak_d0 = __get_kwargs_value('list_fit_peak_d0',
                                              data_type='array')

        o_gui = GuiUtilities(parent=self.parent)
        o_gui.reset_peak_range_table()
        o_gui.fill_peak_range_table(list_fit_peak_ranges=list_fit_peak_ranges,
                                    list_fit_peak_labels=list_fit_peak_labels,
                                    list_fit_peak_d0=list_fit_peak_d0)

        self.parent.ui.peak_range_table.blockSignals(False)
        o_gui.check_if_fitting_widgets_can_be_enabled()

    def update_fit_result_table(self):
        if self.parent.fit_result:
            self.parent.populate_fit_result_table(fit_result=self.parent.fit_result)

    def update_fit_peak_ranges_plot(self):
        # retrieve all peaks and labels from table
        table_ui = self.parent.ui.peak_range_table
        table_ui.blockSignals(True)

        nbr_row = table_ui.rowCount()

        list_peak_ranges = []
        list_fit_peak_labels = []
        list_fit_peak_d0 = []
        for _row in np.arange(nbr_row):
            _value1 = GuiUtilities.get_item_value(table_ui, _row, 0)
            _value2 = GuiUtilities.get_item_value(table_ui, _row, 1)

            try:
                _value1_float = np.float(_value1)
                _value2_float = np.float(_value2)
                _array = [_value1_float, _value2_float]

                _value1 = np.nanmin(_array)
                _value2 = np.nanmax(_array)

                _item0 = QTableWidgetItem("{:.3f}".format(_value1))
                self.parent.ui.peak_range_table.setItem(_row, 0, _item0)

                _item1 = QTableWidgetItem("{:.3f}".format(_value2))
                self.parent.ui.peak_range_table.setItem(_row, 1, _item1)

                list_peak_ranges.append([_value1, _value2])

            except ValueError:
                continue

            _label = GuiUtilities.get_item_value(table_ui, _row, 2)
            list_fit_peak_labels.append(_label)

            _d0 = np.float(str(GuiUtilities.get_item_value(table_ui, _row, 3)))
            list_fit_peak_d0.append(_d0)

        # replace the list_peak_ranges and list_fit_peak_labels from mplfitplottingwidget.py
        self.parent._ui_graphicsView_fitSetup.list_peak_ranges = list_peak_ranges
        self.parent._ui_graphicsView_fitSetup.list_fit_peak_labels = list_fit_peak_labels
        self.parent._ui_graphicsView_fitSetup.list_fit_peak_d0 = list_fit_peak_d0
        self.parent._ui_graphicsView_fitSetup.plot_data_with_fitting_ranges()

        table_ui.blockSignals(False)

    def __retrieving_json_file_name(self, save_file=True):
        file_filter = 'JSON (*.json)'
        json_file_name = browse_file(self.parent,
                                     'Peak Range File',
                                     os.getcwd(),
                                     file_filter,
                                     file_list=False,
                                     save_file=save_file)
        return json_file_name

    def save_peak_range(self):
        json_file_name = self.__retrieving_json_file_name(save_file=True)

        if json_file_name is None:
            return  # user clicked cancel

        # retrieve peak infos
        list_peak_ranges = self.parent._ui_graphicsView_fitSetup.list_peak_ranges
        list_peak_labels = self.parent._ui_graphicsView_fitSetup.list_fit_peak_labels
        list_peak_d0 = self.parent.list_peak_d0

        # create dictionary
        dict = {}
        for _index, peak_range in enumerate(list_peak_ranges):
            dict[_index] = {'peak_range': peak_range,
                            'peak_label': list_peak_labels[_index],
                            'd0': list_peak_d0}

        with open(json_file_name, 'w') as outfile:
            json.dump(dict, outfile)

    def load_peak_range(self):
        json_file_name = self.__retrieving_json_file_name(save_file=False)

        if json_file_name is None:
            return  # user clicked cancel

        with open(json_file_name) as json_file:
            _dict = json.load(json_file)

        peak_range = []
        peak_label = []
        peak_d0 = []
        for _index in _dict.keys():
            peak_range.append(_dict[_index]['peak_range'])
            peak_label.append(_dict[_index]['peak_label'])
            peak_d0.append(_dict[_index]['d0'])

        # save peak infos
        self.parent._ui_graphicsView_fitSetup.list_peak_ranges = peak_range
        self.parent._ui_graphicsView_fitSetup.list_fit_peak_labels = peak_label
        self.parent.list_peak_d0 = peak_d0

        self.parent.update_peak_ranges_table(release=True,
                                             list_fit_peak_labels=peak_label,
                                             list_fit_peak_ranges=peak_range,
                                             list_fit_peak_d0=peak_d0,
                                             list_fit_peak_ranges_matplotlib_id=[],
                                             list_fit_peak_labels_matplotlib_id=[])
        self.parent._ui_graphicsView_fitSetup.plot_data_with_fitting_ranges()

    def peak_range_table_right_click(self, position=-1):

        nbr_row = self.parent.ui.peak_range_table.rowCount()
        if nbr_row == 0:
            return

        menu = QMenu(self.parent)
        _remove_row = menu.addAction("Remove")
        action = menu.exec_(QCursor.pos())

        if action == _remove_row:
            self.remove_peak_range_table_row()

        o_gui = GuiUtilities(parent=self.parent)
        o_gui.check_if_fitting_widgets_can_be_enabled()

    def remove_peak_range_table_row(self):
        row_selected = self.parent.ui.peak_range_table.selectedRanges()[0]
        row_to_remove = row_selected.topRow()
        self.parent.ui.peak_range_table.removeRow(row_to_remove)

        new_list_peak_ranges = []
        new_list_peak_labels = []
        new_list_matplotlib_id = []
        old_list_peak_label = self.parent._ui_graphicsView_fitSetup.list_fit_peak_labels
        old_list_matplotlib_id = self.parent._ui_graphicsView_fitSetup.list_peak_labels_matplotlib_id
        for _row, peak_range in enumerate(self.parent._ui_graphicsView_fitSetup.list_peak_ranges):
            if _row == row_to_remove:
                _peak_label_id = old_list_matplotlib_id[_row]
                _peak_label_id.remove()
                continue

            new_list_peak_ranges.append(peak_range)
            new_list_peak_labels.append(old_list_peak_label[_row])
            new_list_matplotlib_id.append(old_list_matplotlib_id[_row])

        self.parent._ui_graphicsView_fitSetup.list_fit_peak_labels = new_list_peak_labels
        self.parent._ui_graphicsView_fitSetup.list_peak_ranges = new_list_peak_ranges
        self.parent._ui_graphicsView_fitSetup.list_peak_labels_matplotlib_id = new_list_matplotlib_id

        self.parent._ui_graphicsView_fitSetup.plot_data_with_fitting_ranges()

    def fit_table_selection_changed(self):
        '''as soon as a row is selected, switch to the slider view and go to right sub_run'''
        row_selected = GuiUtilities.get_row_selected(table_ui=self.parent.ui.tableView_fitSummary)
        if row_selected is None:
            return
        self.parent.ui.radioButton_individualSubRuns.setChecked(True)
        self.parent.check_subRunsDisplayMode()
        self.parent.ui.horizontalScrollBar_SubRuns.setValue(row_selected+1)
        self.parent.plot_scan()
