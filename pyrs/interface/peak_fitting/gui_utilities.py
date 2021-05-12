import numpy as np
from qtpy.QtWidgets import QTableWidgetItem

from pyrs.interface.peak_fitting.config import LIST_AXIS_TO_PLOT
from pyrs.interface.peak_fitting.config import DEFAUT_AXIS


class GuiUtilities:

    def __init__(self, parent=None):
        self.parent = parent

    def enabled_sub_runs_interation_widgets(self, enabled=True):
        list_widgets = [self.parent.ui.groupBox_SubRuns,
                        ]
        self.enabled_list_widgets(list_widgets=list_widgets,
                                  enabled=enabled)

    def enabled_fitting_widgets(self, enabled=True):
        list_widgets = [self.parent.ui.groupBox_FittingFunctions,
                        ]
        self.enabled_list_widgets(list_widgets=list_widgets,
                                  enabled=enabled)

    def check_if_fitting_widgets_can_be_enabled(self):
        status = False
        nbr_row_fitting_table = self.parent.ui.peak_range_table.rowCount()
        if nbr_row_fitting_table > 0:
            status = True
        self.enabled_fitting_widgets(enabled=status)

    def enabled_export_csv_widgets(self, enabled=True):
        list_widgets = [self.parent.ui.pushButton_exportCSV,
                        ]
        self.enabled_list_widgets(list_widgets=list_widgets,
                                  enabled=enabled)

    def enabled_peak_ranges_widgets(self, enabled=True):
        list_widgets = [self.parent.ui.groupBox_peak_ranges,
                        ]
        self.enabled_list_widgets(list_widgets=list_widgets,
                                  enabled=enabled)

    def enabled_save_peak_range_widget(self, enabled=True):
        list_widgets = [self.parent.ui.pushButton_save_peak_range]
        self.enabled_list_widgets(list_widgets=list_widgets,
                                  enabled=enabled)

    def update_save_peak_range_widget_status(self):
        list_peak_ranges = self.parent._ui_graphicsView_fitSetup.list_peak_ranges
        if list_peak_ranges:
            enabled = True
        else:
            enabled = False
        self.enabled_save_peak_range_widget(enabled=enabled)

    def enabled_1dplot_widgets(self, enabled=True):
        list_widgets = [self.parent.ui.frame_1dplot,
                        # self.parent.ui.graphicsView_fitResult_frame,
                        self.parent.ui.graphicsView_fitResult,
                        ]
        self.enabled_list_widgets(list_widgets=list_widgets,
                                  enabled=enabled)

    def enabled_2dplot_widgets(self, enabled=True):
        list_widgets = [self.parent.ui.frame_2dplot,
                        self.parent.ui.graphicsView_2dPlot_frame,
                        ]
        self.enabled_list_widgets(list_widgets=list_widgets,
                                  enabled=enabled)

    def enabled_save_files_widget(self, enabled=False):
        list_widgets = [self.parent.ui.actionSave,
                        self.parent.ui.actionSaveAs]
        self.enabled_list_widgets(list_widgets=list_widgets,
                                  enabled=enabled)

    def make_visible_listsubruns_warning(self, visible=True):
        self.parent.ui.listsubruns_warning_icon.setVisible(visible)

    def enabled_list_widgets(self, list_widgets=[], enabled=True):
        for _widget in list_widgets:
            _widget.setEnabled(enabled)

    def enabled_data_fit_plot(self, enabled=True):
        self.parent._ui_graphicsView_fitSetup.setEnabled(enabled)

    def initialize_fitting_slider(self, max=-1):
        self.parent.ui.horizontalScrollBar_SubRuns.setMaximum(max)
        self.parent.ui.horizontalScrollBar_SubRuns.setValue(1)
        self.parent.ui.horizontalScrollBar_SubRuns.setMinimum(1)

    def check_subRuns_display_mode(self):
        individual_radio_button_status = self.parent.ui.radioButton_individualSubRuns.isChecked()

        list_ui_individual = [self.parent.ui.horizontalScrollBar_SubRuns,
                              self.parent.ui.label_SubRunsValue]
        list_ui_listruns = [self.parent.ui.label_listSubRunsHelp,
                            self.parent.ui.lineEdit_listSubRuns]

        self.enabled_list_widgets(list_widgets=list_ui_individual,
                                  enabled=individual_radio_button_status)
        self.enabled_list_widgets(list_widgets=list_ui_listruns,
                                  enabled=(not individual_radio_button_status))

    def set_1D_2D_axis_comboboxes(self, with_clear=False, fill_raw=False, fill_fit=False):
        # Set the widgets about viewer: get the sample logs and add the combo boxes for plotting
        sample_log_names = self.parent._core.reduction_service.get_sample_logs_names(self.parent._project_name)

        list_ui = [self.parent.ui.comboBox_xaxisNames,
                   self.parent.ui.comboBox_yaxisNames,
                   self.parent.ui.comboBox_xaxisNames_2dplot,
                   self.parent.ui.comboBox_yaxisNames_2dplot,
                   self.parent.ui.comboBox_zaxisNames_2dplot]
        GuiUtilities.block_widgets(list_ui=list_ui)

        if with_clear:
            self.parent.ui.comboBox_xaxisNames.clear()
            self.parent.ui.comboBox_yaxisNames.clear()
            self.parent.ui.comboBox_xaxisNames_2dplot.clear()
            self.parent.ui.comboBox_yaxisNames_2dplot.clear()

        # Maintain a copy of sample logs!
        self.parent._sample_log_names = list(set(sample_log_names))
        self.parent._sample_log_names.sort()

        if fill_raw:
            _list_axis_to_plot = LIST_AXIS_TO_PLOT['raw']
            self._update_plots_1D_combobox_items(list_axis_to_plot=_list_axis_to_plot)

        if fill_fit:
            _list_axis_to_plot = LIST_AXIS_TO_PLOT['fit']
            self._update_plots_1D_combobox_items(list_axis_to_plot=_list_axis_to_plot)
            self._update_plots_2D_combobox_items()

        GuiUtilities.unblock_widgets(list_ui=list_ui)

    def initialize_combobox(self):
        self.initialize_combobox_1d()
        self.initialize_combobox_2d()
        self.initialize_combobox_peak_label()

    def initialize_combobox_peak_label(self):
        list_of_labels = self.get_list_of_peak_label()

        list_ui = [self.parent.ui.plot1d_xaxis_peak_label_comboBox,
                   self.parent.ui.plot1d_yaxis_peak_label_comboBox,
                   self.parent.ui.plot2d_xaxis_peak_label_comboBox,
                   self.parent.ui.plot2d_yaxis_peak_label_comboBox,
                   self.parent.ui.plot2d_zaxis_peak_label_comboBox]

        GuiUtilities.block_widgets(list_ui)

        GuiUtilities.clear_comboboxes(list_ui=list_ui)
        GuiUtilities.fill_comboboxes(list_ui=list_ui, list_values=list_of_labels)

        GuiUtilities.unblock_widgets(list_ui)

    @staticmethod
    def fill_comboboxes(list_ui=[], list_values=[]):
        for _ui in list_ui:
            _ui.addItems(list_values)

    @staticmethod
    def clear_comboboxes(list_ui=[]):
        for _ui in list_ui:
            _ui.clear()

    def get_number_of_peak_selected(self):
        return len(self.get_list_of_peak_label())

    def get_list_of_peak_label(self):
        return self.parent._ui_graphicsView_fitSetup.list_fit_peak_labels

    def initialize_combobox_1d(self):
        _index_xaxis = self.parent.ui.comboBox_xaxisNames.findText(DEFAUT_AXIS['1d']['xaxis'])
        self.parent.ui.comboBox_xaxisNames.setCurrentIndex(_index_xaxis)
        _index_yaxis = self.parent.ui.comboBox_xaxisNames.findText(DEFAUT_AXIS['1d']['yaxis'])
        self.parent.ui.comboBox_yaxisNames.setCurrentIndex(_index_yaxis)

    def initialize_combobox_2d(self):
        list_ui = [self.parent.ui.comboBox_xaxisNames_2dplot,
                   self.parent.ui.comboBox_yaxisNames_2dplot,
                   self.parent.ui.comboBox_zaxisNames_2dplot]
        GuiUtilities.__block_widgets(list_ui, True)
        _index_xaxis = self.parent.ui.comboBox_xaxisNames_2dplot.findText(DEFAUT_AXIS['2d']['xaxis'])
        self.parent.ui.comboBox_xaxisNames_2dplot.setCurrentIndex(_index_xaxis)
        _index_yaxis = self.parent.ui.comboBox_yaxisNames_2dplot.findText(DEFAUT_AXIS['2d']['yaxis'])
        self.parent.ui.comboBox_yaxisNames_2dplot.setCurrentIndex(_index_yaxis)
        _index_zaxis = self.parent.ui.comboBox_zaxisNames_2dplot.findText(DEFAUT_AXIS['2d']['zaxis'])
        self.parent.ui.comboBox_zaxisNames_2dplot.setCurrentIndex(_index_zaxis)
        GuiUtilities.__block_widgets(list_ui, False)

    def _update_plots_1D_combobox_items(self, list_axis_to_plot=[]):
        _list_comboboxes = [self.parent.ui.comboBox_xaxisNames,
                            self.parent.ui.comboBox_yaxisNames]
        for sample_log in list_axis_to_plot:
            for _ui in _list_comboboxes:
                _ui.addItem(sample_log)
            self.parent._sample_log_name_set.add(sample_log)

    def _update_plots_2D_combobox_items(self):
        _list_xy_comboboxes = [self.parent.ui.comboBox_xaxisNames_2dplot,
                               self.parent.ui.comboBox_yaxisNames_2dplot]
        _list_z_comboboxes = [self.parent.ui.comboBox_zaxisNames_2dplot]
        for sample_log in LIST_AXIS_TO_PLOT['3d_axis']['xy_axis']:
            for _ui in _list_xy_comboboxes:
                _ui.addItem(sample_log)
            self.parent._sample_log_name_set.add(sample_log)
        for sample_log in LIST_AXIS_TO_PLOT['3d_axis']['z_axis']:
            for _ui in _list_z_comboboxes:
                _ui.addItem(sample_log)
            self.parent._sample_log_name_set.add(sample_log)

    def make_visible_peak_label_of_1d_widgets(self, visible=True):
        list_ui = [self.parent.ui.plot1d_peak_label,
                   self.parent.ui.plot1d_peak_label_comboBox]
        GuiUtilities.make_visible_ui(list_ui=list_ui,
                                     visible=visible)

    def check_axis1d_status(self):
        # check first if the widgets are enabled
        if self.parent.ui.comboBox_xaxisNames.isEnabled():
            xaxis_selected = str(self.parent.ui.comboBox_xaxisNames.currentText())
            yaxis_selected = str(self.parent.ui.comboBox_yaxisNames.currentText())

            if self.get_number_of_peak_selected() < 2:
                self.parent.ui.plot1d_xaxis_peak_label_comboBox.setVisible(False)
                self.parent.ui.plot1d_yaxis_peak_label_comboBox.setVisible(False)
            else:
                if xaxis_selected in LIST_AXIS_TO_PLOT['fit']:
                    self.parent.ui.plot1d_xaxis_peak_label_comboBox.setVisible(True)
                else:
                    self.parent.ui.plot1d_xaxis_peak_label_comboBox.setVisible(False)

                if yaxis_selected in LIST_AXIS_TO_PLOT['fit']:
                    self.parent.ui.plot1d_yaxis_peak_label_comboBox.setVisible(True)
                else:
                    self.parent.ui.plot1d_yaxis_peak_label_comboBox.setVisible(False)
        else:
            self.parent.ui.plot1d_xaxis_peak_label_comboBox.setVisible(False)
            self.parent.ui.plot1d_yaxis_peak_label_comboBox.setVisible(False)

    def get_plot1d_axis_peak_label_index(self, is_xaxis=True):
        if is_xaxis:
            ui = self.parent.ui.plot1d_xaxis_peak_label_comboBox
        else:
            ui = self.parent.ui.plot1d_yaxis_peak_label_comboBox

        if ui.isVisible():
            return ui.currentIndex()
        else:
            return 0

    def get_plot2d_axis_peak_label_index(self, axis='x'):
        if axis == 'x':
            ui = self.parent.ui.plot2d_xaxis_peak_label_comboBox
        elif axis == 'y':
            ui = self.parent.ui.plot2d_yaxis_peak_label_comboBox
        else:
            ui = self.parent.ui.plot2d_zaxis_peak_label_comboBox

        if ui.isVisible():
            return ui.currentIndex()
        else:
            return 0

    def check_axis2d_status(self):

        if self.parent.ui.comboBox_xaxisNames_2dplot.isEnabled():
            xaxis_selected = str(self.parent.ui.comboBox_xaxisNames_2dplot.currentText())
            yaxis_selected = str(self.parent.ui.comboBox_yaxisNames_2dplot.currentText())
            zaxis_selected = str(self.parent.ui.comboBox_zaxisNames_2dplot.currentText())

            if self.get_number_of_peak_selected() < 2:
                self.parent.ui.plot2d_xaxis_peak_label_comboBox.setVisible(False)
                self.parent.ui.plot2d_yaxis_peak_label_comboBox.setVisible(False)
                self.parent.ui.plot2d_zaxis_peak_label_comboBox.setVisible(False)
            else:
                if xaxis_selected in LIST_AXIS_TO_PLOT['3d_axis']:
                    self.parent.ui.plot2d_xaxis_peak_label_comboBox.setVisible(True)
                else:
                    self.parent.ui.plot2d_xaxis_peak_label_comboBox.setVisible(False)

                if yaxis_selected in LIST_AXIS_TO_PLOT['3d_axis']:
                    self.parent.ui.plot2d_yaxis_peak_label_comboBox.setVisible(True)
                else:
                    self.parent.ui.plot2d_yaxis_peak_label_comboBox.setVisible(False)

                if zaxis_selected in LIST_AXIS_TO_PLOT['fit']:
                    self.parent.ui.plot2d_zaxis_peak_label_comboBox.setVisible(True)
                else:
                    self.parent.ui.plot2d_zaxis_peak_label_comboBox.setVisible(False)

        else:
            self.parent.ui.plot2d_xaxis_peak_label_comboBox.setVisible(False)
            self.parent.ui.plot2d_yaxis_peak_label_comboBox.setVisible(False)
            self.parent.ui.plot2d_zaxis_peak_label_comboBox.setVisible(False)

    def reset_peak_range_table(self):
        nbr_row = self.parent.ui.peak_range_table.rowCount()
        for _ in np.arange(nbr_row):
            self.parent.ui.peak_range_table.removeRow(0)

    def fill_peak_range_table(self, list_fit_peak_ranges=[],
                              list_fit_peak_labels=[],
                              list_fit_peak_d0=[]):

        for _index, _range in enumerate(list_fit_peak_ranges):
            self.parent.ui.peak_range_table.insertRow(_index)

            _item0 = QTableWidgetItem("{:.3f}".format(np.min(_range)))
            self.parent.ui.peak_range_table.setItem(_index, 0, _item0)

            _item1 = QTableWidgetItem("{:.3f}".format(np.max(_range)))
            self.parent.ui.peak_range_table.setItem(_index, 1, _item1)

            _label = QTableWidgetItem(list_fit_peak_labels[_index])
            self.parent.ui.peak_range_table.setItem(_index, 2, _label)

            _value = QTableWidgetItem("{:.3f}".format(list_fit_peak_d0[_index]))
            self.parent.ui.peak_range_table.setItem(_index, 3, _value)

    @staticmethod
    def get_row_selected(table_ui=None):
        selection = table_ui.selectedRanges()
        if len(selection) > 0:
            return selection[0].topRow()
        else:
            return None

    @staticmethod
    def make_visible_ui(list_ui=[], visible=True):
        for _ui in list_ui:
            _ui.setVisible(visible)

    @staticmethod
    def block_widgets(list_ui=[]):
        GuiUtilities.__block_widgets(list_ui, True)

    @staticmethod
    def unblock_widgets(list_ui=[]):
        GuiUtilities.__block_widgets(list_ui, False)

    @staticmethod
    def __block_widgets(list_ui, block):
        for _ui in list_ui:
            _ui.blockSignals(block)

    @staticmethod
    def get_item_value(ui=None, row=-1, column=-1):
        _item = ui.item(row, column).text()
        return str(_item)
