import numpy as np
from qtpy.QtWidgets import QTableWidgetItem

from pyrs.interface.peak_fitting.config import LIST_AXIS_TO_PLOT
from pyrs.interface.peak_fitting.config import DEFAUT_AXIS


class GuiUtilities:

    def __init__(self, parent=None):
        self.parent = parent

    def enabled_fitting_widgets(self, enabled=True):
        list_widgets = [self.parent.ui.groupBox_FittingFunctions,
                        self.parent.ui.groupBox_SubRuns,
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
                        self.parent.ui.graphicsView_fitResult_frame,
                        ]
        self.enabled_list_widgets(list_widgets=list_widgets,
                                  enabled=enabled)

    def enabled_2dplot_widgets(self, enabled=True):
        list_widgets = [self.parent.ui.frame_2dplot,
                        self.parent.ui.graphicsView_2dPlot_frame,
                        ]
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
        sample_log_names = self.parent._core.reduction_service.get_sample_logs_names(self.parent._project_name,
                                                                                     can_plot=True)
        GuiUtilities.block_widgets(list_ui=[self.parent.ui.comboBox_xaxisNames,
                                            self.parent.ui.comboBox_yaxisNames,
                                            self.parent.ui.comboBox_yaxisNames_2dplot,
                                            self.parent.ui.comboBox_yaxisNames_2dplot,
                                            self.parent.ui.comboBox_zaxisNames_2dplot])

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
            print("filling raw")
            print(_list_axis_to_plot)
            print("------------")
            self._update_plots_combobox_items(list_axis_to_plot=_list_axis_to_plot)

        if fill_fit:
            _list_axis_to_plot = LIST_AXIS_TO_PLOT['fit']
            self._update_plots_combobox_items(list_axis_to_plot=_list_axis_to_plot)

        GuiUtilities.unblock_widgets(list_ui=[self.parent.ui.comboBox_xaxisNames,
                                              self.parent.ui.comboBox_yaxisNames])

        # enabled the 1D and 2D plot widgets
        self.enabled_1dplot_widgets(enabled=True)
        self.enabled_2dplot_widgets(enabled=True)

    def initialize_combobox(self):
        self.initialize_combobox_1d()
        self.initialize_combobox_2d()

    def initialize_combobox_1d(self):
        _index_xaxis = self.parent.ui.comboBox_xaxisNames.findText(DEFAUT_AXIS['1d']['xaxis'])
        self.parent.ui.comboBox_xaxisNames.setCurrentIndex(_index_xaxis)
        _index_yaxis = self.parent.ui.comboBox_xaxisNames.findText(DEFAUT_AXIS['1d']['yaxis'])
        self.parent.ui.comboBox_yaxisNames.setCurrentIndex(_index_yaxis)

    def initialize_combobox_2d(self):
        _index_xaxis = self.parent.ui.comboBox_xaxisNames_2dplot.findText(DEFAUT_AXIS['2d']['xaxis'])
        self.parent.ui.comboBox_xaxisNames_2dplot.setCurrentIndex(_index_xaxis)
        _index_yaxis = self.parent.ui.comboBox_xaxisNames_2dplot.findText(DEFAUT_AXIS['2d']['yaxis'])
        self.parent.ui.comboBox_yaxisNames_2dplot.setCurrentIndex(_index_yaxis)
        _index_zaxis = self.parent.ui.comboBox_xaxisNames_2dplot.findText(DEFAUT_AXIS['2d']['zaxis'])
        self.parent.ui.comboBox_zaxisNames_2dplot.setCurrentIndex(_index_zaxis)

    def _update_plots_combobox_items(self, list_axis_to_plot=[]):
        _list_comboboxes = [self.parent.ui.comboBox_xaxisNames,
                            self.parent.ui.comboBox_yaxisNames,
                            self.parent.ui.comboBox_xaxisNames_2dplot,
                            self.parent.ui.comboBox_yaxisNames_2dplot,
                            self.parent.ui.comboBox_zaxisNames_2dplot]
        for sample_log in list_axis_to_plot:
            for _ui in _list_comboboxes:
                _ui.addItem(sample_log)
            self.parent._sample_log_name_set.add(sample_log)

    def make_visible_d01d_widgets(self, visible=True):
        list_ui = [self.parent.ui.label_d01d,
                   self.parent.ui.label_d0units1d,
                   self.parent.ui.lineEdit_d01d]
        GuiUtilities.make_visible_ui(list_ui=list_ui,
                                     visible=visible)

    def make_visible_d02d_widgets(self, visible=True):
        list_ui = [self.parent.ui.label_d02d,
                   self.parent.ui.label_d0units2d,
                   self.parent.ui.lineEdit_d02d]
        GuiUtilities.make_visible_ui(list_ui=list_ui,
                                     visible=visible)

    def check_axis1d_status(self):
        xaxis_selected = str(self.parent.ui.comboBox_xaxisNames.currentText())
        yaxis_selected = str(self.parent.ui.comboBox_yaxisNames.currentText())
        if (xaxis_selected == 'strain') or (yaxis_selected == 'strain'):
            self.make_visible_d01d_widgets(True)
        else:
            self.make_visible_d01d_widgets(False)

    def check_axis2d_status(self):
        xaxis_selected = str(self.parent.ui.comboBox_xaxisNames_2dplot.currentText())
        yaxis_selected = str(self.parent.ui.comboBox_yaxisNames_2dplot.currentText())
        zaxis_selected = str(self.parent.ui.comboBox_zaxisNames_2dplot.currentText())
        if (xaxis_selected == 'strain') or (yaxis_selected == 'strain') or \
           (zaxis_selected == 'strain'):
            self.make_visible_d02d_widgets(True)
        else:
            self.make_visible_d02d_widgets(False)

    def reset_peak_range_table(self):
        nbr_row = self.parent.ui.peak_range_table.rowCount()
        for _ in np.arange(nbr_row):
            self.parent.ui.peak_range_table.removeRow(0)

    def fill_peak_range_table(self, list_fit_peak_ranges=[],
                              list_fit_peak_labels=[]):

        for _index, _range in enumerate(list_fit_peak_ranges):
            self.parent.ui.peak_range_table.insertRow(_index)

            _item0 = QTableWidgetItem("{:.3f}".format(np.min(_range)))
            self.parent.ui.peak_range_table.setItem(_index, 0, _item0)

            _item1 = QTableWidgetItem("{:.3f}".format(np.max(_range)))
            self.parent.ui.peak_range_table.setItem(_index, 1, _item1)

            _label = QTableWidgetItem(list_fit_peak_labels[_index])
            self.parent.ui.peak_range_table.setItem(_index, 2, _label)

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
