from pyrs.interface.peak_fitting.config import LIST_AXIS_TO_PLOT


class GuiUtilities:

    def __init__(self, parent=None):
        self.parent = parent

    def enabled_fitting_widgets(self, enabled=True):
        list_widgets = [self.parent.ui.groupBox_FittingFunctions,
                        self.parent.ui.groupBox_SubRuns,
                        ]
        self.enabled_list_widgets(list_widgets=list_widgets,
                                  enabled=enabled)

    def enabled_1dplot_widgets(self, enabled=True):
        list_widgets = [self.parent.ui.frame_1dplot,
                        self.parent.ui.graphicsView_fitResult_frame,
                        ]
        self.enabled_list_widgets(list_widgets=list_widgets,
                                  enabled=enabled)

    def enabled_2dplot_widgets(self, enabled=True):
        list_widgets = [self.parent.ui.frame_2dplot,
                        self.parent.ui.widget_contour_plot,
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
            self._update_plots_combobox_items(list_axis_to_plot=_list_axis_to_plot)

        if fill_fit:
            _list_axis_to_plot = LIST_AXIS_TO_PLOT['fit']
            self._update_plots_combobox_items(list_axis_to_plot=_list_axis_to_plot)

        GuiUtilities.unblock_widgets(list_ui=[self.parent.ui.comboBox_xaxisNames,
                                              self.parent.ui.comboBox_yaxisNames])

        # enabled the 1D and 2D plot widgets
        self.enabled_1dplot_widgets(enabled=True)
        self.enabled_2dplot_widgets(enabled=True)

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
