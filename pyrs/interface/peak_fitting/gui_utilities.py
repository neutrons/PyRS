LIST_AXIS_TO_PLOT = {'Sub-runs': 'subrun', 'sx': 'sx', 'sy': 'sy', 'sz': 'sz',
                     'vx': 'vx', 'vy': 'vy', 'vz': 'vz',
                     'phi': 'phi', 'chi': 'chi', 'omega': 'omega',
                     'Peak Height': 'PeakHeight',
                     'Full Width Half Max': 'FWHM', 'intensity':'intensity',
                     'PeakCenter': 'PeakCenter',
                     'd-spacing': 'd-spacing'}


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

    def set_1D_2D_axis_comboboxes(self):
        # Set the widgets about viewer: get the sample logs and add the combo boxes for plotting
        sample_log_names = self.parent._core.reduction_service.get_sample_logs_names(self.parent._project_name,
                                                                                     can_plot=True)

        GuiUtilities.block_widgets(list_ui=[self.parent.ui.comboBox_xaxisNames,
                                            self.parent.ui.comboBox_yaxisNames,
                                            self.parent.ui.comboBox_yaxisNames_2dplot,
                                            self.parent.ui.comboBox_xaxisNames_2dplot])

        self.parent.ui.comboBox_xaxisNames.clear()
        self.parent.ui.comboBox_yaxisNames.clear()
        self.parent.ui.comboBox_xaxisNames_2dplot.clear()
        self.parent.ui.comboBox_yaxisNames_2dplot.clear()

        # Maintain a copy of sample logs!
        self.parent._sample_log_names = list(set(sample_log_names))
        self.parent._sample_log_names.sort()

        for sample_log in LIST_AXIS_TO_PLOT:
            self.parent.ui.comboBox_xaxisNames.addItem(sample_log)
            self.parent.ui.comboBox_yaxisNames.addItem(sample_log)
            self.parent.ui.comboBox_xaxisNames_2dplot.addItem(sample_log)
            self.parent.ui.comboBox_yaxisNames_2dplot.addItem(sample_log)
            self.parent._sample_log_name_set.add(sample_log)

        GuiUtilities.unblock_widgets(list_ui=[self.parent.ui.comboBox_xaxisNames,
                                              self.parent.ui.comboBox_yaxisNames])

        # enabled the 1D and 2D plot widgets
        self.enabled_1dplot_widgets(enabled=True)
        self.enabled_2dplot_widgets(enabled=True)

    @staticmethod
    def block_widgets(list_ui=[]):
        GuiUtilities.__block_widgets(list_ui, True)

    @staticmethod
    def unblock_widgets(list_ui=[]):
        GuiUtilities.__block_widgets(list_ui, True)

    @staticmethod
    def __block_widgets(list_ui, block):
        for _ui in list_ui:
            _ui.blockSignals(block)
