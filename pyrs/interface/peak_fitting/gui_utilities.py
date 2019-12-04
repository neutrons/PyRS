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
