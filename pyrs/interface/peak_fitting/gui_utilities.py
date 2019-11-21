class GuiUtilities:

    def __init__(self, parent=None):
        self.parent = parent

    def enabled_fitting_widgets(self, enabled=True):
        list_widgets = [self.parent.ui.groupBox_FittingFunctions,
                        self.parent.ui.groupBox_SubRuns,
                        ]
        self.enabled_list_widgets(list_widgets=list_widgets,
                                  enabled=enabled)

    def enabled_list_widgets(self, list_widgets=[], enabled=True):
        for _widget in list_widgets:
            _widget.setEnabled(enabled)

    # def check_prev_next_sub_runs_widgets(self):
    #
    #     enabled_next = True
    #     enabled_prev = True
    #
    #     if str(self.parent.ui.lineEdit_ScanNumbers.text()) == str(self.parent.ui.label_MinScanNumber.text()):
    #         enabled_prev = False
    #     elif str(self.parent.ui.lineEdit_ScanNumbers.text()) == str(self.parent.ui.label_MaxScanNumber.text()):
    #         enabled_next = False
    #
    #     self.parent.ui.pushButton_PlotPreviousScan.setEnabled(enabled_prev)
    #     self.parent.ui.pushButton_PlotNextScan.setEnabled(enabled_next)

    def enabled_data_fit_plot(self, enabled=True):
        self.parent._ui_graphicsView_fitSetup.setEnabled(enabled)
