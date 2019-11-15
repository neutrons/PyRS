class GuiUtilities:

    def __init__(self, parent=None):
        self.parent = parent

    def enabled_fitting_widgets(self, enabled=True):
        list_widgets = [self.parent.ui.label_sub_runs,
                        self.parent.ui.label_functions,
                        self.parent.ui.lineEdit_scanNumbers,
                        self.parent.ui.pushButton_plotPreviousScan,
                        self.parent.ui.pushButton_plotNextScan,
                        self.parent.ui.label_min,
                        self.parent.ui.label_max,
                        self.parent.ui.label_logIndexMin,
                        self.parent.ui.label_logIndexMax,
                        self.parent.ui.checkBox_keepPrevPlot,
                        self.parent.ui.pushButton_plotPeaks,
                        self.parent.ui.comboBox_peakType,
                        self.parent.ui.comboBox_backgroundType,
                        self.parent.ui.pushButton_fitPeaks,
                        self.parent.ui.checkBox_fitSubRuns,
                        self.parent.ui.checkBox_autoSaveFitResult,
                        self.parent.ui.pushButton_saveFitResult,
                       ]
        self.enabled_list_widgets(list_widgets=list_widgets,
                                  enabled=enabled)

    def enabled_list_widgets(self, list_widgets=[], enabled=True):
        for _widget in list_widgets:
            _widget.setEnabled(enabled)

    def check_prev_next_sub_runs_widgets(self):

        enabled_next = True
        enabled_prev = True

        if (str(self.parent.ui.lineEdit_scanNumbers.text()) == str(self.parent.ui.label_logIndexMin.text())):
            enabled_prev = False
        elif (str(self.parent.ui.lineEdit_scanNumbers.text()) == str(self.parent.ui.label_logIndexMax.text())):
            enabled_next = False

        self.parent.ui.pushButton_plotNextScan.setEnabled(enabled_next)
        self.parent.ui.pushButton_plotPreviousScan.setEnabled(enabled_prev)
