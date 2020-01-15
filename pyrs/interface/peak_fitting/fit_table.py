from pyrs.core.peak_profile_utility import PeakShape


class FitTable:

    def __init__(self, parent=None, fit_result=None):
        self.parent = parent
        self.fit_result = fit_result

    def initialize_table(self):
        nbr_peaks = len(self.fit_result.peakcollections)
        self.parent.ui.spinBox_peak_index.setRange(1, nbr_peaks)

    def populate_fit_result_table(self):
        # get list of columns
        column_names = ["chi^2"]
        _peak_function_name = str(self.parent.ui.comboBox_peakType.currentText())
        for _value in PeakShape.getShape(_peak_function_name).native_parameters:
            column_names.append(_value)


