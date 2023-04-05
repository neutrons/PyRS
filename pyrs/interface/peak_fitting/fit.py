from collections import namedtuple
import numpy as np

from pyrs.interface.peak_fitting.plot import Plot
from pyrs.interface.peak_fitting.utilities import Utilities
from pyrs.interface.peak_fitting.gui_utilities import GuiUtilities
from pyrs.peaks import FitEngineFactory as PeakFitEngineFactory  # type: ignore
from qtpy.QtWidgets import QApplication  # type:ignore
from qtpy.QtCore import Qt  # type: ignore

PeakInfo = namedtuple('PeakInfo', 'center left_bound right_bound tag')


class Fit:

    def __init__(self, parent=None):
        self.parent = parent

    def fit_multi_peaks(self):

        QApplication.setOverrideCursor(Qt.WaitCursor)

        _peak_range_list = [tuple(_range) for _range in self.parent._ui_graphicsView_fitSetup.list_peak_ranges]
        _peak_center_list = [np.mean([left, right]) for (left, right) in _peak_range_list]
        _peak_tag_list = ["peak{}".format(_index) for _index, _ in enumerate(_peak_center_list)]
        _peak_function_name = str(self.parent.ui.comboBox_peakType.currentText())

        _peak_xmin_list = [left for (left, _) in _peak_range_list]
        _peak_xmax_list = [right for (_, right) in _peak_range_list]

        # Fit peak
        hd_ws = self.parent.hidra_workspace

        _wavelength = hd_ws.get_wavelength(True, True)
        fit_engine = PeakFitEngineFactory.getInstance(hd_ws,
                                                      _peak_function_name, 'Linear',
                                                      wavelength=_wavelength)
        fit_result = fit_engine.fit_multiple_peaks(_peak_tag_list,
                                                   _peak_xmin_list,
                                                   _peak_xmax_list)
        self.parent.fit_result = fit_result

        self.parent.populate_fit_result_table(fit_result=fit_result)
        # self.parent.update_list_of_2d_plots_axis()

        o_gui = GuiUtilities(parent=self.parent)
        o_gui.set_1D_2D_axis_comboboxes(with_clear=True, fill_raw=True, fill_fit=True)
        o_gui.initialize_combobox()
        o_gui.enabled_export_csv_widgets(enabled=True)
        o_gui.enabled_2dplot_widgets(enabled=True)

        o_plot = Plot(parent=self.parent)
        o_plot.plot_2d()

        QApplication.restoreOverrideCursor()

    def initialize_fitting_table(self):
        # Set the table
        if self.parent.ui.tableView_fitSummary.rowCount() > 0:
            self.parent.ui.tableView_fitSummary.remove_all_rows()

        o_utility = Utilities(parent=self.parent)
        sub_run_list = o_utility.get_subruns_limit()
        self.parent.ui.tableView_fitSummary.init_exp(sub_run_list)
