import os

from pyrs.interface.gui_helper import pop_message
from pyrs.interface.gui_helper import browse_file
from pyrs.interface.peak_fitting.load import Load
from pyrs.interface.peak_fitting.plot import Plot
from pyrs.interface.peak_fitting.fit import Fit
from pyrs.interface.peak_fitting.gui_utilities import GuiUtilities
# from pyrs.interface.peak_fitting.utilities import Utilities


class EventHandler:

    def __init__(self, parent=None):
        self.parent = parent

    def browse_load_plot_hdf(self):
        if self.parent._core is None:
            raise RuntimeError('Not set up yet!')

        # o_utility = Utilities(parent=self.parent)
        # hydra_file_name = o_utility.get_default_hdf()
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

        # Add file name to line edit to show
        self.parent.ui.lineEdit_expFileName.setText(hidra_file_name)  # REMOVE THIS ONCE ITS BEEN REPLACED
        self.parent.ui.statusbar.showMessage("Working with: {}".format(hidra_file_name))

        try:
            o_load = Load(parent=self.parent)
            o_load.load(project_file=hidra_file_name)

        except RuntimeError as run_err:
            pop_message(self, 'Failed to load {}'.format(hidra_file_name),
                        str(run_err), 'error')

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
            o_gui.enabled_fitting_widgets(True)
            o_gui.enabled_data_fit_plot(True)

        except RuntimeError as run_err:
            pop_message(self, 'Failed to initialize widgets for {}'.format(hidra_file_name),
                        str(run_err), 'error')

