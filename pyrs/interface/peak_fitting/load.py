import os

from pyrs.interface.gui_helper import pop_message
from pyrs.interface.peak_fitting.gui_utilities import GuiUtilities
from pyrs.interface.peak_fitting.utilities import Utilities


class Load:

    def __init__(self, parent=None):
        self.parent = parent

    def load(self, project_file=None):
        if project_file is None:
            return

        try:
            self.__set_up_project_name(project_file=project_file)
            self.parent._core.load_hidra_project(project_file,
                                                 project_name=self.parent._project_name,
                                                 load_detector_counts=False,
                                                 load_diffraction=True)
            # Record data key and next
            self.parent._curr_file_name = project_file
        except (RuntimeError, TypeError) as run_err:
            pop_message(self, 'Unable to load {}'.format(project_file),
                        detailed_message=str(run_err),
                        message_type='error')

        # Edit information on the UI for user to visualize
        self.parent.ui.label_loadedFileInfo.setText('Loaded {}; Project name: {}'
                                                    .format(project_file, self.parent._project_name))

        # Get and set the range of sub runs
        o_utility = Utilities(parent=self.parent)
        sub_run_list = o_utility.get_subruns_limit(self.parent._project_name)

        o_gui = GuiUtilities(parent=self.parent)
        o_gui.initialize_fitting_slider(max=len(sub_run_list))

        # Set the widgets about viewer: get the sample logs and add the combo boxes for plotting
        sample_log_names = self.parent._core.reduction_service.get_sample_logs_names(self.parent._project_name,
                                                                                     can_plot=True)
        self._set_sample_logs_for_plotting(sample_log_names)


        # try:
        #     # Auto fit for all the peaks
        #     if self.parent.ui.checkBox_autoFit.isChecked():
        #         o_fit = Fit(parent=self.parent)
        #         o_fit.fit_peaks(all_sub_runs=True)
        # except (AttributeError) as err:
        #     pop_message(self, 'some errors during fitting!', detailed_message=str(err),
        #                 message_type='warning')

        # enabled all fitting widgets
        o_gui.enabled_fitting_widgets(True)

    def __set_up_project_name(self, project_file=""):
        """Keep the basename and removed the nxs and h5 extenstions"""
        self.parent._project_name = os.path.basename(project_file).split('.')[0]

    def _set_sample_logs_for_plotting(self, sample_log_names):
        """ There are 2 combo boxes containing sample logs' names for plotting.  Clear the existing ones
        and add the sample log names specified     def get_subruns_limit(self):
        sub_run_list = self.parent._core.reduction_service.get_sub_runs(self.parent._project_name)
        # self.parent.ui.label_logIndexMin.setText(str(sub_run_list[0]))
        # self.parent.ui.label_logIndexMax.setText(str(sub_run_list[-1]))
        # self.parent.ui.label_MinScanNumber.setText(str(sub_run_list[0]))
        # self.parent.ui.label_MaxScanNumber.setText(str(sub_run_list[-1]))
        return sub_run_list
to them
        :param sample_log_names:
        :return:
        """
        self.parent._sample_log_names_mutex = True
        self.parent.ui.comboBox_xaxisNames.clear()
        self.parent.ui.comboBox_yaxisNames.clear()

        # Maintain a copy of sample logs!
        self.parent._sample_log_names = list(set(sample_log_names))
        self.parent._sample_log_names.sort()

        for sample_log in sample_log_names:
            self.parent.ui.comboBox_xaxisNames.addItem(sample_log)
            self.parent.ui.comboBox_yaxisNames.addItem(sample_log)
            self.parent._sample_log_name_set.add(sample_log)
        self.parent._sample_log_names_mutex = False
