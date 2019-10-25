import os

from pyrs.interface.gui_helper import pop_message
from pyrs.utilities.rs_project_file import HidraConstants


class LoadProjectFile:

    def __init__(self, parent=None):
        self.parent = parent

    def load_hydra_file(self, project_file_name):
        """
        Load Hidra project file to the core
        :param project_file_name:
        :return:
        """
        # Load data file
        project_name = os.path.basename(project_file_name).split('.')[0]
        try:
            self.parent._hydra_workspace = self.parent._core.load_hidra_project(project_file_name,
                                                                                project_name=project_name,
                                                                                load_detector_counts=True,
                                                                                load_diffraction=True)
        except (RuntimeError, IOError) as load_err:
            self.parent._hydra_workspace = None
            pop_message(self.parent, 'Loading {} failed'.format(project_file_name),
                                                  detailed_message='{}'.format(load_err),
                                                  message_type='error')
            return

        # Set value for the loaded project
        self.parent._project_file_name = project_file_name
        self.parent._project_data_id = project_name

        # Fill sub runs to self.ui.comboBox_sub_runs
        self.parent._set_sub_runs()

        # Set to first sub run and plot
        self.parent.ui.comboBox_sub_runs.setCurrentIndex(0)

        # Fill in self.ui.frame_subRunInfoTable
        meta_data_array = self.parent._core.reduction_manager.get_sample_logs_values(self.parent._project_data_id,
                                                                              [HidraConstants.SUB_RUNS,
                                                                               HidraConstants.TWO_THETA])
        self.parent.ui.rawDataTable.add_subruns_info(meta_data_array, clear_table=True)
