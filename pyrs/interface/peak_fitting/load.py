import os

from pyrs.interface.gui_helper import pop_message
from pyrs.interface.peak_fitting.gui_utilities import GuiUtilities
from pyrs.interface.peak_fitting.utilities import Utilities
from pyrs.projectfile import HidraProjectFile  # type: ignore


class Load:

    def __init__(self, parent=None):
        self.parent = parent

    def load(self, project_file=None):
        if project_file is None:
            return

        try:
            self.__set_up_project_name(project_file=project_file)
            ws = self.__load_multiple_file(project_file)

            # Record data key and next
            self.parent._curr_file_name = self.__parse_working_files(project_file)
            self.parent.hidra_workspace = ws
            self.parent.fit_result = None
            self.parent.create_plot_color_range()
        except (RuntimeError, TypeError) as run_err:
            pop_message(self, 'Unable to load {}'.format(project_file),
                        detailed_message=str(run_err),
                        message_type='error')

        # Get and set the range of sub runs
        o_utility = Utilities(parent=self.parent)
        sub_run_list = o_utility.get_subruns_limit()

        o_gui = GuiUtilities(parent=self.parent)
        o_gui.initialize_fitting_slider(max=len(sub_run_list))

        o_gui.set_1D_2D_axis_comboboxes(with_clear=True, fill_raw=True)
        o_gui.enabled_1dplot_widgets(enabled=True)
        o_gui.initialize_combobox()

        self.parent.ui.graphicsView_plot2D.reset_viewer()

    def __load_multiple_file(self, project_files):
        '''
        Load project files for peak fitting

        Parameters
        ----------
        project_files : Hidraproject file
            DESCRIPTION.

        Returns
        -------
        _hidra_ws : HIDRAWORKSPACE

        '''
        _hidra_ws = self.parent._core.load_hidra_project(project_files[0],
                                                         project_name=self.parent._project_name,
                                                         load_detector_counts=False,
                                                         load_diffraction=True)

        for project in project_files[1:]:
            _project = HidraProjectFile(project)
            _hidra_ws.append_hidra_project(_project)
            _project.close()

        return _hidra_ws

    def __set_up_project_name(self, project_file=""):
        """Keep the basename and removed the nxs and h5 extenstions"""
        if type(project_file) is list:
            self.parent._project_name = 'HB2B' + ''.join(['_{}'.format(run.split('.')[0].split('_')[-1])
                                                          for run in project_file])
        else:
            self.parent._project_name = os.path.basename(project_file).split('.')[0]

    def __parse_working_files(self, project_file=""):
        """Keep the filepath and append runs being fitted"""
        if type(project_file) is list:
            return project_file[0].split('HB2B')[0] + ''.join(['HB2B_{} '.format(run.split('.')[0].split('_')[-1])
                                                               for run in project_file])
        else:
            return project_file
