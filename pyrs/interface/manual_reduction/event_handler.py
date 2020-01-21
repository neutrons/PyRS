import os

from pyrs.interface.gui_helper import browse_file
from pyrs.interface.gui_helper import pop_message
from pyrs.interface.manual_reduction.load_project_file import LoadProjectFile


class EventHandler:

    def __init__(self, parent=None):
        self.parent = parent

        # project ID (current)
        self._project_data_id = None

    def browse_nexus_file(self):
        """
        allow users to browse for a nexus file to convert to project file
        """
        nexus_file = browse_file(self.parent,
                                 caption='Select a NeXus file',
                                 default_dir=self.parent._core.working_dir,
                                 file_filter='NeXus (*.h5)',
                                 file_list=False,
                                 save_file=False)

        self.convert_to_project_file(nexus_file)

    @staticmethod
    def convert_to_project_file(nexus_filename):
        """
        Convert nexus_filename into a project file
        :param nexus_filename:
        """
        # TODO - Implement!

        return

    @staticmethod
    def browse_project_file(parent):
        """Browse Hidra project file in h5 format

        Parameters
        ----------
        parent

        Returns
        -------
        str or None
            project file name or None for user's canceling the browse operation

        """
        project_h5_name = browse_file(parent,
                                      'HIDRA Project File',
                                      os.getcwd(),
                                      file_filter='*.hdf5;;*.h5',
                                      file_list=False,
                                      save_file=False)

        return project_h5_name

    @staticmethod
    def load_project_file(parent, file_name):

        try:
            o_load = LoadProjectFile(parent=parent)
            self._project_data_id = o_load.load_hydra_file(project_h5_name)
        except RuntimeError as run_err:
            pop_message(self,
                        'Failed to load project file {}: {}'.format(project_h5_name, run_err),
                        None, 'error')
        else:
            print('Loaded {} to {}'.format(project_h5_name, self._project_data_id))
