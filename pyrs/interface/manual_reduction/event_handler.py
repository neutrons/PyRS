import os

from pyrs.interface.gui_helper import browse_file
from pyrs.interface.gui_helper import pop_message
from pyrs.interface.manual_reduction.load_project_file import LoadProjectFile


class EventHandler:

    def __init__(self, parent=None):
        self.parent = parent

    def load_nexus_file(self):
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

    def convert_to_project_file(self, nexus_filename):
        """
        Convert nexus_filename into a project file
        :param nexus_filename:
        """
        return

    def load_project_file(self):
        """ Load project file in HDF5 format
                :return:
                """
        project_h5_name = browse_file(self.parent,
                                      'HIDRA Project File',
                                      os.getcwd(),
                                      file_filter='*.hdf5;;*.h5',
                                      file_list=False,
                                      save_file=False)

        try:
            o_load = LoadProjectFile(parent=self.parent)
            o_load.load_hydra_file(project_h5_name)
        except RuntimeError as run_err:
            pop_message(self,
                        'Failed to load project file {}: {}'.format(project_h5_name, run_err),
                        None, 'error')
        else:
            print('Loaded {} to {}'.format(project_h5_name, self._project_data_id))