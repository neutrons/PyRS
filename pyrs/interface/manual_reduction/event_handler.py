from pyrs.interface.gui_helper import browse_file


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

