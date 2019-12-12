from pyrs.interface.gui_helper import browse_dir


class Export:

    def __init__(self, parent=None):
        self.parent = parent

    def select_output_folder(self):
        pass


class ExportCSV(Export):

    def select_output_folder(self):
        print("project name is : {}".format(self.parent._project_name))
         out_folder = browse_dir(self,
                                 caption='Choose a file to save fitted peaks to',
                                                                      default_dir=self._core.working_dir,
                                                                      file_filter='HDF (*.hdf5)',
                                                                      save_file=True)
