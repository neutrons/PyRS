import os

from pyrs.interface.gui_helper import browse_dir
from pyrs.utilities.rs_project_file import HidraProjectFile
from pyrs.core.summary_generator import SummaryGenerator


class Export:

    _output_folder_selected = ''
    _csv_file_name = ''

    def __init__(self, parent=None):
        self.parent = parent

    def select_output_folder(self):
        pass


class ExportCSV(Export):

    def select_output_folder(self):
        out_folder = browse_dir(self.parent,
                                caption='Choose where to create the CSV file',
                                default_dir=self.parent._core.working_dir)

        self._output_folder_selected = out_folder
        self._csv_file_name = os.path.join(out_folder, self.parent._project_name + '.csv')

    def create_csv(self):
        peaks = self.parent._core.get_peak(self.parent._project_name, 'Peak 1')
        sample_logs = self.parent.hidra_workspace._sample_logs

        generator = SummaryGenerator(self._csv_file_name,
                                     log_list=sample_logs.keys())
        generator.setHeaderInformation(dict())
        generator.write_csv(sample_logs, [peaks])

    def _retrieve_project(self):
        _hidra_project_file = self.parent.hidra_workspace._project_file
        return _hidra_project_file


