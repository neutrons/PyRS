import os

from pyrs.interface.gui_helper import browse_dir
from pyrs.core.summary_generator import SummaryGenerator
from pyrs.core import MonoSetting  # type: ignore


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
        try:
            peaks = self.parent.fit_result.peakcollections
            sample_logs = self.parent.hidra_workspace._sample_logs
            monosetting = MonoSetting.getFromRotation(self.parent.hidra_workspace.get_sample_log_value('mrot', 1))

            print("sample_log: {}".format(sample_logs))

            generator = SummaryGenerator(self._csv_file_name,
                                         log_list=sample_logs.keys())

            generator.setHeaderInformation({'cal_wavelength': self.parent.hidra_workspace.get_wavelength(False, False),
                                            'mono_wavelength': monosetting.value,
                                            'mono_setting': monosetting.name,
                                            'cal_file': self.parent.hidra_workspace.calibration_file})

            generator.write_csv(sample_logs, peaks)

            new_message = self.parent.current_root_statusbar_message + "\t\t\t\t Last Exported CSV: {}" \
                                                                       "".format(self._csv_file_name)
            self.parent.ui.statusbar.showMessage(new_message)
        except AttributeError:
            pass

    def _retrieve_project(self):
        _hidra_project_file = self.parent.hidra_workspace._project_file
        return _hidra_project_file
