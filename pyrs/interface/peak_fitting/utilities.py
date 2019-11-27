from pyrs.interface.gui_helper import parse_integers


class Utilities:

    def __init__(self, parent=None):
        self.parent = parent

    def parse_sub_runs(self):
        """ Parse sub run numbers specified in lineEdit_scanNumbers
        :return: List (of integers) or None
        """
        int_string_list = str(self.parent.ui.lineEdit_scanNumbers.text()).strip()
        if len(int_string_list) == 0 or not self.parent.ui.fit_selected.isChecked():
            sub_run_list = None  # not set and thus default for all
        else:
            sub_run_list = parse_integers(int_string_list)

        return sub_run_list

    def get_subruns_limit(self, project_name):
        sub_run_list = self.parent._core.reduction_service.get_sub_runs(project_name)
        # self.parent.ui.label_logIndexMin.setText(str(sub_run_list[0]))
        # self.parent.ui.label_logIndexMax.setText(str(sub_run_list[-1]))
        # self.parent.ui.label_MinScanNumber.setText(str(sub_run_list[0]))
        # self.parent.ui.label_MaxScanNumber.setText(str(sub_run_list[-1]))
        return sub_run_list
