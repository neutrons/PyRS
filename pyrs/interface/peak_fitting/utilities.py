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
