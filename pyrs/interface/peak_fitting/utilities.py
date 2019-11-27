from pyrs.interface.gui_helper import parse_integers
from pyrs.interface.gui_helper import parse_integer
from pyrs.interface.gui_helper import pop_message
from pyrs.utilities import hb2b_utilities


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
        return sub_run_list

    def get_default_hdf(self):
        """
        use IPTS and run number to determine the name of the project file
        By default, look into the /HFIR/HB2B/IPTS-####/shared/pyrs_reduction folder
        if the project file for the given run number exists.
        If it does not, look into the /HFIR/HB2B/IPTS-####/autoreduce folder
        """
        try:
            ipts_number = parse_integer(self.parent.ui.lineEdit_iptsNumber)
            exp_number = parse_integer(self.parent.ui.lineEdit_expNumber)
        except RuntimeError as run_err:
            pop_message(self, 'Unable to parse IPTS or Exp due to {0}'.format(run_err))
            return None

        # Locate default saved HidraProject data
        archive_data = hb2b_utilities.get_hb2b_raw_data(ipts_number, exp_number)

        return archive_data
