try:
    from PyQt5.QtWidgets import QMainWindow, QFileDialog
except ImportError:
    from PyQt4.QtGui import QMainWindow, QFileDialog
import ui.ui_manualreductionwindow
from pyrs.core.pyrscore import PyRsCore
from pyrs.utilities import hb2b_utilities
import os
import gui_helper
import numpy


class ManualReductionWindow(QMainWindow):
    """
    GUI window for user to fit peaks
    """
    def __init__(self, parent):
        """
        initialization
        :param parent:
        """
        super(ManualReductionWindow, self).__init__(parent)

        # class variables
        self._core = None
        self._currIPTSNumber = None
        self._currExpNumber = None

        # set up UI
        self.ui = ui.ui_manualreductionwindow.Ui_MainWindow()
        self.ui.setupUi(self)

        # set up the event handling
        self.ui.pushButton_setIPTSNumber.clicked.connect(self.do_set_ipts_exp)
        self.ui.pushButton_browseOutputDir.clicked.connect(self.do_browse_output_dir)
        self.ui.pushButton_setCalibrationFile.clicked.connect(self.do_browse_set_caliration_file)

        self.ui.pushButton_batchReduction.clicked.connect(self.do_reduce_batch_runs)
        self.ui.pushButton_chopReduce.clicked.connect(self.do_chop_reduce_run)

        return

    def do_browse_output_dir(self):
        """
        browse and set output directory
        :return:
        """
        output_dir = gui_helper.browse_dir(self, caption='Output directory for reduced data',
                                           default_dir=os.path.expanduser('~'))
        if output_dir is not None or output_dir != '':
            self.ui.lineEdit_outputDir.setText(output_dir)
            self._core.reduction_engine.set_output_dir(output_dir)

        return

    def do_reduce_batch_runs(self):
        """
        (simply) reduce a list of runs in same experiment in a batch
        :return:
        """
        # get run numbers
        run_number_list = gui_helper.parse_integers(str(self.ui.lineEdit_runNumbersList.text()))

        error_msg = ''
        for run_number in run_number_list:
            try:
                self._core.reduction_engine.reduce_run(self._currIPTSNumber, self._expNumber, run_number)
            except RuntimeError as run_err:
                error_msg += 'Failed to reduce run {} due to {}\n'.format(run_number, run_err)
        # END-FOR

        # pop out error message if there is any
        if error_msg != '':
            error_msg = 'Reducing IPTS-{} Exp-{}:\n{}'.format(self._currIPTSNumber, self._currExpNumber, error_msg)
            gui_helper.pop_message(self, 'Batch reduction fails (partially or completely)', detailed_message=error_msg,
                                   message_type='error')
            return

        return

    def do_set_ipts_exp(self):
        """
        set IPTS number
        :return:
        """
        try:
            ipts_number = gui_helper.parse_integer(str(self.ui.lineEdit_iptsNumber.text()))
            exp_number = gui_helper.parse_integer(str(self.ui.lineEdit_expNumber.text()))
            self._currIPTSNumber = ipts_number
            self._currExpNumber = exp_number
            self._core.reduction_engine.set_ipts_exp_number(ipts_number, exp_number)
        except RuntimeError:
            gui_helper.pop_message(self, 'IPTS number shall be set to an integer.', message_type='error')

        return

    def setup_window(self, pyrs_core):
        """
        set up the manual reduction window from its parent
        :param pyrs_core:
        :return:
        """
        # check
        assert isinstance(pyrs_core, PyRsCore), 'Controller core {0} must be a PyRSCore instance but not a {1}.' \
                                                ''.format(pyrs_core, pyrs_core.__class__.__name__)

        self._core = pyrs_core

        return

