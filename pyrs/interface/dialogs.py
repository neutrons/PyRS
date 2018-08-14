try:
    from PyQt5.QtWidgets import QDialog
except ImportError:
    from PyQt4.QtGui import QDialog
    from PyQt4.QtCore import pyqtSignal

import gui_helper
from ui import ui_newsessiondialog


class CreateNewSessionDialog(QDialog):
    """ Create a new strain/stress session dialog
    """


    NewSessionSignal = pyqtSignal(str, bool, bool, name='signal')

    def __init__(self, parent):
        """
        initialization
        :param parent:
        """
        super(CreateNewSessionDialog, self).__init__(parent)

        self.ui = ui_newsessiondialog.Ui_Dialog()
        self.ui.setupUi(self)

        self.ui.buttonBox.accepted.connect(self.do_quit)
        self.ui.buttonBox.rejected.connect(self.do_quit)

        # connect the signal
        self.NewSessionSignal.connect(parent.new_strain_stress_session)

        # init widgets
        self.ui.comboBox_strainStressType.clear()


        return

    def do_new_session(self):
        """
        new session
        :return:
        """
        session_name = str(self.ui.lineEdit_sessionName.text()).strip()
        if len(session_name) == 0:
            gui_helper.pop_message(self, 'Session name must be specified', 'error')
            return

        ss_type = str(self.ui.comboBox_strainStressType.currentText()).lower()
        if ss_type


        # quit eventually
        self.do_quit()

        return

    def do_quit(self):

        self.close()

    def reset_dialog(self):
        """
        reset the dialog input from last use
        :return:
        """
        self.ui.lineEdit_sessionName.setText('')

        return

