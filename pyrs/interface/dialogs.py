try:
    from PyQt5.QtWidgets import QDialog
except ImportError:
    from PyQt4.QtGui import QDialog

from ui import ui_newsessiondialog


class CreateNewSessionDialog(QDialog):
    """

    """
    def __init__(self, parent):

        super(CreateNewSessionDialog, self).__init__(parent)

        self.ui = ui_newsessiondialog.Ui_Dialog()
        self.ui.setupUi(self)

        self.ui.buttonBox.accepted.connect(self.do_quit)
        self.ui.buttonBox.rejected.connect(self.do_quit)

        return

    def do_quit(self):

        self.close()

