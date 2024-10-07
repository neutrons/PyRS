from qtpy.QtWidgets import QHBoxLayout, QLabel, QWidget  # type:ignore
from qtpy.QtWidgets import QLineEdit, QPushButton  # type:ignore
from qtpy.QtWidgets import QFileDialog, QGroupBox  # type:ignore

from qtpy.QtWidgets import QGridLayout  # type:ignore
from qtpy.QtWidgets import QMainWindow  # type:ignore
from qtpy.QtCore import Qt  # type: ignore

from pyrs.interface.gui_helper import pop_message


class FileLoad(QWidget):
    def __init__(self, name=None, fileType="HidraProjectFile (*.h5);;All Files (*)", parent=None):
        self._parent = parent
        super().__init__(parent)
        self.name = name
        self.fileType = fileType
        layout = QHBoxLayout()
        if name == "Run Numbers:":
            label = QLabel(name)
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            layout.addWidget(label)
            self.lineEdit = QLineEdit()
            self.lineEdit.setReadOnly(False)
            self.lineEdit.setFixedWidth(300)

            layout.addWidget(self.lineEdit)

            self.browse_button = QPushButton("Load")
            self.browse_button.clicked.connect(self.loadRunNumbers)
        else:
            if name is None:
                self.browse_button = QPushButton("Browse Exp Data")
            else:
                self.browse_button = QPushButton("Browse")

            self.browse_button.clicked.connect(self.openFileDialog)

        layout.addWidget(self.browse_button)
        self.setLayout(layout)

    def _reset_fit_data(self):
        self._parent.fit_summary.fit_table_operator.fits = None
        self._parent.fit_summary.fit_table_operator.fit_result = None

    def openFileDialog(self):
        self._parent._project_files, _ = QFileDialog.getOpenFileNames(self,
                                                                      self.name,
                                                                      "",
                                                                      self.fileType,
                                                                      options=QFileDialog.DontUseNativeDialog)

        if self._parent._project_files:
            self._parent.load_project_files = self._parent._project_files

            self.load_project_files()

    def saveFileDialog(self, combined_files):
        if combined_files != 0:
            self._parent._project_files, _ = QFileDialog.getSaveFileName(self,
                                                                         'Save Combined Proeject File',
                                                                         "",
                                                                         self.fileType,
                                                                         options=QFileDialog.DontUseNativeDialog)

    def loadRunNumbers(self):
        self._parent._project_files = self._parent.controller.parse_entry_list(self.lineEdit.text())
        combined_files = self._parent.controller.load_combine_projects(self._parent._project_files)
        self.saveFileDialog(combined_files)

    def load_project_files(self):
        try:
            combined_files = self._parent.controller.load_combine_projects(self._parent._project_files)
            self.saveFileDialog(combined_files)

        except (FileNotFoundError, RuntimeError, ValueError) as run_err:
            pop_message(self, f'Failed to find run {self._parent._project_files}',
                        str(run_err), 'error')

            self._parent.load_project_files = None

    def setFilenamesText(self, filenames):
        self.lineEdit.setText(filenames)


class FileLoading(QGroupBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Load Project Files")
        layout = QHBoxLayout()

        self.file_load_run_number = FileLoad(name="Run Numbers:", parent=parent)
        self.file_load_dilg = FileLoad(name=None, parent=parent)

        layout.addWidget(self.file_load_run_number)
        layout.addWidget(self.file_load_dilg)

        self.setLayout(layout)

    def set_text_values(self, direction, text):
        getattr(self, f"file_load_e{direction}").setFilenamesText(text)


class CombineRunsViewer(QMainWindow):
    def __init__(self, combine_runs_model, combine_runs_ctrl, parent=None):

        self._model = combine_runs_model
        self._ctrl = combine_runs_ctrl
        self._nexus_file = None
        self._run_number = None
        self._calibration_input = None

        super().__init__(parent)

        self.setWindowTitle("PyRS Combine Projectfiles Window")

        self.fileLoading = FileLoading(self)

        self.window = QWidget()
        self.layout = QGridLayout()
        self.setCentralWidget(self.fileLoading)
        self.window.setLayout(self.layout)

    @property
    def controller(self):
        return self._ctrl

    @property
    def model(self):
        return self._model
