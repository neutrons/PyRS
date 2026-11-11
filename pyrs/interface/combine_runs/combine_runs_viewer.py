from qtpy.QtWidgets import QHBoxLayout, QLabel, QWidget  # type:ignore
from qtpy.QtWidgets import QLineEdit, QPushButton  # type:ignore
from qtpy.QtWidgets import QFileDialog, QGroupBox, QMessageBox  # type:ignore

from qtpy.QtWidgets import QGridLayout  # type:ignore
from qtpy.QtWidgets import QMainWindow  # type:ignore
from qtpy.QtCore import Qt  # type: ignore

from pyrs.interface.gui_helper import pop_message, report_save_failure, resolve_save_target, write_nxstress_or_report
from pyrs.projectfile.file_object import CalibrationMismatchError, ReductionMismatchError
from pyrs.utilities.config import Config
# from pyrs.interface.gui_helper import browse_file


class FileLoad(QWidget):
    def __init__(self, name=None, fileType="HidraProjectFile (*.h5);;All Files (*)", parent=None):
        self._parent = parent
        super().__init__(parent)
        self.name = name
        self.fileType = fileType
        layout = QHBoxLayout()
        self._auto_prompt_export = True  # Defined variable to help with pytest catching multiple file dialogs
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

    def openFileDialog(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, self.name, "", self.fileType, options=QFileDialog.DontUseNativeDialog
        )
        if not files:
            return  # cancelled: the dialog returns an empty list, not None

        self._parent._project_files = files
        self.load_project_files()

    def saveFileDialog(self) -> None:
        if self._parent._project_files is not None:
            _export_project, _ = QFileDialog.getSaveFileName(
                self, "Save Combined Proeject File", "", self.fileType, options=QFileDialog.DontUseNativeDialog
            )

            # An empty string means the dialog was cancelled.
            if not _export_project:
                return
            target = resolve_save_target(
                self, _export_project, extension=Config["legacy_io.extension"], format_name="HidraProjectFile"
            )
            if target is None:
                return
            # Reported rather than raised: an exception escaping a Qt slot aborts
            # PyRS (see docs/ground_truths.md). `OSError` is what `h5py` raises for
            # a destination it cannot create.
            try:
                self._parent.controller.export_combined_projectfile(target)
            except (OSError, RuntimeError, ValueError) as export_err:
                report_save_failure(self, target, export_err)

    def saveFileDialogNXstress(self) -> None:
        if self._parent._project_files is not None:
            _export_project, _ = QFileDialog.getSaveFileName(
                self,
                "Save Combined Proeject File (NXstress)",
                "",
                "NXstress (*.nxs)",
                options=QFileDialog.DontUseNativeDialog,
            )

            # Resolves the extension, writes to a temporary file that replaces the
            # destination only on success, and reports any failure.
            if _export_project:
                write_nxstress_or_report(self, _export_project, self._parent.controller.export_combined_projectfile)

    def loadRunNumbers(self) -> None:
        try:
            self._parent._project_files = self._parent.controller.parse_entry_list(self.lineEdit.text())
        except (FileNotFoundError, RuntimeError, ValueError) as run_err:
            # e.g. a run that has not been reduced yet (#931)
            pop_message(self, f"Failed to find run {self.lineEdit.text()}", str(run_err), "error")
            return
        # the same checks and messages as the file dialog (this used to call saveFileDialog with an
        # argument it does not take, so a failure was never shown)
        self.load_project_files()

    def load_project_files(self) -> None:
        """Combine the chosen files: different calibrations are refused, different vanadium runs, masks or
        binning need the user's approval, and historic files without a record are flagged (#1063, #1068)."""
        files = self._parent._project_files
        try:
            try:
                combined = self._parent.controller.load_combine_projects(files)
            except ReductionMismatchError as differences:
                answer = QMessageBox.question(
                    self, "Different reductions", str(differences), QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                )
                if answer != QMessageBox.Yes:
                    self._parent._project_files = None
                    return
                combined = self._parent.controller.load_combine_projects(files, accept_reduction_differences=True)
        except CalibrationMismatchError as mismatch:
            pop_message(self, "Runs reduced with different calibrations", str(mismatch), "error")
            self._parent._project_files = None
            return
        # OSError: a project file that has gone missing or isn't HDF5; KeyError: an HDF5 file that isn't a
        # project file. Uncaught, either escapes the Qt slot and can close PyRS (#931)
        except (OSError, KeyError, RuntimeError, ValueError) as run_err:
            pop_message(self, f"Failed to load {files}", str(run_err), "error")
            self._parent._project_files = None
            return
        if not combined:
            # nothing was combined: exporting now would write an earlier combination, or crash (#931)
            pop_message(self, "Select at least two runs to combine", f"Only {len(files or [])} chosen", "error")
            self._parent._project_files = None
            return
        for warning in self._parent.model.warnings:
            pop_message(self, "Reduction not fully checked", warning, "warning")
        if not self._auto_prompt_export:
            return
        # Two sequential auto-prompts, each gated by its own format's config
        # flag -- one dialog completes, then (if enabled) the other pops up.
        # No persistent Export button exists in this viewer to grey out, so
        # "enablement" is realized as "does this dialog get offered at all".
        # Each export reports its own failure, so neither raises here.
        if Config["legacy_io.enable"]:
            self.saveFileDialog()
        # NXstress additionally requires the merged workspace to have
        # instrument geometry -- combine_project_files loads with
        # load_raw_counts=False, and (see HidraWorkspace.load_hidra_project)
        # geometry may genuinely be absent from an older/legacy source file;
        # in that case NXstress export is silently skipped here (analogous to
        # a disabled/greyed-out action -- HidraWorkspace.load_hidra_project
        # already logs a warning at load time explaining why).
        if Config["nxstress.enable"] and self._parent.model._hidra_ws.get_instrument_setup() is not None:
            self.saveFileDialogNXstress()

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


class CombineRunsViewer(QMainWindow):
    def __init__(self, combine_runs_model, combine_runs_ctrl, parent=None):
        self._model = combine_runs_model
        self._ctrl = combine_runs_ctrl
        self._run_number = None
        self._project_files = None

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
