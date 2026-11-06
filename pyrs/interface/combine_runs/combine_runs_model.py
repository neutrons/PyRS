from qtpy.QtCore import Signal, QObject  # type:ignore
from pyrs.core.workspaces import HidraWorkspace
from pyrs.calibration.provenance import file_record
from pyrs.projectfile import HidraProjectFile  # type: ignore
from pyrs.projectfile.file_object import combine_reduction_records


class CombineRunsModel(QObject):
    propertyUpdated = Signal(str)
    failureMsg = Signal(str, str, str)

    def __init__(self):
        super().__init__()
        self._hidra_ws = None
        self._record = None  # the combined files' calibration record, for the exported file (#1063)
        self.warnings = []  # e.g. files whose calibration could not be checked

    def combine_project_files(self, project_files, accept_reduction_differences=False):
        """Combine reduced project files, refusing files reduced with different calibrations (#1063).

        The files' calibration records are checked before anything is loaded: different calibrations
        raise :class:`CalibrationMismatchError`, and different vanadium runs, masks or binning (#1068)
        raise :class:`ReductionMismatchError` unless ``accept_reduction_differences`` (the user approved
        them); either way nothing is combined. Historic files without a record are combined, and
        :attr:`warnings` names them.
        """
        sources = []
        for file_name in project_files:
            project = HidraProjectFile(file_name)
            source = file_record(file_name)
            source["record"] = project.read_reduction_provenance()
            project.close()
            sources.append(source)
        # raises before anything is combined
        record = combine_reduction_records(sources, accept_reduction_differences=accept_reduction_differences)

        self._hidra_ws = HidraWorkspace("Combined Project Files")
        _project = HidraProjectFile(project_files[0])
        self._hidra_ws.load_hidra_project(_project, load_raw_counts=False, load_reduced_diffraction=True)
        _project.close()

        for project in project_files[1:]:
            _project = HidraProjectFile(project)
            self._hidra_ws.append_hidra_project(_project)
            _project.close()

        self._record = record
        self.warnings = [record["warning"]] if record.get("warning") else []

    def export_project_files(self, fileout):
        export_project = HidraProjectFile(fileout, "w")
        self._hidra_ws.save_experimental_data(
            export_project, sub_runs=self._hidra_ws._sample_logs.subruns, ignore_raw_counts=True
        )
        self._hidra_ws.save_reduced_diffraction_data(export_project)
        # which calibration(s) the combined data were reduced with, and from which files (#1063)
        if self._record is not None:
            export_project.write_reduction_provenance(applied_calibration=self._record)
        export_project.save()
