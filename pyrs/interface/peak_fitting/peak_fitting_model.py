"""
Model for the peak-fitting interface.

This is the "M" of the Model-View-Presenter structure shared with the
texture-fitting UI. It owns all data and business logic: loading Hidra project
files, running the peak-fit engine, and writing results back out to HDF5/CSV/JSON.

Unlike :class:`pyrs.interface.texture_fitting.texture_fitting_model.TextureFittingModel`,
which operates on a raw ``HidraWorkspace``, this model wraps a
:class:`pyrs.core.pyrscore.PyRsCore` instance and routes data through its
session-based API (``get_diffraction_data(session_name=...)``,
``get_peak_fitting_result(...)``, ``reduction_service.get_sample_logs_names(...)``).
The two models therefore share method *names* but not implementations; do not try to
unify them without also rewriting the peak-fitting data path.
"""

import json
import os
import traceback
from shutil import copyfile

from qtpy.QtCore import QObject, Signal  # type:ignore

from pyrs.core import MonoSetting  # type: ignore
from pyrs.core.summary_generator import SummaryGenerator
from pyrs.peaks import FitEngineFactory as PeakFitEngineFactory  # type: ignore
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore


class PeakFittingModel(QObject):
    """Data and business logic for the peak-fitting window.

    Attributes:
        failureMsg: Emitted with ``(title, message, traceback)`` when an
            operation fails, so the view can surface it without the model
            depending on any UI helper.
    """

    failureMsg = Signal(str, str, str)

    def __init__(self, peak_fit_core):
        """
        Args:
            peak_fit_core: A :class:`pyrs.core.pyrscore.PyRsCore` instance that
                performs the session-based data access and fitting.
        """
        super().__init__()
        self._core = peak_fit_core
        self._project_name = None
        self.hidra_workspace = None
        self.fit_result = None
        self._curr_file_name = None

    @property
    def working_dir(self):
        """Working directory of the underlying core (used for file dialogs)."""
        return self._core.working_dir

    @property
    def project_name(self):
        """Name of the currently loaded project/session."""
        return self._project_name

    @property
    def curr_file_name(self):
        """Path of the working project file currently loaded."""
        return self._curr_file_name

    def load_hidra_project(self, project_files):
        """Load one or more Hidra project files into a workspace.

        Args:
            project_files: A list of project-file paths. The first is loaded and
                any remaining files are appended.

        Returns:
            The loaded ``HidraWorkspace``.

        Raises:
            RuntimeError, TypeError: If the files cannot be loaded.
        """
        self._set_up_project_name(project_files)
        ws = self._load_multiple_file(project_files)

        self._curr_file_name = self._parse_working_files(project_files)
        self.hidra_workspace = ws
        self.fit_result = None

        return ws

    def _load_multiple_file(self, project_files):
        """Load the first project file and append the rest."""
        hidra_ws = self._core.load_hidra_project(
            project_files[0],
            project_name=self._project_name,
            load_detector_counts=False,
            load_diffraction=True,
        )

        for project in project_files[1:]:
            _project = HidraProjectFile(project)
            hidra_ws.append_hidra_project(_project)
            _project.close()

        return hidra_ws

    def _set_up_project_name(self, project_file):
        """Keep the basename and remove the nxs/h5 extensions."""
        if type(project_file) is list:
            self._project_name = "HB2B" + "".join(
                ["_{}".format(run.split(".")[0].split("_")[-1]) for run in project_file]
            )
        else:
            self._project_name = os.path.basename(project_file).split(".")[0]

    def _parse_working_files(self, project_file):
        """Keep the file path and append the runs being fitted."""
        if type(project_file) is list:
            if len(project_file) == 1:
                project_file = project_file[0]
            else:
                project_file = project_file[0].split("HB2B")[0] + "".join(
                    ["HB2B_{}".format(run.split(".")[0].split("_")[-1]) for run in project_file]
                )
                if project_file[-3:] != ".h5":
                    project_file += ".h5"

        return project_file

    def get_subruns_limit(self):
        """Return the list of sub-run numbers in the loaded workspace."""
        sample_log = self.hidra_workspace._sample_logs
        return list(sample_log._subruns)

    def get_sample_log_names(self):
        """Return the sample-log names available for the current session."""
        return self._core.reduction_service.get_sample_logs_names(self._project_name)

    def get_diffraction_data(self, sub_run, mask=None):
        """Return the reduced diffraction data ``[x, y]`` for a sub-run."""
        return self._core.get_diffraction_data(session_name=self._project_name, sub_run=sub_run, mask=mask)

    def get_peak_fitting_result(self, return_format=dict, effective_parameter=False):
        """Return the peak-fitting result for the current session, or ``None``."""
        if self._project_name is None:
            return None
        return self._core.get_peak_fitting_result(
            self._project_name, 0, return_format=return_format, effective_parameter=effective_parameter
        )

    def fit_diff_peaks(self, peak_tags, x_mins, x_maxs, peak_function_name, background_function_name):
        """Fit one or more peaks and store/return the result.

        Args:
            peak_tags: List of peak labels.
            x_mins: List of left bounds (one per peak).
            x_maxs: List of right bounds (one per peak).
            peak_function_name: Peak profile function (e.g. ``"PseudoVoigt"``).
            background_function_name: Background function (e.g. ``"Linear"``).

        Returns:
            The fit result object produced by the fit engine, or ``None`` if the
            fit failed (``failureMsg`` is emitted in that case).
        """
        try:
            wavelength = self.hidra_workspace.get_wavelength(True, True)
            fit_engine = PeakFitEngineFactory.getInstance(
                self.hidra_workspace, peak_function_name, background_function_name, wavelength=wavelength
            )
            fit_result = fit_engine.fit_multiple_peaks(peak_tags, x_mins, x_maxs)
        except Exception as e:  # noqa: BLE001
            self.failureMsg.emit("Failed to fit peaks", str(e), traceback.format_exc())
            return None

        self.fit_result = fit_result
        return fit_result

    def save_fit_result(self, out_file_name=""):
        """Write the current fit result to an HDF5 project file.

        If ``out_file_name`` differs from the currently loaded file, the loaded
        file is first copied to ``out_file_name`` and the peak parameters are
        written to the copy; otherwise the loaded file is updated in place.

        Args:
            out_file_name: Destination path (or a single-element list).
        """
        fit_result = self.fit_result
        if fit_result is None:
            return

        if type(out_file_name) is list:
            out_file_name = out_file_name[0]

        if out_file_name is not None and self._curr_file_name != out_file_name:
            copyfile(self._curr_file_name, out_file_name)
            current_project_file = out_file_name
        else:
            current_project_file = self._curr_file_name

        project_h5_file = HidraProjectFile(current_project_file, mode=HidraProjectFileMode.READWRITE)
        for peak in fit_result.peakcollections:
            project_h5_file.write_peak_parameters(peak)
        project_h5_file.save(False)
        project_h5_file.close()

    def export_csv(self, csv_file_name):
        """Write the fit result and sample logs to a CSV summary file.

        Args:
            csv_file_name: Destination CSV path.

        Raises:
            AttributeError: If no fit result is available yet.
        """
        peaks = self.fit_result.peakcollections
        sample_logs = self.hidra_workspace._sample_logs
        monosetting = MonoSetting.getFromRotation(self.hidra_workspace.get_sample_log_value("mrot", 1))

        generator = SummaryGenerator(csv_file_name, log_list=sample_logs.keys())
        generator.setHeaderInformation(
            {
                "cal_wavelength": self.hidra_workspace.get_wavelength(False, False),
                "mono_wavelength": monosetting.value,
                "mono_setting": monosetting.name,
                "cal_file": self.hidra_workspace.calibration_file,
            }
        )
        generator.write_csv(sample_logs, peaks)

    def save_peak_range_json(self, filename, list_peak_ranges, list_peak_labels, list_peak_d0):
        """Serialize the peak-range table to a JSON file."""
        output = {}
        for _index, peak_range in enumerate(list_peak_ranges):
            output[_index] = {
                "peak_range": peak_range,
                "peak_label": list_peak_labels[_index],
                "d0": list_peak_d0,
            }

        with open(filename, "w") as outfile:
            json.dump(output, outfile)

    def load_peak_range_json(self, filename):
        """Deserialize a peak-range JSON file.

        Returns:
            A tuple ``(peak_range, peak_label, peak_d0)`` of parallel lists.
        """
        with open(filename) as json_file:
            data = json.load(json_file)

        peak_range = []
        peak_label = []
        peak_d0 = []
        for _index in data.keys():
            peak_range.append(data[_index]["peak_range"])
            peak_label.append(data[_index]["peak_label"])
            peak_d0.append(data[_index]["d0"])

        return peak_range, peak_label, peak_d0

    def save_nexus(self, data_key, file_name):
        """Save data to a Mantid-compatible NeXus file."""
        self._core.save_nexus(data_key, file_name)
