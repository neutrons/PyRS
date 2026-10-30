import os
import time
import warnings
from typing import List, Optional, Tuple

import numpy as np
from qtpy.QtCore import Signal, QObject  # type:ignore
from pyrs.calibration.mantid_peakfit_calibration import FitCalibration
from pyrs.calibration.pixel_calibration_plots import PredictedRing, detector_two_theta_map, predicted_rings
from pyrs.calibration.pixel_pos_calibration import PixelCalibration, extract_ring_points
from pyrs.calibration.powder_standards import standard_for_sy
from pyrs.core.instrument_geometry import DENEXDetectorShift
from pyrs.utilities.calibration_file_io import read_calibration_json_file, write_calibration_to_json

# calibration methods (#1045): FitCalibration re-reduces and re-fits for every trial geometry; the
# ring-based calibration fits each ring once in detector strips, then refines from the ring points
METHOD_FITCALIBRATION = "FitCalibration"
METHOD_RING = "Ring-based"
CALIBRATION_METHODS = (METHOD_FITCALIBRATION, METHOD_RING)
DEFAULT_STRIPS = 5  # as the pyrs-pixel-calibration driver, and every recorded ring-based result

# recipe routine -> stage method; both calibrations have these methods. "iterate" is ring-based only:
# repeated Shift -> Rotation -> wavelength + 2theta_0 passes, then a joint refinement
RECIPE_STAGES = {
    "wavelength": "calibrate_wave_length",
    "wavelength_tth0": "calibrate_wave_shift",
    "rotations": "CalibrateRotation",
    "geometry": "CalibrateGeometry",
    "shifts": "CalibrateShift",
    "shift x": "calibrate_shiftx",
    "shift y": "calibrate_shifty",
    "distance": "calibrate_distance",
    "full": "FullCalibration",
    "tth0": "calibrate_tth0",
}
ITERATE_ROUTINE = "iterate"
ITERATE_PASSES = 3

# Export: the user's folder always; with the shared flag also the HB2B calibration that reduction
# picks up (HB2B_Latest.json), its per-monochromator copy and a timestamped archive copy
SHARED_CALIBRATION_DIR = "/HFIR/HB2B/shared/CALIBRATION"
CALIBRATION_TIME_FORMAT = "%Y-%m-%dT%H:%M"  # as FitCalibration's archive names

# geometries the 2D view can draw predicted rings for (#1039)
RING_GEOMETRY_NOMINAL = "Nominal"
RING_GEOMETRY_CURRENT = "Current calibration"
RING_GEOMETRY_FILE = "From file..."
RING_GEOMETRIES = (RING_GEOMETRY_NOMINAL, RING_GEOMETRY_CURRENT, RING_GEOMETRY_FILE)

# detector mask applied by default; the calibration runs unmasked when the file is not available
DEFAULT_MASK_FILE = "/HFIR/HB2B/shared/CALIBRATION/HB2B_Mask_Latest.xml"


def default_mask_file() -> Optional[str]:
    """The production detector mask if it exists, otherwise None (no mask)."""
    return DEFAULT_MASK_FILE if os.path.isfile(DEFAULT_MASK_FILE) else None


class DetectorCalibrationModel(QObject):
    propertyUpdated = Signal(str)
    failureMsg = Signal(str, str, str)

    def __init__(self, peak_fit_core):
        super().__init__()
        self._peak_fit = peak_fit_core
        self._hidra_ws = None
        self.peak_fit_engine = None
        self._run_number = None
        self._calibration_obj = None
        self._nexus_file = None
        self._mask_file: Optional[str] = None

        self.detector_params = [0, 0, 0, 0, 0, 0, 0, 0]
        # self.sub_runs = np.array([1])

        self._calibration_obj = None

        # ring-based calibration (#1045); FitCalibration's object still holds the converted run
        self.method = METHOD_FITCALIBRATION
        self._ring_calibration: Optional[PixelCalibration] = None
        self._ring_start: Optional[np.ndarray] = None
        self._ring_history: List[np.ndarray] = []
        self._ring_refinement = {"method": "trf", "max_nfev": None}
        self._keep_list: Optional[np.ndarray] = None
        self.calibration_warnings: List[str] = []
        self.ring_summary = ""

    @property
    def monochromator(self) -> str:
        """The loaded run's monochromator setting, e.g. ``"Si422"`` ("SiXXX" before a run is loaded)."""
        return "SiXXX" if self._calibration_obj is None else self._calibration_obj.monosetting.name

    def set_method(self, method: str) -> None:
        """Choose the calibration method, one of :data:`CALIBRATION_METHODS`."""
        if method not in CALIBRATION_METHODS:
            raise ValueError(f"method must be one of {CALIBRATION_METHODS}, got {method!r}")
        self.method = method

    @property
    def _active(self):
        """The calibration object of the chosen method (None before a run is loaded or rings extracted)."""
        return self._ring_calibration if self.method == METHOD_RING else self._calibration_obj

    @property
    def current_calibration_array(self) -> Optional[np.ndarray]:
        """Calibration vector of the chosen method: refined, or its starting values."""
        if self.method == METHOD_RING:
            if self._ring_calibration is not None:
                return self._ring_calibration.calibration_array
            return None if self._ring_start is None else self._ring_start.copy()
        return None if self._calibration_obj is None else np.asarray(self._calibration_obj.calibration_array)

    @property
    def nexus_file(self):
        return self._nexus_file

    @property
    def mask_file(self) -> Optional[str]:
        """Detector mask used by the calibration (None: no mask)."""
        return self._mask_file

    @property
    def sub_runs(self):
        return self._calibration_obj._hidra_ws.get_sub_runs()

    @property
    def powders(self):
        return self._calibration_obj.powders

    @property
    def get_wavelength(self):
        return self._calibration_obj._hidra_ws.get_wavelength(False, False)

    @property
    def reduction_masks(self):
        return self._calibration_obj._hidra_ws.reduction_masks

    @property
    def rmse(self):
        if self.method == METHOD_RING:
            stages = self._ring_calibration.residual_rmse if self._ring_calibration is not None else []
            return stages[-1] if stages else None
        if self._calibration_obj is not None:
            try:
                return self._calibration_obj.residual_rmse
            except IndexError:  # nothing fitted yet, e.g. right after switching method
                return None
        else:
            return None

    @property
    def r_sum(self):
        if self.method == METHOD_RING:
            stages = self._ring_calibration.residual_sum if self._ring_calibration is not None else []
            return stages[-1] if stages else None
        if self._calibration_obj is not None:
            try:
                return self._calibration_obj.residual_sum
            except IndexError:  # nothing fitted yet, e.g. right after switching method
                return None
        else:
            return None

    def export_calibration(self, folder: str, write_shared: bool = True) -> List[str]:
        """Write the chosen method's calibration to ``folder``, and optionally to the shared HB2B folder.

        The calibration JSON (reduction's own writer, TTH_0 included) is written as
        ``<folder>/HB2B_CAL_<time>.json``. With ``write_shared`` it is also written to
        :data:`SHARED_CALIBRATION_DIR` as ``HB2B_Latest.json`` (the calibration reduction picks up),
        ``HB2B_Latest_<Si...>.json`` and ``<Si...>/HB2B_CAL_<time>.json``, ``<Si...>`` being the run's
        monochromator setting.

        Args:
            folder: Existing folder for the user's copy.
            write_shared: Also write the shared copies.

        Returns:
            The files written.

        Raises:
            RuntimeError: If there is no calibration to export.
            FileNotFoundError: If ``folder`` does not exist.
            PermissionError: If the shared folder is not writable (the user's copy is still written).
        """
        values, errors, status = self._export_values()
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"folder {folder} does not exist")
        stamp = time.strftime(CALIBRATION_TIME_FORMAT, time.localtime())
        mono = self.monochromator
        targets = [os.path.join(folder, f"HB2B_CAL_{stamp}.json")]
        if write_shared:
            archive = os.path.join(SHARED_CALIBRATION_DIR, mono)
            if not (
                os.access(SHARED_CALIBRATION_DIR, os.W_OK)
                and (not os.path.exists(archive) or os.access(archive, os.W_OK))
            ):
                self._write(values, errors, status, targets[0])
                raise PermissionError(
                    f"cannot write to {SHARED_CALIBRATION_DIR}; the calibration was written to {targets[0]} only"
                )
            os.makedirs(archive, exist_ok=True)
            targets += [
                os.path.join(SHARED_CALIBRATION_DIR, "HB2B_Latest.json"),
                os.path.join(SHARED_CALIBRATION_DIR, f"HB2B_Latest_{mono}.json"),
                os.path.join(archive, f"HB2B_CAL_{stamp}.json"),
            ]
        for target in targets:
            self._write(values, errors, status, target)
        return targets

    def _export_values(self) -> Tuple[np.ndarray, np.ndarray, int]:
        """Values, errors and status of the chosen method's calibration."""
        if self.method == METHOD_RING:
            calibration = self._ring_calibration
            if calibration is None or calibration.calibration_status < 0:
                raise RuntimeError("nothing has been refined with the ring-based calibration yet")
            return calibration.calibration_array, calibration.calibration_error_array, calibration.calibration_status
        if self._calibration_obj is None:
            raise RuntimeError("no calibration run is loaded")
        fit = self._calibration_obj
        return np.asarray(fit.calibration_array), np.asarray(fit.calibration_error_array), int(fit._calibstatus)

    @staticmethod
    def _write(values: np.ndarray, errors: np.ndarray, status: int, file_name: str) -> None:
        write_calibration_to_json(
            DENEXDetectorShift(*values[:7]),
            DENEXDetectorShift(*errors[:7]),
            float(values[7]),
            float(errors[7]),
            int(status),
            file_name,
        )

    def set_refinement_params(self, method, max_nfev):
        if self.method == METHOD_RING:
            self._ring_refinement = {"method": method, "max_nfev": int(max_nfev)}
            if self._ring_calibration is not None:
                self._ring_calibration._method = method
                self._ring_calibration._max_nfev = int(max_nfev)
            return
        self._calibration_obj.set_refinement_method(method)
        self._calibration_obj.set_max_nfev(int(max_nfev))

    def set_calibration_params(self, params_list):
        if self.method == METHOD_RING:
            self._ring_start = np.asarray(params_list, dtype=float).copy()
            if self._ring_calibration is not None:
                self._ring_calibration.set_calibration_array(self._ring_start)
            return
        self._calibration_obj.set_calibration_array(params_list)
        self._calibration_obj.set_inst_shifts_wl(params_list)

    def extract_rings(self, strips: int = DEFAULT_STRIPS) -> Tuple[int, int, float, float]:
        """Ring-based "Fit Peaks": find the ring points in ``strips`` strips per sub-run.

        Uses the loaded run (its default detector mask applies) and the starting calibration
        (:meth:`set_calibration_params`, default: nominal geometry at the mono wavelength) to place
        the rings, exactly as ``pyrs-pixel-calibration`` does. Replaces any earlier ring calibration.

        Returns:
            Number of ring points, number of rings, and the RMS and sum of squared 2theta residuals
            (degrees, degrees^2) at the starting calibration.

        Raises:
            RuntimeError: If no run is loaded.
            ValueError: If ``strips`` is invalid or no ring point is found.
        """
        if self._calibration_obj is None:
            raise RuntimeError("load a calibration run first")
        strips = int(strips)
        if strips < 1:
            raise ValueError(f"strips must be at least 1, got {strips}")
        workspace = self._calibration_obj._hidra_ws
        start = self._ring_start
        if start is None:
            start = np.zeros(8)
            start[7] = float(workspace.get_wavelength(False, False))
        shift = None if not np.any(start[:7]) else DENEXDetectorShift(*start[:7])
        sub_runs = None
        if self._keep_list is not None:
            sub_runs = [int(s) for s, keep in zip(workspace.get_sub_runs(), self._keep_list) if keep]
        points = extract_ring_points(workspace, float(start[7]), calibration=shift, strips=strips, sub_runs=sub_runs)
        calibration = PixelCalibration(
            points, workspace.get_instrument_setup(), wavelength=float(start[7]), **self._ring_refinement
        )
        calibration.set_calibration_array(start)
        self._ring_calibration = calibration
        self._ring_history = [calibration.calibration_array]
        residual = calibration.residual(start)
        rings = len(set(zip(points.sub_run.tolist(), points.hkl_squared.tolist())))
        self.ring_summary = f"{len(points)} ring points on {rings} rings ({strips} strips per sub-run)"
        return len(points), rings, float(np.sqrt(np.mean(residual**2))), float(np.sum(residual**2))

    def set_reduction_param(self, tth_bins, eta_bins):
        if self._calibration_obj is not None:
            self._calibration_obj = FitCalibration(
                nexus_file=self._nexus_file, mask_file=self._mask_file, eta_slice=eta_bins, bins=tth_bins
            )

    def _init_calibration(self, nexus_file, tth=512, eta=3.0):
        self._mask_file = default_mask_file()
        self._calibration_obj = FitCalibration(
            nexus_file=nexus_file, mask_file=self._mask_file, eta_slice=eta, bins=tth
        )
        self._nexus_file = nexus_file

    def fit_diffraction_peaks(self):
        if self._calibration_obj is not None:
            residual = self._calibration_obj.fit_peaks()
            return np.sqrt((residual**2).sum() / residual.shape[0]), residual.sum()

    def get_calibration_values(self, x_item, y_item):
        if self.method == METHOD_RING:
            history = np.array(self._ring_history).T  # (8, steps + 1): the start, then each step
            if history.size == 0:
                return np.array([]), np.array([])
            _x = np.arange(history.shape[1] - 1) if x_item == 0 else history[x_item - 1, 1:]
            return _x, history[y_item, 1:]
        if x_item == 0:
            _x = np.arange(self._calibration_obj._calibration.shape[1] - 1)
        else:
            _x = self._calibration_obj._calibration[x_item - 1, 1:]

        return _x, self._calibration_obj._calibration[y_item, 1:]

    def calibrate_detector(self, fit_recipe):
        if self.method == METHOD_RING:
            return self._calibrate_rings(fit_recipe)
        if self._calibration_obj is not None:
            # self._calibration_obj.initalize_calib_results()
            calibration = []
            calibration_error = []
            for recipe in fit_recipe:
                if recipe == "wavelength":
                    self._calibration_obj.calibrate_wave_length()
                elif recipe == "rotations":
                    self._calibration_obj.CalibrateRotation()
                elif recipe == "geometry":
                    self._calibration_obj.CalibrateGeometry()
                elif recipe == "shifts":
                    self._calibration_obj.CalibrateShift()
                elif recipe == "shift x":
                    self._calibration_obj.calibrate_shiftx()
                elif recipe == "shift y":
                    self._calibration_obj.calibrate_shifty()
                elif recipe == "distance":
                    self._calibration_obj.calibrate_distance()
                elif recipe == "full":
                    self._calibration_obj.FullCalibration()
                elif recipe == "wavelength_tth0":
                    self._calibration_obj.calibrate_wave_shift()
                elif recipe == "tth0":
                    self._calibration_obj.calibrate_tth0()

                if recipe != "":
                    calibration.append(np.copy(self._calibration_obj.calibration_array))
                    calibration_error.append(np.copy(self._calibration_obj.calibration_error_array))

            try:
                return (
                    calibration,
                    calibration_error,
                    self._calibration_obj.residual_sum,
                    self._calibration_obj.residual_rmse,
                )
            except IndexError:
                return calibration, -1, -1, -1

    def _calibrate_rings(self, fit_recipe):
        """Run the recipe with the ring-based calibration, then one error estimate with 2theta_0 held.

        Each routine is the stage of the same name (:data:`RECIPE_STAGES`), or "iterate". Warnings
        (e.g. a parameter stopped at a bound) are kept in :attr:`calibration_warnings`. Returns the
        same ``(values per step, errors per step, residual sum, RMS)`` as FitCalibration's recipe.
        """
        calibration = self._ring_calibration
        if calibration is None:
            raise RuntimeError("extract the rings first (Fit Peaks)")
        unknown = [r for r in fit_recipe if r and r != ITERATE_ROUTINE and r not in RECIPE_STAGES]
        if unknown:
            raise ValueError(f"unknown recipe routines {unknown}")
        values, errors = [], []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for routine in fit_recipe:
                if not routine:
                    continue
                if routine == ITERATE_ROUTINE:
                    calibration.iterate(passes=ITERATE_PASSES)
                else:
                    getattr(calibration, RECIPE_STAGES[routine])()
                values.append(calibration.calibration_array)
                errors.append(calibration.calibration_error_array)
                self._ring_history.append(calibration.calibration_array)
            if values:
                # once, after the last step, as the driver does: the result exported is this one
                calibration.estimate_errors()
                values[-1], errors[-1] = calibration.calibration_array, calibration.calibration_error_array
                self._ring_history[-1] = calibration.calibration_array
        self.calibration_warnings = [str(w.message) for w in caught]
        return values, errors, self.r_sum, self.rmse

    def set_keep_subrun_list(self, keep_list):
        self._keep_list = None if keep_list is None else np.asarray(keep_list, dtype=bool)
        if self._calibration_obj is not None:
            self._calibration_obj.set_keep_subrun_list(keep_list)

    def get_reduced_diffraction_data(self, sub_run, mask):
        return self._calibration_obj.reducer.get_diffraction_data(sub_run, mask_id=mask)

    def get_fitted_diffraction_data(self, sub_run):
        if self._calibration_obj.fitted_ws is not None:
            return self._calibration_obj.fitted_ws[sub_run - 1]

    def ring_overlay(
        self, sub_run: int, geometry: str, calibration_file: Optional[str] = None
    ) -> Optional[Tuple[np.ndarray, List[PredictedRing], Optional[np.ndarray]]]:
        """Pixel 2theta map, predicted rings and detector mask for the 2D view's ring overlay.

        The rings are the Bragg angles of the sub-run's powder standard (matched by ``sy``) at the
        geometry's wavelength, mapped onto the detector with that geometry; nothing is fitted.

        Args:
            sub_run: Sub-run shown.
            geometry: One of :data:`RING_GEOMETRIES`: the nominal geometry at the run's mono
                wavelength, the calibration currently in the window, or ``calibration_file``.
            calibration_file: Calibration JSON, for :data:`RING_GEOMETRY_FILE`.

        Returns:
            ``(two_theta_map, rings, mask)``, or None if no run is loaded or the sub-run's ``sy``
            matches no powder standard.

        Raises:
            ValueError: On an unknown geometry, or :data:`RING_GEOMETRY_FILE` without a file.
            RuntimeError: If the calibration file cannot be read.
        """
        if geometry not in RING_GEOMETRIES:
            raise ValueError(f"geometry must be one of {RING_GEOMETRIES}, got {geometry!r}")
        if self._calibration_obj is None:
            return None
        workspace = self._calibration_obj._hidra_ws

        shift: Optional[DENEXDetectorShift]
        if geometry == RING_GEOMETRY_NOMINAL:
            shift, wavelength = None, float(workspace.get_wavelength(False, False))
        elif geometry == RING_GEOMETRY_CURRENT:
            current = self.current_calibration_array
            if current is None:
                return None
            values = np.asarray(current, dtype=float)
            shift, wavelength = DENEXDetectorShift(*values[:7]), float(values[7])
        else:
            if not calibration_file:
                raise ValueError("choose a calibration file to draw its rings")
            shift, _, file_wavelength, _, _ = read_calibration_json_file(calibration_file)
            wavelength = float(file_wavelength)

        standard = standard_for_sy(float(workspace.get_sample_log_value("sy", sub_run)))
        if standard is None:
            return None
        two_theta = detector_two_theta_map(
            workspace.get_instrument_setup(), workspace.get_detector_2theta(sub_run), workspace.get_l2(sub_run), shift
        )
        mask = workspace.get_detector_mask(True)
        visible = two_theta if mask is None else two_theta.ravel()[np.asarray(mask).ravel() != 0]
        rings = predicted_rings(standard, wavelength, (float(np.min(visible)), float(np.max(visible))))
        return two_theta, rings, mask

    def get_2D_diffraction_counts(self, sub_run):
        if self._nexus_file is not None:
            return self._calibration_obj._hidra_ws.get_detector_counts(sub_run)
