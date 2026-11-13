"""Calibration diagnostics in the NeXus calibration file (#1054 phase 2): covariance, ring points, strip fits"""

import json
import warnings

import h5py
import numpy as np
import pytest

from pyrs.calibration import pixel_refinement
from pyrs.calibration.pixel_pos_calibration import PixelCalibration, calibrate_rings
from pyrs.calibration.mantid_peakfit_calibration import FitCalibration
from pyrs.calibration.powder_standards import PowderStandard
from pyrs.calibration.pixel_refinement import UNDETERMINED, _parameter_errors, _parameter_errors_and_covariance
from pyrs.utilities.calibration_file_io import read_calibration_file
from pyrs.utilities.calibration_nexus import read_calibration_nexus_diagnostics, write_calibration_nexus
from tests.util.instrument_helpers import default_hb2b_geometry
from tests.util.ring_calibration_helpers import synthetic_calibration_workspace, synthetic_ring_points

TRUE_PARAMS = np.array([0.004, -0.003, 0.012, 0.3, -0.5, 1.2, 0.656, 1.5308])


def _old_parameter_errors(jacobian, residual, dof):
    """_parameter_errors as it was before the covariance was split out (#1054), for the bit-identity check"""
    scale = np.linalg.norm(jacobian, axis=0)
    if dof <= 0 or not np.all(np.isfinite(jacobian)) or np.any(scale == 0):
        return np.full(jacobian.shape[1], UNDETERMINED)
    scaled = jacobian / scale

    def rank(matrix):
        singular = np.linalg.svd(matrix, compute_uv=False)
        return int(np.sum(singular > pixel_refinement._RANK_RTOL * singular[0]))

    full_rank = rank(scaled)
    undetermined = np.zeros(scaled.shape[1], dtype=bool)
    if full_rank < scaled.shape[1]:
        for column in range(scaled.shape[1]):
            if rank(np.delete(scaled, column, axis=1)) == full_rank:
                undetermined[column] = True
    covariance = np.linalg.pinv(scaled.T @ scaled, rcond=pixel_refinement._RANK_RTOL**2, hermitian=True)
    variances = np.diag(covariance) / scale**2 * np.sum(residual**2) / dof
    return np.where(undetermined | ~(variances >= 0), UNDETERMINED, np.sqrt(np.abs(variances)))


def _jacobians():
    rng = np.random.default_rng(7)
    full = rng.normal(size=(60, 8)) * np.array([1e3, 1e3, 1e3, 1, 1, 1, 1, 1e4])
    degenerate = full.copy()
    degenerate[:, 6] = degenerate[:, 0] * 2.5 - degenerate[:, 4]  # an exact degeneracy, as Shift_x/Rot_y/TTH_0
    return {"full-rank": full, "degenerate": degenerate}


@pytest.mark.parametrize("kind", ["full-rank", "degenerate"])
def test_errors_unchanged_by_covariance_split(kind):
    """The written errors are bit for bit what they were before the covariance was kept"""
    # Arrange
    jacobian = _jacobians()[kind]
    residual = np.random.default_rng(3).normal(scale=0.01, size=jacobian.shape[0])

    # Act / Assert
    np.testing.assert_array_equal(
        _parameter_errors(jacobian, residual, residual.size - 8), _old_parameter_errors(jacobian, residual, 52)
    )


@pytest.mark.parametrize("kind", ["full-rank", "degenerate"])
def test_covariance_diagonal_is_the_errors(kind):
    # Arrange
    jacobian = _jacobians()[kind]
    residual = np.random.default_rng(3).normal(scale=0.01, size=jacobian.shape[0])

    # Act
    errors, covariance = _parameter_errors_and_covariance(jacobian, residual, residual.size - 8)

    # Assert - determined parameters: sqrt of the diagonal; undetermined: NaN rows and columns
    determined = errors != UNDETERMINED
    np.testing.assert_allclose(np.sqrt(np.diag(covariance)[determined]), errors[determined], rtol=1e-12)
    assert np.all(np.isnan(covariance[~determined, :])) and np.all(np.isnan(covariance[:, ~determined]))
    if kind == "degenerate":
        assert not determined.all()


def test_no_covariance_without_degrees_of_freedom():
    # Act
    errors, covariance = _parameter_errors_and_covariance(np.ones((3, 3)), np.zeros(3), 0)

    # Assert
    assert covariance is None and np.all(errors == UNDETERMINED)


def test_no_ring_covariance_after_a_restart():
    """set_calibration_array starts again: the earlier stages' covariance no longer describes it (review)"""
    # Arrange
    calibration = PixelCalibration(
        synthetic_ring_points(default_hb2b_geometry(), TRUE_PARAMS), default_hb2b_geometry(), 1.54
    )
    calibration.FullCalibration()

    # Act
    calibration.set_calibration_array(TRUE_PARAMS)

    # Assert
    assert calibration.covariance is None


def test_ring_covariance_is_the_last_stage():
    # Arrange
    calibration = PixelCalibration(
        synthetic_ring_points(default_hb2b_geometry(), TRUE_PARAMS), default_hb2b_geometry(), 1.54
    )

    # Act
    calibration.FullCalibration()
    names, matrix = calibration.covariance

    # Assert
    assert names == calibration.stages[-1].free_parameters and matrix.shape == (8, 8)
    errors = calibration.stages[-1].errors
    determined = errors != UNDETERMINED
    np.testing.assert_allclose(np.sqrt(np.diag(matrix)[determined]), errors[determined], rtol=1e-12)


@pytest.fixture(scope="module")
def ring_with_fits():
    workspace = synthetic_calibration_workspace(default_hb2b_geometry(), TRUE_PARAMS)
    fits = []
    calibration = calibrate_rings(workspace, 1.54, max_rounds=2, diagnostics=fits)
    calibration.estimate_errors()
    return calibration, fits


def test_ring_diagnostics_round_trip(ring_with_fits, tmp_path):
    """The NeXus file holds the ring points, their residuals, the strip fits and the covariance"""
    # Arrange
    calibration, fits = ring_with_fits
    path = str(tmp_path / "ring.nxs")

    # Act
    calibration.write_calibration(path, strip_fits=fits)
    diagnostics = read_calibration_nexus_diagnostics(path)

    # Assert
    assert diagnostics["method"] == "ring"
    points = diagnostics["tables"]["ring_points"]
    np.testing.assert_array_equal(points["residual"], calibration.residual(calibration.calibration_array))
    np.testing.assert_array_equal(points["index_0"], calibration.ring_points.index_0)
    assert points["powder"] == list(calibration.ring_points.powder)
    strips = diagnostics["tables"]["strip_fits"]
    assert len(strips["reason"]) == len(fits) and list(strips["accepted"]) == [fit.accepted for fit in fits]
    names, matrix = diagnostics["covariance"]
    expected_names, expected = calibration.covariance
    assert names == expected_names
    np.testing.assert_array_equal(matrix, expected)


def test_nexus_and_json_hold_the_same_calibration(ring_with_fits, tmp_path):
    # Arrange
    calibration, _ = ring_with_fits
    json_file, nexus_file = str(tmp_path / "cal.json"), str(tmp_path / "cal.nxs")

    # Act
    calibration.write_calibration(json_file)
    calibration.write_calibration(nexus_file)

    # Assert
    from_json, from_nexus = read_calibration_file(json_file), read_calibration_file(nexus_file)
    assert from_json[0].convert_to_dict() == from_nexus[0].convert_to_dict()
    assert from_json[1].convert_to_dict() == from_nexus[1].convert_to_dict()  # the shift errors
    assert (from_json[2], from_json[3], int(from_json[4])) == (from_nexus[2], from_nexus[3], from_nexus[4])


def test_other_extension_is_written_as_json(ring_with_fits, tmp_path):
    """Only .nxs goes to NeXus: any other name is JSON, as before #1054 (review)"""
    # Arrange
    calibration, _ = ring_with_fits
    path = tmp_path / "HB2B_cal.txt"

    # Act
    calibration.write_calibration(str(path))

    # Assert
    assert json.loads(path.read_text())["Lambda"] == calibration.calibration_array[7]


def test_json_file_carries_no_diagnostics(ring_with_fits, tmp_path):
    """The JSON is unchanged: the diagnostics go only into the NeXus file"""
    # Arrange
    calibration, fits = ring_with_fits
    path = tmp_path / "cal.json"

    # Act
    calibration.write_calibration(str(path), strip_fits=fits)

    # Assert - no strip fits or covariance (the provenance only counts the ring points, as before)
    text = path.read_text()
    assert "strip_fits" not in text and "covariance" not in text and "two_theta_model" not in text


def test_failed_diagnostics_still_write_the_calibration(ring_with_fits, tmp_path, monkeypatch):
    # Arrange
    calibration, _ = ring_with_fits

    def failing(self, strip_fits=None):
        raise RuntimeError("no diagnostics")

    monkeypatch.setattr(PixelCalibration, "diagnostics", failing)
    path = str(tmp_path / "cal.nxs")

    # Act
    with pytest.warns(UserWarning, match="diagnostics could not be built"):
        calibration.write_calibration(path)

    # Assert - the calibration, without diagnostics
    assert read_calibration_nexus_diagnostics(path) is None
    assert read_calibration_file(path)[2] == calibration.calibration_array[7]


def _write(path, diagnostics):
    calibration = PixelCalibration(
        synthetic_ring_points(default_hb2b_geometry(), TRUE_PARAMS), default_hb2b_geometry(), 1.54
    )
    shift = pixel_refinement.to_shift(TRUE_PARAMS)
    write_calibration_nexus(
        str(path), shift, shift, 1.54, 1e-5, 1, calibration.model.instrument_setup, None, diagnostics
    )


@pytest.mark.parametrize(
    "diagnostics, match",
    [
        ({"tables": {"t": {"a": [1, 2], "b": [1]}}}, "different lengths"),
        ({"covariance": (("Shift_x", "Shift_y"), np.eye(3))}, "covariance of shape"),
    ],
    ids=["ragged-table", "covariance-shape"],
)
def test_malformed_diagnostics_raise(tmp_path, diagnostics, match):
    # Act / Assert
    with pytest.raises(TypeError, match=match):
        _write(tmp_path / "cal.nxs", diagnostics)


def test_patterns_round_trip(tmp_path):
    # Arrange
    pattern = {"two_theta": np.linspace(70, 80, 5), "intensity": np.arange(5.0), "errors": np.ones(5)}
    path = tmp_path / "cal.nxs"

    # Act
    _write(path, {"method": "peak-fit", "patterns": {"sub_run_1_eta_0": pattern}})
    read = read_calibration_nexus_diagnostics(str(path))["patterns"]["sub_run_1_eta_0"]

    # Assert
    for key in pattern:
        np.testing.assert_array_equal(read[key], pattern[key])
    with h5py.File(path, "r") as nexus:
        assert nexus["entry/calibration"].attrs["NX_class"] == "NXprocess"


NICKEL = PowderStandard("Ni", "FCC", 3.523799438, 62.0)


def _fit_calibration(final_peaks):
    """A FitCalibration holding only what diagnostics() reads, without a run"""
    fit = FitCalibration.__new__(FitCalibration)
    fit._standards = [NICKEL]
    fit._final_peaks = final_peaks
    fit._covariance = None
    return fit


def test_peaks_table_holds_bragg_and_fitted_centers():
    # Arrange - the 111 and 200 peaks of nickel, fitted 0.01 deg off Bragg
    wavelength = 1.54
    bragg = NICKEL.two_theta(wavelength)[:2]
    labels = [(1, "Ni", "FCC", 3, "eta_0"), (1, "Ni", "FCC", 4, "eta_0")]
    fit = _fit_calibration((np.full(2, 0.01), labels, np.full(2, 0.002), bragg + 0.01, wavelength))

    # Act
    peaks = fit.diagnostics()["tables"]["peaks"]

    # Assert - Bragg from the standard's own physics, the fitted centers as fitted
    np.testing.assert_allclose(peaks["bragg_two_theta"], bragg, rtol=1e-12)
    np.testing.assert_array_equal(peaks["fitted_two_theta"], bragg + 0.01)
    assert peaks["hkl_squared"] == [3, 4] and peaks["powder"] == ["Ni", "Ni"]


def test_patterns_need_the_peaks_at_the_solution():
    """Without a re-fit at the solution the last reduction is some other point: no patterns (review)"""
    # Arrange
    fit = _fit_calibration(None)
    fit.reducer = object()  # never read

    # Act
    with pytest.warns(UserWarning, match="not re-fitted at the solution"):
        diagnostics = fit.diagnostics(patterns=True)

    # Assert
    assert "patterns" not in diagnostics and diagnostics["tables"] == {}


def test_no_patterns_asked_no_warning():
    # Act / Assert
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert "patterns" not in _fit_calibration(None).diagnostics()


def _group_in_place_of(path):
    def change(nexus):
        del nexus[path]
        nexus.create_group(path)

    return change


@pytest.mark.parametrize(
    "path",
    [
        "entry/calibration/method",
        "entry/calibration/covariance",
        "entry/calibration/peaks/residual",
        "entry/calibration/patterns/sub_run_1_eta_0/two_theta",
    ],
    ids=["method", "covariance", "table-column", "pattern"],
)
def test_diagnostics_with_a_group_for_a_field_raise(tmp_path, path):
    """A ValueError naming the part, as the calibration reader gives, not an AttributeError (review)"""
    # Arrange - every part of the diagnostics present, then one field replaced by a group
    pattern = {"two_theta": np.linspace(70, 80, 5), "intensity": np.arange(5.0), "errors": np.ones(5)}
    diagnostics = {
        "method": "peak-fit",
        "covariance": (("Shift_x", "Shift_y"), np.eye(2)),
        "tables": {"peaks": {"residual": [0.01, -0.02], "powder": ["Ni", "Fe"]}},
        "patterns": {"sub_run_1_eta_0": pattern},
    }
    nexus_path = tmp_path / "cal.nxs"
    _write(nexus_path, diagnostics)
    assert read_calibration_nexus_diagnostics(str(nexus_path))["tables"]["peaks"]["powder"] == ["Ni", "Fe"]
    with h5py.File(nexus_path, "a") as nexus:
        _group_in_place_of(path)(nexus)

    # Act / Assert
    with pytest.raises(ValueError, match="expected a data field"):
        read_calibration_nexus_diagnostics(str(nexus_path))
