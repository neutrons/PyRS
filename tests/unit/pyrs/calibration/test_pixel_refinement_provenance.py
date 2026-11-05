"""Unit tests for the ring calibration's provenance (#1053): PixelCalibration.provenance and write_calibration"""

import copy
import json

import numpy as np
import pytest

from pyrs.calibration.pixel_pos_calibration import PixelCalibration
from pyrs.calibration.pixel_refinement import AT_BOUND, PARAMETER_NAMES, RANK_DEFICIENT
from pyrs.utilities.calibration_file_io import (
    read_calibration_geometry,
    read_calibration_json_file,
    read_calibration_provenance,
)
from tests.util.instrument_helpers import default_hb2b_geometry
from tests.util.ring_calibration_helpers import synthetic_ring_points

TRUE_PARAMS = np.array([0.004, -0.003, 0.012, 0.3, -0.5, 1.2, 0.656, 1.5308])
NOMINAL_WAVELENGTH = 1.540


@pytest.fixture(scope="module")
def hb2b():
    return default_hb2b_geometry()


@pytest.fixture(scope="module")
def _refined_once(hb2b):
    rings = synthetic_ring_points(hb2b, TRUE_PARAMS, strips=20, noise_deg=0.01, rng=np.random.default_rng(5))
    calibration = PixelCalibration(rings, hb2b, wavelength=NOMINAL_WAVELENGTH)
    calibration.iterate(passes=2)
    return calibration


@pytest.fixture
def refined(_refined_once):
    """A fresh copy per test: the stage methods change the calibration in place"""
    return copy.deepcopy(_refined_once)


def test_steps_follow_the_stages_in_order(refined):
    # Arrange
    refined.estimate_errors()

    # Act
    block = refined.provenance()

    # Assert
    assert block["method"] == "ring"
    assert [step["routine"] for step in block["steps"]] == [stage.name for stage in refined.stages]
    assert [step["step"] for step in block["steps"]] == list(range(1, len(refined.stages) + 1))
    first, last = block["steps"][0], block["steps"][-1]
    assert first["free_parameters"] == list(refined.stages[0].free_parameters)
    assert first["message"]  # the optimizer's termination message
    # the last step's values are the calibration written
    final = dict(zip(PARAMETER_NAMES, refined.calibration_array))
    assert last["values"] == {name: final[name] for name in last["free_parameters"]}
    assert last["routine"] == "estimate_errors" and "TTH_0" not in last["free_parameters"]
    assert block["held_parameters"] == ["TTH_0"]


def test_fit_quality_labels_every_ring_point(refined):
    # Act
    quality = refined.provenance()["fit_quality"]

    # Assert
    assert quality["n"] == len(refined.ring_points)
    assert quality["dof"] == quality["n"] - quality["n_free_parameters"]
    groups = quality["by_reflection_and_run"]
    assert sum(group["n"] for group in groups) == len(refined.ring_points)
    reflections = {group["reflection"] for group in groups}
    assert all(label.split()[0] in {"Ni", "Fe", "Mo"} and label.split()[1].isdigit() for label in reflections)


def test_degenerate_set_is_undetermined_with_its_reason(refined):
    """After FullCalibration, Shift_x / Rot_y / TTH_0 are exactly degenerate at one L2; Shift_y, Shift_z
    and Rot_z share the null move slightly, so the rank test flags them too (docs/ground_truths.md)"""
    # Arrange
    refined.FullCalibration()

    # Act
    block = refined.provenance()

    # Assert
    undetermined = {entry["parameter"]: entry["reason"] for entry in block["undetermined"]}
    assert set(undetermined) == {"Shift_x", "Shift_y", "Shift_z", "Rot_y", "Rot_z", "TTH_0"}
    assert set(undetermined.values()) == {RANK_DEFICIENT}


def test_parameter_at_a_bound_has_that_reason(refined):
    # Arrange - a bound far tighter than the true Shift_z of 12 mm
    stage = refined.refine("tight_shift_z", ["Shift_z"], [-1e-4], [1e-4], initial_guess=[0.0])

    # Act
    block = refined.provenance()

    # Assert
    assert stage.undetermined_reasons == {"Shift_z": AT_BOUND}
    assert {"parameter": "Shift_z", "reason": AT_BOUND} in block["undetermined"]


def test_start_records_set_calibration_array(hb2b):
    # Arrange
    rings = synthetic_ring_points(hb2b, TRUE_PARAMS, strips=5)
    calibration = PixelCalibration(rings, hb2b, wavelength=NOMINAL_WAVELENGTH)
    start = np.array([0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.53])

    # Act
    nominal = calibration.provenance()["start"]
    calibration.set_calibration_array(start)
    given = calibration.provenance()["start"]

    # Assert
    assert nominal["values"]["Lambda"] == NOMINAL_WAVELENGTH and "nominal" in nominal["source"]
    assert given == {"source": "set_calibration_array", "values": dict(zip(PARAMETER_NAMES, start))}


def test_set_two_theta_0_step_says_what_it_did(refined):
    # Act
    refined.set_two_theta_0(0.0)
    step = refined.provenance()["steps"][-1]

    # Assert
    assert step["routine"] == "set_two_theta_0" and step["nfev"] == 0
    assert step["message"].startswith("no refinement")


def test_write_carries_provenance_and_values(refined, tmp_path):
    # Arrange
    refined.estimate_errors()
    path = tmp_path / "ring.json"
    inputs = {"runs": [{"file": "/data/HB2B_3510.nxs.h5", "sha256": "abc"}], "monochromator": "Si422"}

    # Act
    refined.write_calibration(path, inputs=inputs)

    # Assert
    block = read_calibration_provenance(str(path))
    assert block["inputs"]["runs"] == inputs["runs"] and block["inputs"]["monochromator"] == "Si422"
    assert block["inputs"]["ring_points"] == len(refined.ring_points)
    shift, shift_error, wavelength, wavelength_error, status = read_calibration_json_file(str(path))
    assert wavelength == refined.calibration_array[7] and status == refined.calibration_status
    assert shift_error.center_shift_x == refined.calibration_error_array[0]
    assert "NaN" not in path.read_text()
    geometry = read_calibration_geometry(str(path))  # the calibration carries its detector geometry
    assert geometry.detector_size == refined._model.instrument_setup.detector_size
    json.loads(path.read_text())


def test_new_start_drops_earlier_steps(refined):
    # Arrange
    assert refined.stages  # the fixture ran a recipe

    # Act
    refined.set_calibration_array(refined.calibration_array)
    before = refined.provenance()["steps"]
    refined.CalibrateShift()
    after = refined.provenance()["steps"]

    # Assert - stages are kept for the GUI, but only those since the start are this calibration's
    assert before == []
    assert [(step["step"], step["routine"]) for step in after] == [(1, "CalibrateShift")]


def test_restore_drops_steps_and_returns_the_start(refined):
    # Arrange
    routines = [step["routine"] for step in refined.provenance()["steps"]]
    start = refined.provenance()["start"]
    checkpoint = refined.checkpoint()
    refined.set_calibration_array(refined.calibration_array)  # a new start, then a failure
    refined.CalibrateShift()

    # Act
    refined.restore(checkpoint)

    # Assert
    block = refined.provenance()
    assert [step["routine"] for step in block["steps"]] == routines
    assert block["start"] == start


def test_write_is_never_blocked_by_provenance(refined, tmp_path, monkeypatch):
    # Arrange
    def broken(inputs=None):
        raise KeyError("first_stage")

    monkeypatch.setattr(refined, "provenance", broken)
    path = tmp_path / "ring.json"

    # Act
    with pytest.warns(UserWarning, match="could not be built"):
        refined.write_calibration(path)

    # Assert - the calibration is written; its provenance says why it is missing
    assert read_calibration_json_file(str(path))[2] == refined.calibration_array[7]
    assert "KeyError" in read_calibration_provenance(str(path))["error"]
