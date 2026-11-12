"""Repeated ring extraction for the ring-based calibration (#1038), on a synthetic calibration run"""

import numpy as np
import pytest

import copy

from pyrs.calibration import ring_rounds
from pyrs.calibration.pixel_pos_calibration import (
    CONVERGENCE_FRACTION,
    MIN_ROUNDS,
    PixelCalibration,
    calibrate_rings,
    continue_rounds,
    extract_ring_points,
    pixel_two_theta_change,
    ring_position_change,
    statistical_precision,
)
from pyrs.core.detector_model import from_shift, to_shift
from tests.util.instrument_helpers import default_hb2b_geometry
from tests.util.ring_calibration_helpers import synthetic_calibration_workspace

TRUE_PARAMS = np.array([0.004, -0.003, 0.012, 0.3, -0.5, 1.2, 0.656, 1.5308])
MONO = 1.54


@pytest.fixture(scope="module")
def workspace():
    return synthetic_calibration_workspace(default_hb2b_geometry(), TRUE_PARAMS)


@pytest.fixture(scope="module")
def converged_run(workspace):
    """The default calibration, and the strip-fit diagnostics it was given a stale list for (#1101: one run
    shared by every test that only inspects it)"""
    diagnostics = ["stale"]
    return calibrate_rings(workspace, MONO, diagnostics=diagnostics), diagnostics


@pytest.fixture(scope="module")
def converged(converged_run):
    return converged_run[0]


@pytest.fixture(scope="module")
def unconverged(workspace):
    """Three rounds that cannot converge, each refinement warning once: the round-limit warning and every
    round's warnings in one run (#1101)"""
    real_iterate = PixelCalibration.iterate

    def warning_iterate(self, *args, **kwargs):
        self.warning_log.append(f"warned with {len(self.ring_points)} points")
        return real_iterate(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ring_rounds, "CONVERGENCE_FRACTION", 0.0)
        patch.setattr(PixelCalibration, "iterate", warning_iterate)
        with pytest.warns(UserWarning, match="not converged after 3 rounds") as warned:
            calibration = calibrate_rings(workspace, MONO, max_rounds=3)
    return calibration, warned


@pytest.fixture(scope="module")
def one_extraction(workspace):
    """One extraction of the ring points, outside calibrate_rings (#1101: shared; each test takes a copy)"""
    return extract_ring_points(workspace, MONO, strips=5)


def test_rounds_stop_below_half_the_precision(converged):
    # Act
    rounds = converged.extraction_rounds

    # Assert - at least two rounds; only the last is below the tolerance
    assert MIN_ROUNDS <= len(rounds) < ring_rounds.DEFAULT_MAX_ROUNDS
    assert rounds[0]["step_deg"] is None and [r["round"] for r in rounds] == list(range(1, len(rounds) + 1))
    assert rounds[-1]["step_deg"] < CONVERGENCE_FRACTION * rounds[-1]["precision_deg"]
    for record in rounds[1:-1]:
        assert record["round"] < MIN_ROUNDS or record["step_deg"] >= CONVERGENCE_FRACTION * record["precision_deg"]
    assert converged.warning_log == [] and converged.extraction_converged is True


def test_rounds_recover_the_true_geometry(converged, workspace):
    # Act - compared as pixel 2theta (Rot_y, Shift_x and 2theta_0 are degenerate)
    change = pixel_two_theta_change(converged, TRUE_PARAMS, converged.calibration_array)

    # Assert
    assert change < 0.01
    assert converged.calibration_array[7] == pytest.approx(TRUE_PARAMS[7], abs=1e-4)


def test_round_record_matches_the_calibration(converged):
    # Act
    last = converged.extraction_rounds[-1]

    # Assert
    assert last["values"] == pytest.approx(list(converged.calibration_array))
    assert last["ring_points"] == len(converged.ring_points)
    assert last["precision_deg"] == pytest.approx(statistical_precision(converged))


def test_rounds_are_in_the_provenance(converged):
    # Act
    provenance = converged.provenance()

    # Assert
    assert provenance["extraction_rounds"] == converged.extraction_rounds


def test_single_extraction_has_no_rounds_in_provenance(one_extraction):
    """A calibration not made by calibrate_rings writes the provenance it wrote before"""
    # Arrange
    calibration = PixelCalibration(copy.deepcopy(one_extraction), default_hb2b_geometry(), MONO)

    # Act
    calibration.iterate(passes=1)

    # Assert
    assert calibration.extraction_rounds is None and "extraction_rounds" not in calibration.provenance()


def test_one_round_is_the_single_extraction(workspace):
    """max_rounds=1 reproduces the workflow before #1038: no second extraction, no warning"""
    # Act
    calibration = calibrate_rings(workspace, MONO, max_rounds=1)

    # Assert - nothing to judge: neither converged nor not
    assert len(calibration.extraction_rounds) == 1 and calibration.warning_log == []
    assert calibration.extraction_converged is None


def test_round_limit_reached_warns(unconverged):
    """P4: a calibration still moving at the round limit is flagged, not corrected"""
    # Arrange - no round can converge (the fixture), and it warned (checked there)
    calibration, warned = unconverged

    # Assert - the warning stays with the result, e.g. for the GUI
    assert any("not converged after 3 rounds" in str(record.message) for record in warned)
    assert len(calibration.extraction_rounds) == 3 and calibration.extraction_converged is False
    assert any("not converged after 3 rounds" in message for message in calibration.warning_log)


def test_earlier_rounds_warnings_are_kept(unconverged):
    # Arrange - every round's refinement warned once (the fixture)
    calibration, _ = unconverged

    # Assert - one per round, all kept; besides them only the round-limit warning
    per_round = [message for message in calibration.warning_log if message.startswith("warned with")]
    assert len(per_round) == len(calibration.extraction_rounds) == 3
    assert len(calibration.warning_log) == len(per_round) + 1


def test_diagnostics_are_the_last_rounds(converged_run):
    # Arrange - the run was given a stale list (the fixture)
    _, diagnostics = converged_run

    # Assert
    assert diagnostics and "stale" not in diagnostics


def test_start_is_recorded_after_several_rounds(workspace):
    """Later rounds start where the previous ended; the first round's record keeps the real start"""
    # Arrange
    start = TRUE_PARAMS.copy()

    # Act
    calibration = calibrate_rings(workspace, MONO, start=start)

    # Assert
    assert len(calibration.extraction_rounds) >= 2
    first = calibration.extraction_rounds[0]["start"]
    assert first["source"] == "given starting calibration" and first["values"] == pytest.approx(list(start))
    assert calibration.provenance()["extraction_rounds"][0]["start"] == first


def test_nominal_start_is_recorded(converged):
    # Act
    first = converged.extraction_rounds[0]["start"]

    # Assert
    assert first["values"] == pytest.approx([0.0] * 7 + [MONO])


def test_second_call_appends_its_rounds(converged, workspace):
    """e.g. a GUI recipe with "iterate" twice: both calls' rounds, numbered on"""
    # Arrange
    earlier = list(converged.extraction_rounds)

    # Act
    again = continue_rounds(copy.deepcopy(converged), workspace, max_rounds=2)

    # Assert
    rounds = again.extraction_rounds
    assert rounds[: len(earlier)] == earlier and len(rounds) == len(earlier) + 2
    assert [r["round"] for r in rounds] == list(range(1, len(rounds) + 1))
    assert rounds[len(earlier)]["step_deg"] is None and "start" in rounds[len(earlier)]


@pytest.mark.parametrize("start", [np.zeros(7), np.zeros((2, 8))], ids=["seven", "two-d"])
def test_start_of_wrong_shape_raises(workspace, start):
    # Act / Assert
    with pytest.raises(ValueError, match="8 values"):
        calibrate_rings(workspace, MONO, start=start)


def test_zero_rounds_raises_before_extracting(workspace, monkeypatch):
    # Arrange - the extraction takes most of the time: a bad limit must not wait for it
    def no_extraction(*args, **kwargs):
        raise AssertionError("extracted before checking the round limit")

    monkeypatch.setattr(ring_rounds, "extract_ring_points", no_extraction)

    # Act / Assert
    with pytest.raises(ValueError, match="max_rounds"):
        calibrate_rings(workspace, MONO, max_rounds=0)


def test_precision_of_unrefined_calibration_raises(one_extraction):
    # Arrange
    calibration = PixelCalibration(copy.deepcopy(one_extraction), default_hb2b_geometry(), MONO)

    # Act / Assert
    with pytest.raises(ValueError, match="not been refined"):
        statistical_precision(calibration)


def test_precision_is_rms_times_root_p_over_n(converged):
    # Act / Assert
    expected = converged.stages[-1].rmse * np.sqrt(8 / len(converged.ring_points))
    assert statistical_precision(converged) == pytest.approx(expected)


def test_same_detector_described_two_ways_has_no_change(converged):
    """Rot_y / Shift_x / 2theta_0 trade-offs place the pixels identically"""
    # Arrange
    values = converged.calibration_array
    equivalent = converged.model.with_two_theta_0(
        to_shift(values[:7]), 0.0, converged.model.instrument_setup.arm_length
    )
    other = np.r_[from_shift(equivalent), values[7]]

    # Act / Assert
    assert pixel_two_theta_change(converged, values, other) < 1e-6


def test_different_geometry_changes_pixel_two_theta(converged):
    # Arrange - 1 mm along the beam at the detector
    values = converged.calibration_array
    moved = values.copy()
    moved[0] += 0.001

    # Act / Assert - about 1 mm / 0.985 m = 0.058 deg at the detector centre
    assert 0.03 < pixel_two_theta_change(converged, values, moved) < 0.1


def test_wavelength_only_round_moves_the_rings(converged):
    """A round that only moves lambda leaves the pixels where they were but not the rings (review)"""
    # Arrange
    values = converged.calibration_array
    moved = values.copy()
    moved[7] *= 1.0001

    # Act
    pixel, rings = pixel_two_theta_change(converged, values, moved), ring_position_change(converged, values, moved)

    # Assert - 1e-4 relative at 2theta ~ 80-100 deg: about 2 tan(theta) 1e-4 rad = 12 mdeg
    assert pixel == 0.0 and 0.005 < rings < 0.03


def test_step_is_the_larger_change(converged):
    # Act / Assert
    for record in converged.extraction_rounds[1:]:
        assert record["step_deg"] == max(record["pixel_step_deg"], record["ring_step_deg"])
