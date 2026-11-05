"""Unit tests for pyrs.core.powder_pattern: calibration file formats accepted by ReductionApp (#1051)"""

from types import SimpleNamespace

import pytest

from pyrs.core.powder_pattern import ReductionApp


@pytest.mark.parametrize("file_name", ["HB2B_geometry.txt", "HB2B_Latest.JSON.bak"])
def test_non_json_calibration_file_raises(file_name):
    """The ASCII geometry reader was removed: it crashed before reading, and its keys were wrong"""
    # Arrange
    reducer = ReductionApp()
    reducer._hydra_ws = SimpleNamespace(get_detector_shift=lambda: None)

    # Act / Assert
    with pytest.raises(ValueError, match="must be a .json file"):
        reducer.reduce_data([1], None, file_name, None)


def test_failed_calibration_record_does_not_stop_reduction(monkeypatch):
    """#1053 part B: recording what was applied must never stop a reduction"""
    # Arrange
    from pyrs.core import powder_pattern

    def broken(path):
        raise OSError("calibration file vanished")

    monkeypatch.setattr(powder_pattern.calibration_file_io, "read_calibration_provenance", broken)
    reducer = ReductionApp()
    values = (SimpleNamespace(center_shift_x=0.0), None, 1.54, 1e-4, 1)

    # Act
    with pytest.warns(UserWarning, match="could not be recorded"):
        record = reducer._record_applied_calibration("calibration.json", values, None)

    # Assert
    assert record["source"] == "unknown" and "calibration file vanished" in record["error"]
