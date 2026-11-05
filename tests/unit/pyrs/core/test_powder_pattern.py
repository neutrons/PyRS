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
