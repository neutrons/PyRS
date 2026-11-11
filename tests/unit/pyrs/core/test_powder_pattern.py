"""Unit tests for pyrs.core.powder_pattern: calibration file formats accepted by ReductionApp (#1051)"""

from types import SimpleNamespace

import numpy as np
import pytest

from pyrs.core.powder_pattern import ReductionApp, _describe_extra_mask


@pytest.mark.parametrize(
    "file_name", ["HB2B_geometry.txt", "HB2B_Latest.JSON.bak", "HB2B_1327.h5", "HB2B_3510.nxs.h5"]
)
def test_non_calibration_file_raises(file_name):
    """Only JSON and NeXus (.nxs, #1054) calibrations: the ASCII geometry reader was removed (#1051), .h5 is a
    project file and .nxs.h5 raw event data"""
    # Arrange
    reducer = ReductionApp()
    reducer._hydra_ws = SimpleNamespace(get_detector_shift=lambda: None)

    # Act / Assert
    with pytest.raises(ValueError, match="must be a .json or .nxs file|raw event data"):
        reducer.reduce_data([1], None, file_name, None)


def test_describe_extra_mask_none():
    # Act / Assert
    assert _describe_extra_mask(None) is None


def test_describe_extra_mask_counts_and_fingerprints_an_array():
    """#1068: the record names the extra mask reduce_data applied, without storing a pixel array"""
    # Act
    described = _describe_extra_mask(np.array([1, 0, 1, 0, 0]))

    # Assert
    assert described["array"] == "supplied by the caller" and described["pixels_masked"] == 3
    assert len(described["sha256"]) == 64


def test_describe_extra_mask_names_differ_by_content():
    """The same name for different pixels (e.g. in two files) is not the same mask (review)"""
    # Act
    first = _describe_extra_mask("upper", np.array([1, 0, 1, 1]))
    second = _describe_extra_mask("upper", np.array([1, 1, 0, 1]))

    # Assert - same name and count, different pixels
    assert first["name"] == second["name"] == "upper" and first["pixels_masked"] == second["pixels_masked"] == 1
    assert first["sha256"] != second["sha256"]


def test_describe_extra_mask_ignores_the_array_dtype():
    # Act / Assert - 1/0 as int, float or bool keep the same pixels
    assert len({_describe_extra_mask(np.array([1, 0, 1], dtype=t))["sha256"] for t in (int, float, bool)}) == 1
