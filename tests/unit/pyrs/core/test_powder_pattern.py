"""Unit tests for pyrs.core.powder_pattern: calibration file formats accepted by ReductionApp (#1051)"""

from types import SimpleNamespace

import numpy as np
import pytest

from pyrs.core.monosetting import MonoSetting
from pyrs.core.powder_pattern import ReductionApp, _describe_extra_mask, _reduction_l2, _run_monosetting
from pyrs.core.workspaces import HidraWorkspace
from tests.util.workspace_helpers import make_minimal_HidraWorkspace


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


def test_reduction_l2_is_the_log():
    # Arrange
    workspace = make_minimal_HidraWorkspace(sub_runs=[1, 2])
    workspace.set_sample_log("L2", np.array([1, 2]), np.array([0.95, 0.95]))

    # Act / Assert
    assert _reduction_l2(workspace, [1, 2]) == 0.95


def test_reduction_l2_without_log_is_arm_length():
    # Arrange
    workspace = make_minimal_HidraWorkspace(sub_runs=[1, 2])

    # Act / Assert
    assert _reduction_l2(workspace, [1, 2]) == workspace.get_instrument_setup().arm_length


def test_differing_l2_raises():
    """One reduction applies one calibration at one distance (#1141)"""
    # Arrange
    workspace = make_minimal_HidraWorkspace(sub_runs=[1, 2])
    workspace.set_sample_log("L2", np.array([1, 2]), np.array([0.95, 0.96]))

    # Act / Assert
    with pytest.raises(RuntimeError, match=r"L2\) differs .*\[0.95, 0.96\]"):
        _reduction_l2(workspace, [1, 2])


def test_monosetting_is_from_mrot():
    # Arrange
    workspace = make_minimal_HidraWorkspace(sub_runs=[1, 2])
    workspace.set_sample_log("mrot", np.array([1, 2]), np.array([-170.0, -170.0]))

    # Act / Assert
    assert _run_monosetting(workspace, [1, 2]) is MonoSetting.Si331


def test_differing_monochromators_raise():
    # Arrange
    workspace = make_minimal_HidraWorkspace(sub_runs=[1, 2])
    workspace.set_sample_log("mrot", np.array([1, 2]), np.array([-170.0, -20.0]))

    # Act / Assert
    with pytest.raises(RuntimeError, match="different monochromators"):
        _run_monosetting(workspace, [1, 2])


def test_missing_mrot_raises():
    # Arrange
    workspace = HidraWorkspace("no_mono")
    workspace.set_sub_runs(np.array([1, 2]))

    # Act / Assert
    with pytest.raises(RuntimeError, match="'no_mono' has no 'mrot' log"):
        _run_monosetting(workspace, [1, 2])


def test_unknown_rotation_raises():
    # Arrange
    workspace = make_minimal_HidraWorkspace(sub_runs=[1, 2])
    workspace.set_sample_log("mrot", np.array([1, 2]), np.array([100.0, 100.0]))

    # Act / Assert
    with pytest.raises(ValueError, match="monochromator rotation angle"):
        _run_monosetting(workspace, [1, 2])
