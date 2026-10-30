"""A project file keeps one 2theta axis for all masks: writes must not relabel masks already in it"""

import h5py
import numpy as np
import pytest

from pyrs.dataobjects.constants import HidraConstants
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore

AXIS = np.tile(np.linspace(60.0, 70.0, 5), (2, 1))  # 2 sub-runs x 5 bins
OTHER_AXIS = AXIS + 8.0


def write(path, mode, axis, masks):
    project = HidraProjectFile(str(path), mode)
    project.write_reduced_diffraction_data_set(
        axis, {m: np.full(axis.shape, i + 1.0) for i, m in enumerate(masks)}, None
    )
    project.close()


def stored(path):
    with h5py.File(path, "r") as f:
        group = f[HidraConstants.REDUCED_DATA]
        return group[HidraConstants.TWO_THETA][()], sorted(k for k in group if k != HidraConstants.TWO_THETA)


def test_append_with_different_axis_raises(tmp_path):
    # Arrange
    path = tmp_path / "project.h5"
    write(path, HidraProjectFileMode.OVERWRITE, AXIS, ["left"])

    # Act / Assert - mask "left" would be left on the new axis
    with pytest.raises(RuntimeError, match="left"):
        write(path, HidraProjectFileMode.READWRITE, OTHER_AXIS, ["right"])
    axis, _ = stored(path)
    np.testing.assert_array_equal(axis, AXIS)


def test_append_with_same_axis_keeps_both(tmp_path):
    # Arrange
    path = tmp_path / "project.h5"
    write(path, HidraProjectFileMode.OVERWRITE, AXIS, ["left"])

    # Act
    write(path, HidraProjectFileMode.READWRITE, AXIS.copy(), ["right"])

    # Assert
    axis, names = stored(path)
    np.testing.assert_array_equal(axis, AXIS)
    assert names == ["left", "left_var", "right", "right_var"]


@pytest.mark.parametrize("rewritten", [["main"], ["main", "wedge"]])
def test_rewriting_every_mask_may_change_the_axis(tmp_path, rewritten):
    # Arrange
    path = tmp_path / "project.h5"
    write(path, HidraProjectFileMode.OVERWRITE, AXIS, rewritten)

    # Act - nothing is left behind, so a new axis is fine
    write(path, HidraProjectFileMode.READWRITE, OTHER_AXIS, rewritten)

    # Assert
    axis, _ = stored(path)
    np.testing.assert_array_equal(axis, OTHER_AXIS)
