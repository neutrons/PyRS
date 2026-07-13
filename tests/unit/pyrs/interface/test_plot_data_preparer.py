"""Unit tests for the shared 3D plot-data preparer.

These exercise the colormap/griddata/scatter-vs-contour decision that was
previously duplicated between the peak-fitting and texture-fitting controllers.
"""

import numpy as np
import pytest

from pyrs.interface.utilities.plot_data_preparer import prepare_3d_plot_data


# A small regular grid: x and y both repeat, so the data is griddable
# (neither axis is all-unique).
GRID_X = np.array([1.0, 1.0, 2.0, 2.0])
GRID_Y = np.array([1.0, 2.0, 1.0, 2.0])
GRID_Z = np.array([10.0, 20.0, 30.0, 40.0])


def test_prepare_3d_plot_data_scatter_returns_input_copies():
    """Scatter mode returns copies of the inputs flagged as a scatter plot."""
    # Arrange
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    z = np.array([7.0, 8.0, 9.0])

    # Act
    vec_x, vec_y, vec_z, colors, plot_scatter = prepare_3d_plot_data(x, y, z, mode="scatter")

    # Assert
    assert plot_scatter is True
    assert colors is not None
    np.testing.assert_array_equal(vec_x, x)
    np.testing.assert_array_equal(vec_y, y)
    np.testing.assert_array_equal(vec_z, z)


def test_prepare_3d_plot_data_contour_grids_without_colors():
    """Contour mode on griddable data returns a 2D grid and no colors."""
    # Act
    vec_x, vec_y, vec_z, colors, plot_scatter = prepare_3d_plot_data(GRID_X, GRID_Y, GRID_Z, mode="contour")

    # Assert
    assert plot_scatter is False
    assert colors is None
    assert vec_x.shape == (2, 2)
    assert vec_z.shape == (2, 2)


def test_prepare_3d_plot_data_lines_grids_with_colors():
    """3D-line mode on griddable data returns a grid plus an RGBA color array."""
    # Act
    vec_x, vec_y, vec_z, colors, plot_scatter = prepare_3d_plot_data(GRID_X, GRID_Y, GRID_Z, mode="lines")

    # Assert
    assert plot_scatter is False
    assert colors is not None
    assert vec_z.shape == (2, 2)
    # RGBA color per grid cell
    assert colors.shape == (2, 2, 4)


def test_prepare_3d_plot_data_contour_auto_promotes_to_scatter_when_ungriddable():
    """All-unique x cannot be gridded, so contour mode falls back to scatter."""
    # Arrange: x is strictly increasing/unique -> not griddable
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 1.0, 2.0, 2.0])
    z = np.array([5.0, 6.0, 7.0, 8.0])

    # Act
    vec_x, vec_y, vec_z, colors, plot_scatter = prepare_3d_plot_data(x, y, z, mode="contour")

    # Assert
    assert plot_scatter is True
    np.testing.assert_array_equal(vec_x, x)


def test_prepare_3d_plot_data_empty_z_returns_no_colors():
    """Empty z data has no min/max; colors fall back to None rather than raising."""
    # Arrange
    empty = np.array([])

    # Act
    vec_x, vec_y, vec_z, colors, plot_scatter = prepare_3d_plot_data(empty, empty, empty, mode="scatter")

    # Assert
    assert plot_scatter is True
    assert colors is None


def test_prepare_3d_plot_data_unknown_mode_defaults_to_scatter():
    """An unrecognized mode is treated as scatter."""
    # Arrange
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 4.0])
    z = np.array([5.0, 6.0])

    # Act
    _, _, _, _, plot_scatter = prepare_3d_plot_data(x, y, z, mode="not-a-mode")

    # Assert
    assert plot_scatter is True


def test_prepare_3d_plot_data_accepts_python_lists():
    """Inputs may be plain lists, not only numpy arrays."""
    # Act
    vec_x, vec_y, vec_z, colors, plot_scatter = prepare_3d_plot_data(
        [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], mode="scatter"
    )

    # Assert
    assert isinstance(vec_x, np.ndarray)
    assert plot_scatter is True
    assert vec_z[-1] == pytest.approx(9.0)
