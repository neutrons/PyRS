"""Rotation helpers: rotation matrices, signed sample axes and the rotation checks."""

import numpy as np
import pytest

from pyrs.pole_figures.core.rotation import axis_name, axis_vector, check_rotation, nearest_rotation, rot


def test_rot_is_right_handed() -> None:
    # Arrange
    R = rot([0, 0, 1], 90)

    # Act
    x_turned = R @ [1, 0, 0]

    # Assert
    np.testing.assert_allclose(x_turned, [0, 1, 0], atol=1e-15)


def test_rot_is_orthogonal() -> None:
    # Arrange
    R = rot([1, 2, 3], 37)

    # Act
    product = R @ R.T

    # Assert
    np.testing.assert_allclose(product, np.eye(3), atol=1e-15)


def test_axis_vector_signed_axis() -> None:
    # Arrange
    axis = "-vz"

    # Act
    v = axis_vector(axis)

    # Assert
    np.testing.assert_array_equal(v, [0.0, 0.0, -1.0])


def test_axis_vector_unknown_axis_raises() -> None:
    # Arrange
    axis = "z"

    # Act / Assert
    with pytest.raises(ValueError, match="sample axis must be one of"):
        axis_vector(axis)


def test_axis_name_takes_largest_component() -> None:
    # Arrange
    v = [0.1, -0.9, 0.2]

    # Act
    name = axis_name(v)

    # Assert
    assert name == "-vy"


def test_check_rotation_reflection_raises() -> None:
    # Arrange
    m = np.diag([1.0, 1.0, -1.0])

    # Act / Assert
    with pytest.raises(ValueError, match="is a reflection"):
        check_rotation(m)


def test_nearest_rotation_keeps_identity_exact() -> None:
    # Arrange
    m = np.eye(3)

    # Act
    r = nearest_rotation(m)

    # Assert
    np.testing.assert_array_equal(r, np.eye(3))
