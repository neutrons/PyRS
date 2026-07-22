"""Unit tests for pyrs.core.reduce_hb2b_pyrs"""

import numpy as np

from pyrs.core.instrument_geometry import DENEXDetectorGeometry
from pyrs.core.reduce_hb2b_pyrs import ResidualStressInstrument


def _make_instrument():
    pixel_size = 0.3 / 1024.0
    arm_length = 0.985
    test_setup = DENEXDetectorGeometry(1024, 1024, pixel_size, pixel_size, arm_length, False)
    return ResidualStressInstrument(test_setup)


def test_generate_rotation_matrix_combined_rotation_is_orthogonal():
    """Test that combining non-zero rotations about all three axes still yields a proper rotation

    Regression test: generate_rotation_matrix previously combined the three axis rotation
    matrices with element-wise multiplication (`*`) instead of matrix multiplication (`@`),
    which produces a near-diagonal matrix that is not orthogonal whenever two or more angles
    are simultaneously non-zero.
    """
    # Arrange
    instrument = _make_instrument()
    rot_x_rad = np.deg2rad(0.3)
    rot_y_rad = np.deg2rad(-0.5)
    rot_z_rad = np.deg2rad(1.2)

    # Act
    rotation_matrix = instrument.generate_rotation_matrix(rot_x_rad, rot_y_rad, rot_z_rad)

    # Assert - a valid rotation matrix is orthogonal (R @ R.T == I) with determinant +1
    np.testing.assert_allclose(rotation_matrix @ rotation_matrix.T, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(rotation_matrix), 1.0, atol=1e-6)


def test_generate_rotation_matrix_single_axis_matches_axis_matrix():
    """Test that rotating about a single axis reduces to that axis's own rotation matrix"""
    # Arrange
    instrument = _make_instrument()
    rot_x_rad = np.deg2rad(2.0)

    # Act
    rotation_matrix = instrument.generate_rotation_matrix(rot_x_rad, 0.0, 0.0)
    expected = instrument._cal_rotation_matrix_x(rot_x_rad)

    # Assert
    np.testing.assert_allclose(rotation_matrix, expected, atol=1e-6)
