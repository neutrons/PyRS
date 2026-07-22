"""Unit tests for pyrs.core.reduce_hb2b_pyrs"""

import numpy as np

from pyrs.core.instrument_geometry import DENEXDetectorGeometry
from pyrs.core.reduce_hb2b_pyrs import PyHB2BReduction, ResidualStressInstrument


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


def _make_histogram_inputs():
    pixel_2theta = np.array([10.0, 10.0, 11.0, 11.0])
    counts = np.array([100.0, 150.0, 80.0, 120.0])
    vanadium = np.array([50.0, 50.0, 200.0, 200.0])
    two_theta_bins = np.array([9.5, 10.5, 11.5])
    return pixel_2theta, counts, vanadium, two_theta_bins


def test_histogram_by_numpy_van_duration_scales_intensity():
    """Test that the vanadium-normalized intensity scales linearly with van_duration

    Regression test: sub_run_duration and van_duration were threaded through
    reduce_sub_run_diffraction/reduce_sub_run_texture but never passed on to
    convert_counts_to_diffraction/reduce_to_2theta_histogram/histogram_by_numpy, so the
    vanadium-normalized intensity was silently independent of how long the vanadium and
    sample runs each counted for.
    """
    # Arrange
    pixel_2theta, counts, vanadium, two_theta_bins = _make_histogram_inputs()

    # Act - vary van_duration only, holding sub_run_duration fixed
    _, hist_van20, _ = PyHB2BReduction.histogram_by_numpy(
        pixel_2theta, counts, two_theta_bins, True, vanadium, sub_run_duration=10.0, van_duration=20.0
    )
    _, hist_van10, _ = PyHB2BReduction.histogram_by_numpy(
        pixel_2theta, counts, two_theta_bins, True, vanadium, sub_run_duration=10.0, van_duration=10.0
    )

    # Assert - halving van_duration (20 -> 10) halves the normalized intensity
    np.testing.assert_allclose(hist_van10, hist_van20 / 2.0)


def test_histogram_by_numpy_sub_run_duration_scales_intensity_inversely():
    """Test that the vanadium-normalized intensity scales inversely with sub_run_duration

    Regression test: this pins down that sub_run_duration is actually used as the divisor
    (not just van_duration as the multiplier) — a bug that dropped sub_run_duration entirely
    (e.g. `normalized_data * van_duration`) would still pass a test that only varies
    van_duration, since the observed output ratio would look identical.
    """
    # Arrange
    pixel_2theta, counts, vanadium, two_theta_bins = _make_histogram_inputs()

    # Act - vary sub_run_duration only, holding van_duration fixed
    _, hist_sample10, _ = PyHB2BReduction.histogram_by_numpy(
        pixel_2theta, counts, two_theta_bins, True, vanadium, sub_run_duration=10.0, van_duration=20.0
    )
    _, hist_sample5, _ = PyHB2BReduction.histogram_by_numpy(
        pixel_2theta, counts, two_theta_bins, True, vanadium, sub_run_duration=5.0, van_duration=20.0
    )

    # Assert - halving sub_run_duration (10 -> 5) doubles the normalized intensity
    np.testing.assert_allclose(hist_sample5, hist_sample10 * 2.0)


def test_histogram_by_numpy_no_duration_given_matches_shape_correction_only():
    """Test that omitting sub_run_duration/van_duration leaves the existing shape-only correction unchanged"""
    # Arrange
    pixel_2theta, counts, vanadium, two_theta_bins = _make_histogram_inputs()

    # Act
    _, hist_no_duration, _ = PyHB2BReduction.histogram_by_numpy(pixel_2theta, counts, two_theta_bins, True, vanadium)
    _, hist_ratio_1, _ = PyHB2BReduction.histogram_by_numpy(
        pixel_2theta, counts, two_theta_bins, True, vanadium, sub_run_duration=10.0, van_duration=10.0
    )

    # Assert - a duration ratio of 1 (equal run times) matches the no-duration-given baseline
    np.testing.assert_allclose(hist_no_duration, hist_ratio_1)
