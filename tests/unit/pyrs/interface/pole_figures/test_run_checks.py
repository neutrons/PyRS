"""The Data and Corrections checks: spatial-map runs, the gauge margin inside the sample, negative net rates."""

import numpy as np

from pyrs.interface.pole_figures.run_checks import (
    GAUGE_MARGIN_MM,
    inside_with_margin,
    is_spatial_map,
    mostly_negative,
    orientation_count,
)
from tests.util.pole_figures_helpers import SLAB_CENTER_MM, SLAB_SIZE_MM, slab_mesh, synthetic_run_data


def test_orientation_count_rounds_to_hundredth_deg() -> None:
    # Arrange: encoder noise below 0.005 deg, and one real step
    chi = np.array([0.0, 0.002, -0.001, 10.0])
    phi = np.array([90.0, 90.003, 89.999, 90.0])

    # Act
    n = orientation_count(chi, phi)

    # Assert
    assert n == 2


def test_is_spatial_map_one_orientation_true() -> None:
    # Arrange: a vy x vz map at chi = phi = 0 (PyRS example runs 2246, 2247, 2251)
    run = synthetic_run_data(n=9)
    run.chi, run.phi = np.zeros(9), np.zeros(9)

    # Act
    result = is_spatial_map(run)

    # Assert
    assert result


def test_is_spatial_map_chi_scan_false() -> None:
    # Arrange: chi 0 to 50 deg at phi 0
    run = synthetic_run_data(n=6)

    # Act
    result = is_spatial_map(run)

    # Assert
    assert not result


def test_inside_with_margin_center_true() -> None:
    # Arrange
    mesh = slab_mesh()

    # Act
    ok = inside_with_margin(mesh, SLAB_CENTER_MM)

    # Assert
    assert ok


def test_inside_with_margin_corner_false() -> None:
    # Arrange: (0, 0, 0) is a corner of the slab
    mesh = slab_mesh()

    # Act
    ok = inside_with_margin(mesh, (0.0, 0.0, 0.0))

    # Assert
    assert not ok


def test_inside_with_margin_near_face_false() -> None:
    # Arrange: half the margin below the top face, and twice the margin below it
    mesh = slab_mesh()
    top = SLAB_SIZE_MM[2]

    # Act
    near = inside_with_margin(mesh, (20.0, 20.0, top - GAUGE_MARGIN_MM / 2))
    clear = inside_with_margin(mesh, (20.0, 20.0, top - 2 * GAUGE_MARGIN_MM))

    # Assert
    assert not near and clear


def test_mostly_negative_median_below_zero_true() -> None:
    # Arrange
    rates = np.array([-5.0, -1.0, 2.0, np.nan])

    # Act
    flags = [mostly_negative(rates), mostly_negative(-rates), mostly_negative(np.array([np.nan]))]

    # Assert
    assert flags == [True, False, False]
