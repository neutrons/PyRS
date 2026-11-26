"""The chi/phi grid: points, equal spacing and solid-angle weights."""

import numpy as np
import pytest

from pyrs.pole_figures.core.grid import Grid, solid_angle_weights


def test_grid_points_pf_grid() -> None:
    # Arrange
    grid = Grid(10, 5)

    # Act
    chi, phi = grid.points()

    # Assert
    assert len(chi) == 649
    assert (chi[-1], phi[-1]) == (90, 0)


def test_solid_angle_weights_cover_hemisphere() -> None:
    # Arrange
    chi, _ = Grid(10, 5).points()

    # Act
    w = solid_angle_weights(chi, chi_step=10, phi_step=5)

    # Assert
    # the weights integrate to the hemisphere, 2 pi, in units of the cell size
    assert (w.sum() * np.radians(10) * np.radians(5)) == pytest.approx(2 * np.pi, rel=0.01)


def test_grid_describe_default() -> None:
    # Arrange
    grid = Grid(10, 5)

    # Act
    text = grid.describe()

    # Assert
    assert text == "10 x 5 deg"


def test_equal_spacing_points() -> None:
    # Arrange
    eq = Grid(10, 5, equal_spacing=True)

    # Act
    chi, phi = eq.points()

    # Assert
    assert len(chi) == 454
    assert (chi[-1], phi[-1]) == (90, 0)


def test_equal_spacing_ring_steps() -> None:
    # Arrange
    eq = Grid(10, 5, equal_spacing=True)

    # Act
    steps = [eq.ring_phi_step(c) for c in eq.rings()]

    # Assert
    assert steps == [5, 5, 5, 6, 6, 8, 10, 15, 30]


def test_equal_spacing_steps_divide_360() -> None:
    # Arrange
    eq = Grid(10, 5, equal_spacing=True)

    # Act
    steps = [eq.ring_phi_step(c) for c in eq.rings()]

    # Assert
    assert all(360 % s == 0 and s >= 5 for s in steps)


def test_equal_spacing_keeps_arc_spacing() -> None:
    """The spacing on the sphere stays near the chi 0 spacing on every ring."""
    # Arrange
    eq = Grid(10, 5, equal_spacing=True)

    # Act
    ratios = [eq.ring_phi_step(c) * np.cos(np.radians(c)) / 5 for c in eq.rings()]

    # Assert
    assert all(0.8 < r < 1.25 for r in ratios)


def test_equal_spacing_weights_equal_pf_total() -> None:
    # Arrange
    pf, eq = Grid(10, 5), Grid(10, 5, equal_spacing=True)
    cell = np.radians(10) * np.radians(5)

    # Act
    total_eq = eq.weights(eq.points()[0]).sum() * cell

    # Assert
    assert total_eq == pytest.approx(pf.weights(pf.points()[0]).sum() * cell, rel=1e-9)


def test_equal_spacing_weights_cover_hemisphere() -> None:
    # Arrange
    eq = Grid(10, 5, equal_spacing=True)
    cell = np.radians(10) * np.radians(5)

    # Act
    total = eq.weights(eq.points()[0]).sum() * cell

    # Assert
    assert total == pytest.approx(2 * np.pi, rel=0.01)


def test_grid_weights_equal_solid_angle_weights() -> None:
    # Arrange
    pf = Grid(10, 5)
    chi = pf.points()[0]

    # Act
    w = pf.weights(chi)

    # Assert
    np.testing.assert_allclose(w, solid_angle_weights(chi, chi_step=10, phi_step=5))


def test_equal_spacing_describe() -> None:
    # Arrange
    eq = Grid(10, 5, equal_spacing=True)

    # Act
    text = eq.describe()

    # Assert
    assert "equal spacing" in text and "30 deg at chi 80" in text


def test_ring_phi_step_not_dividing_360_kept_at_chi0() -> None:
    # Arrange
    grid = Grid(10, 7, equal_spacing=True)

    # Act
    step = grid.ring_phi_step(0)

    # Assert
    assert step == 7
