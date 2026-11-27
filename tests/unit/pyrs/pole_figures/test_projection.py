"""Equal-area projection of poles and the pole-figure views."""

import numpy as np
import pytest

from pyrs.pole_figures.core.projection import from_view, project, to_view


def test_project_radii() -> None:
    # Arrange
    h = [[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, -1]]

    # Act
    xy = project(h)

    # Assert
    np.testing.assert_allclose(np.hypot(*xy.T), [0, 1, 1, 0], atol=1e-12)


@pytest.mark.parametrize(
    ("h", "view", "want"),
    [([1, 0, 0], "+vz", [1, 0]), ([1, 0, 0], "-vz", [-1, 0]), ([0, 1, 0], "+vx", [1, 0])],
    ids=["plus_vz_vx_right", "minus_vz_vx_left", "plus_vx_vy_right"],
)
def test_project_view_orientation(h: list[float], view: str, want: list[float]) -> None:
    # Arrange
    vector = h

    # Act
    xy = project(vector, view=view)  # type: ignore[arg-type]

    # Assert
    np.testing.assert_allclose(xy[0], want, atol=1e-12)


def test_project_is_equal_area() -> None:
    # Arrange
    a = np.radians(45)  # 45 deg from the center

    # Act
    r = np.hypot(*project([np.sin(a), 0, np.cos(a)])[0])

    # Assert
    assert r == pytest.approx(np.sqrt(2) * np.sin(a / 2))


def test_from_view_inverts_to_view() -> None:
    # Arrange
    h = np.array([[0.2, 0.3, 0.9], [1.0, 0.0, 0.0]])

    # Act
    back = from_view(to_view(h, view="+vx"), view="+vx")

    # Assert
    np.testing.assert_allclose(back, h)


def test_project_unknown_view_raises() -> None:
    # Arrange
    view = "vz"

    # Act / Assert
    with pytest.raises(ValueError, match=r"view must be one of \+vz, -vz, \+vx, got 'vz'"):
        project([0.0, 0.0, 1.0], view=view)  # type: ignore[arg-type]
