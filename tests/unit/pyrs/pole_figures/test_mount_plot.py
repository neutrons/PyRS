"""The mount check figure and the beam path from the gauge center."""

import numpy as np
import pytest
from matplotlib.figure import Figure

from pyrs.pole_figures.core.geometry import Mount, MountConventions
from pyrs.pole_figures.plots.mount import draw_mount, mount_figure, path_at_gauge
from tests.util.pole_figures_helpers import SLAB_CENTER_MM, slab_mesh

CONV = MountConventions(incident_from="+vy")


def test_path_at_gauge_symmetric_reflection() -> None:
    # Arrange: the slab at chi 0, phi 0 with omega = theta, gauge at mid thickness
    mount = Mount(CONV, omega=56.0, tth=112.0)

    # Act
    L_in, L_out = path_at_gauge(slab_mesh(), np.array(SLAB_CENTER_MM), mount=mount, chi=0.0, phi=0.0)

    # Assert: half the thickness / sin(theta) each way
    assert L_in == pytest.approx(SLAB_CENTER_MM[2] / np.sin(np.radians(56)))
    assert L_out == pytest.approx(L_in)


def test_path_at_gauge_center_outside_is_nan() -> None:
    # Arrange
    mount = Mount(CONV, omega=56.0, tth=112.0)

    # Act
    L_in, L_out = path_at_gauge(slab_mesh(), np.array([-5.0, -5.0, -5.0]), mount=mount, chi=0.0, phi=0.0)

    # Assert
    assert np.isnan(L_in) and np.isnan(L_out)


def test_mount_figure_has_three_panels() -> None:
    # Arrange
    mesh = slab_mesh()
    mount = Mount(CONV, omega=45.0, tth=90.0)

    # Act
    fig = mount_figure(mesh, center=SLAB_CENTER_MM, mount=mount, chi_logged=30.0, phi=120.0)

    # Assert
    assert len(fig.axes) == 3
    assert "chi 30 (logged)" in fig.axes[1].get_title(loc="left")


def test_draw_mount_redraw_clears_figure() -> None:
    # Arrange
    fig = Figure()
    mount = Mount(CONV, omega=45.0, tth=90.0)
    draw_mount(fig, slab_mesh(), center=SLAB_CENTER_MM, mount=mount, chi_logged=0.0, phi=0.0)

    # Act
    draw_mount(fig, slab_mesh(), center=SLAB_CENTER_MM, mount=mount, chi_logged=10.0, phi=0.0)

    # Assert
    assert len(fig.axes) == 3
    assert "chi 10 (logged)" in fig.axes[1].get_title(loc="left")
