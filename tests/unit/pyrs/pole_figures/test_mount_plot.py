"""The mount check figure (the beam path from the gauge center is tested with core.gauge)."""

import pytest
from matplotlib.figure import Figure

from pyrs.pole_figures.core.geometry import Mount, MountConventions
from pyrs.pole_figures.core.projection import ViewFrom
from pyrs.pole_figures.plots.mount import draw_mount, mount_figure
from tests.util.pole_figures_helpers import SLAB_CENTER_MM, slab_mesh

CONV = MountConventions(incident_from="+vy")


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


@pytest.mark.parametrize(("view", "label"), [("+vz", "+vx\n(phi axis)"), ("+vx", "vx\n(phi axis)")])
def test_mount_figure_key_uses_pole_figure_frame(view: ViewFrom, label: str) -> None:
    # Arrange: panel (c) is drawn by pf_frame, the same frame as every pole figure
    mount = Mount(CONV, omega=45.0, tth=90.0)

    # Act
    fig = mount_figure(slab_mesh(), center=SLAB_CENTER_MM, mount=mount, chi_logged=0.0, phi=0.0, view=view,
                       vz_name="BD")  # fmt: skip

    # Assert
    key = fig.axes[2]
    assert key.get_xlim() == pytest.approx((-1.18, 1.18))
    assert label in [t.get_text() for t in key.texts]
