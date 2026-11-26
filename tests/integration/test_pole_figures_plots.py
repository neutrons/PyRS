"""Figures and table rows built from a prediction (predict, then plots and table-scan rows), and the mount check."""

from dataclasses import replace

import numpy as np
import pytest

from pyrs.pole_figures.core.correction import Beams
from pyrs.pole_figures.core.geometry import Mount, MountConventions
from pyrs.pole_figures.core.grid import Grid
from pyrs.pole_figures.core.prediction import Plane, Prediction, predict
from pyrs.pole_figures.core.stl_mesh import SampleMesh
from pyrs.pole_figures.core.table_scan import PlaneScan, plan_scan
from pyrs.pole_figures.io.table_scan import COLUMNS, WAIT_SECONDS, run_title, table_rows
from pyrs.pole_figures.plots.mount import mount_figure
from pyrs.pole_figures.plots.predicted import (
    PARTS,
    correction_parts,
    correction_profile_figure,
    plane_figure,
    predicted_figure,
)
from pyrs.pole_figures.plots.table_scan import table_scan_figure
from tests.util.pole_figures_helpers import SLAB_CENTER_MM, STL_DIR, slab_mesh

pytestmark = pytest.mark.integration

CONV = MountConventions(incident_from="+vy")
ScanRows = tuple[PlaneScan, str, list[list[object]]]


@pytest.fixture(scope="module")
def pred() -> Prediction:
    return predict(slab_mesh(), center=SLAB_CENTER_MM, conv=CONV,
                   planes=[Plane.symmetric("200", 57.0), Plane("222", 112.0, 56.0)], beams=Beams(), mu=0.14,
                   grid=Grid(10, 15), n_points=2000)  # fmt: skip


@pytest.fixture(scope="module")
def scan_222(pred: Prediction) -> ScanRows:
    sc = plan_scan(pred.planes[1], cutoff=3.0, base_s=12.0)
    title = run_title("IN718 plate", sc)
    return sc, title, table_rows(sc, title=title, phase_label="IN718 222")


# ---------------------------------------------------------------- predicted figures
def test_correction_parts_multiply_to_total(pred: Prediction) -> None:
    # Arrange
    q = pred.planes[0]

    # Act
    parts = correction_parts(q)

    # Assert
    ok = np.isfinite(parts["total"])
    np.testing.assert_allclose(parts["volume"][ok] * parts["attenuation"][ok], parts["total"][ok], rtol=1e-10)
    assert parts["volume"][0] == pytest.approx(1.0) and parts["attenuation"][0] == pytest.approx(1.0)


def test_correction_parts_without_chi_zero_raises(pred: Prediction) -> None:
    # Arrange: the grid moved off chi 0
    q = replace(pred.planes[0], chi=pred.planes[0].chi + 5.0)

    # Act / Assert
    with pytest.raises(ValueError, match="no grid point at chi 0, phi 0"):
        correction_parts(q)


def test_predicted_figure_two_rows_per_plane(pred: Prediction) -> None:
    # Arrange
    n_planes = len(pred.planes)

    # Act
    fig = predicted_figure(pred, cutoff=3.0)

    # Assert: path and correction per plane, plus two color bars
    assert len(fig.axes) == 2 * n_planes + 2
    assert "ignore > 3" in "".join(t.get_text() for t in fig.axes[-1].texts)


def test_plane_figure_title_names_plane(pred: Prediction) -> None:
    # Arrange
    index = 1

    # Act
    fig = plane_figure(pred, index, cutoff=2.0, view="+vx")

    # Assert
    assert len(fig.axes) == 4
    assert fig.get_suptitle().startswith("Predicted {222}")


def test_correction_profile_figure_draws_parts(pred: Prediction) -> None:
    # Arrange
    parts = ("total", "volume")

    # Act
    fig = correction_profile_figure(pred, 0, cutoff=3.0, parts=parts)

    # Assert
    title = fig.axes[0].get_title(loc="left")
    assert "total" in title and "gauge volume" in title and "attenuation" not in title


def test_correction_profile_figure_unknown_part_raises(pred: Prediction) -> None:
    # Arrange
    parts = ("total", "absorption")

    # Act / Assert
    with pytest.raises(ValueError, match="absorption"):
        correction_profile_figure(pred, 0, cutoff=3.0, parts=parts)


def test_correction_profile_figure_default_parts(pred: Prediction) -> None:
    # Arrange
    words = {"total": "total", "volume": "gauge volume", "attenuation": "attenuation"}

    # Act
    fig = correction_profile_figure(pred, 0, cutoff=3.0)

    # Assert
    title = fig.axes[0].get_title(loc="left")
    assert all(words[p] in title for p in PARTS)


def test_table_scan_figure_counts_points(pred: Prediction, scan_222: ScanRows) -> None:
    # Arrange
    sc, _, _ = scan_222

    # Act
    fig = table_scan_figure(pred, 1, scan=sc)

    # Assert
    assert f"{len(sc.points)} of {sc.n_grid} points kept" in fig.axes[0].get_title(loc="left")


# ---------------------------------------------------------------- table-scan rows
def test_run_title_names_plane_and_angles(scan_222: ScanRows) -> None:
    # Arrange
    _, title, _ = scan_222

    # Act
    head = title[: len("IN718 plate | {222} pole figure | 2theta 112 omega 56")]

    # Assert
    assert head == "IN718 plate | {222} pole figure | 2theta 112 omega 56"


def test_table_rows_header(scan_222: ScanRows) -> None:
    # Arrange
    _, _, rows = scan_222

    # Act
    header = tuple(rows[0])

    # Assert
    assert header == COLUMNS


def test_table_rows_run_row(scan_222: ScanRows) -> None:
    # Arrange
    _, title, rows = scan_222

    # Act
    run_row = rows[1]

    # Assert
    assert run_row[:4] == [title, "IN718 222", 56.0, 112.0]
    assert run_row[4:] == [None] * 4


def test_table_rows_first_point(scan_222: ScanRows) -> None:
    # Arrange
    sc, _, rows = scan_222

    # Act
    first = rows[2]

    # Assert
    assert first[:4] == [None] * 4 and first[6] == WAIT_SECONDS
    assert (first[4], first[5], first[7]) == (sc.points[0].chi, sc.points[0].phi, sc.points[0].count_s)


def test_table_rows_one_row_per_point(scan_222: ScanRows) -> None:
    # Arrange
    sc, _, rows = scan_222

    # Act
    n = len(rows)

    # Assert
    assert n == 2 + len(sc.points)


# ---------------------------------------------------------------- mount check on the STL fixtures
@pytest.mark.parametrize("name", sorted(p.name for p in STL_DIR.glob("*.stl")))
def test_mount_figure_renders_every_shape(name: str) -> None:
    # Arrange
    mesh = SampleMesh.from_stl(STL_DIR / name)
    mount = Mount(CONV, omega=45, tth=90)

    # Act
    fig = mount_figure(mesh, center=(mesh.lo + mesh.hi) / 2, mount=mount, chi_logged=30, phi=120)
    fig.canvas.draw()

    # Assert
    assert len(fig.axes) == 3
