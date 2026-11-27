"""Figures and table rows built from a prediction (predict, then plots and table-scan rows), and the mount check."""

import numpy as np
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure

from pyrs.pole_figures.core.correction import PARTS, CorrectionPart
from pyrs.pole_figures.core.geometry import Mount, MountConventions
from pyrs.pole_figures.core.prediction import Prediction
from pyrs.pole_figures.core.stl_mesh import SampleMesh
from pyrs.pole_figures.core.table_scan import PlaneScan, plan_scan
from pyrs.pole_figures.io.table_scan import COLUMNS, WAIT_SECONDS, run_title, table_rows
from pyrs.pole_figures.plots.correction_profile import correction_profile_figure, draw_correction_profile
from pyrs.pole_figures.plots.mount import mount_figure
from pyrs.pole_figures.plots.predicted import draw_plane, draw_predicted, plane_figure, predicted_figure
from pyrs.pole_figures.plots.style import TIME_CMAP
from pyrs.pole_figures.plots.table_scan import draw_table_scan, scan_times, table_scan_figure
from tests.util.pole_figures_helpers import STL_DIR, slab_prediction

pytestmark = pytest.mark.integration

CONV = MountConventions(incident_from="+vy")
ScanRows = tuple[PlaneScan, str, list[list[object]]]


@pytest.fixture(scope="module")
def pred() -> Prediction:
    return slab_prediction()


@pytest.fixture(scope="module")
def pred_offset() -> Prediction:
    """The same prediction on a goniometer whose chi zero is 5 deg off the cradle's."""
    return slab_prediction(chi_offset_deg=5.0)


@pytest.fixture(scope="module")
def scan_222(pred: Prediction) -> ScanRows:
    sc = plan_scan(pred.planes[1], cutoff=3.0, base_s=12.0)
    title = run_title("IN718 plate", sc)
    return sc, title, table_rows(sc, title=title, phase_label="IN718 222")


# ---------------------------------------------------------------- predicted figures
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


@pytest.mark.parametrize("draw", ["predicted", "plane", "profile", "table_scan"])
def test_draw_into_existing_figure_replaces_content(pred: Prediction, scan_222: ScanRows, draw: str) -> None:
    # Arrange: a canvas figure that already holds a plot (as in the GUI)
    fig = Figure()
    old = fig.add_subplot()
    old.plot([0, 1], [0, 1])

    # Act
    if draw == "predicted":
        draw_predicted(fig, pred, cutoff=3.0)
    elif draw == "plane":
        draw_plane(fig, pred, 1, cutoff=3.0)
    elif draw == "profile":
        draw_correction_profile(fig, pred, 0, cutoff=3.0)
    else:
        draw_table_scan(fig, pred, 1, scan=scan_222[0])

    # Assert: the old plot is gone, the new figure has its title
    assert old not in fig.axes and len(fig.axes) >= 2
    assert fig.get_suptitle() != ""


def test_correction_profile_figure_draws_parts(pred: Prediction) -> None:
    # Arrange
    parts: tuple[CorrectionPart, ...] = ("total", "volume")

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
        correction_profile_figure(pred, 0, cutoff=3.0, parts=parts)  # type: ignore[arg-type]


def test_correction_profile_figure_default_parts(pred: Prediction) -> None:
    # Arrange
    words = {"total": "total", "volume": "gauge volume", "attenuation": "attenuation"}

    # Act
    fig = correction_profile_figure(pred, 0, cutoff=3.0)

    # Assert
    title = fig.axes[0].get_title(loc="left")
    assert all(words[p] in title for p in PARTS)


def test_correction_profile_figure_rings_logged_chi(pred_offset: Prediction) -> None:
    # Arrange: cradle chi 0 to 80 and 90 on a goniometer with a 5 deg chi offset
    q = pred_offset.planes[0]
    want = [f"chi {c - 5.0:g}" for c in np.unique(np.round(q.chi, 3))]

    # Act
    fig = correction_profile_figure(pred_offset, 0, cutoff=3.0)

    # Assert
    legend = fig.axes[0].get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == want


def test_table_scan_figure_counts_points(pred: Prediction, scan_222: ScanRows) -> None:
    # Arrange
    sc, _, _ = scan_222

    # Act
    fig = table_scan_figure(pred, 1, scan=sc)

    # Assert
    assert f"{len(sc.points)} of {sc.n_grid} points kept" in fig.axes[0].get_title(loc="left")


def test_scan_times_chi_offset_match_grid(pred_offset: Prediction) -> None:
    # Arrange: the scan points hold the logged chi, the grid the cradle chi
    q = pred_offset.planes[1]
    sc = plan_scan(q, cutoff=3.0, base_s=12.0)
    assert sc.chi_offset_deg == 5.0 and sc.points[0].chi == q.chi[0] - 5.0

    # Act
    times = scan_times(q, sc)

    # Assert: every scan point found its grid cell
    assert np.isfinite(times).sum() == len(sc.points)
    assert sorted(times[np.isfinite(times)]) == sorted(p.count_s for p in sc.points)


def test_table_scan_figure_chi_offset_colors_count_times(pred_offset: Prediction) -> None:
    # Arrange
    q = pred_offset.planes[1]
    sc = plan_scan(q, cutoff=3.0, base_s=12.0)
    counts = np.array([p.count_s for p in sc.points])

    # Act
    fig = table_scan_figure(pred_offset, 1, scan=sc)

    # Assert: the kept cells span the colors of the shortest and the longest count time
    ax = fig.axes[0]
    (kept,) = [c for c in ax.collections if isinstance(c, PolyCollection) and not c.get_hatch()]
    colors = {tuple(np.round(c, 6)) for c in np.array(kept.get_facecolor(), dtype=float)}
    assert counts.max() > counts.min()
    assert tuple(np.round(TIME_CMAP(0.0), 6)) in colors and tuple(np.round(TIME_CMAP(1.0), 6)) in colors


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
