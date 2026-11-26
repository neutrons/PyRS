"""Table scans planned from a prediction (predict, then plan_scan): skipped points, count times, order and rows."""

from dataclasses import replace

import numpy as np
import pytest

from pyrs.pole_figures.core.geometry import Mount, MountConventions
from pyrs.pole_figures.core.prediction import Beams, Grid, Plane, PlanePrediction, Prediction, predict
from pyrs.pole_figures.core.table_scan import (
    COLUMNS,
    PlaneScan,
    count_time,
    phi_travel,
    plan_scan,
    run_title,
    table_rows,
)
from tests.util.pole_figures_helpers import SLAB_CENTER_MM, slab_mesh

pytestmark = pytest.mark.integration

CONV = MountConventions(incident_from="+vy")


@pytest.fixture(scope="module")
def pred() -> Prediction:
    return predict(slab_mesh(), center=SLAB_CENTER_MM, conv=CONV,
                   planes=[Plane.symmetric("200", 57.0), Plane("222", 112.0, 56.0)], beams=Beams(), mu=0.14,
                   grid=Grid(10, 15), n_points=2000)  # fmt: skip


@pytest.fixture(scope="module")
def q200(pred: Prediction) -> PlanePrediction:
    return pred.planes[0]


def plan(q: PlanePrediction, **kwargs: object) -> PlaneScan:
    """plan_scan with cutoff 3 and 10 s base unless overridden."""
    args: dict[str, object] = {"cutoff": 3.0, "base_s": 10.0, **kwargs}
    return plan_scan(q, **args)  # type: ignore[arg-type]


# ---------------------------------------------------------------- count times
def test_plan_scan_fixed_time(q200: PlanePrediction) -> None:
    # Arrange
    base_s = 12.0

    # Act
    fixed = plan(q200, base_s=base_s, scaled=False)

    # Assert
    assert {p.count_s for p in fixed.points} == {12}
    assert fixed.total_s == 12 * len(fixed.points)


def test_plan_scan_capped_time(q200: PlanePrediction) -> None:
    # Arrange
    cap = 20

    # Act
    capped = plan(q200, base_s=12.0, max_s=cap)

    # Assert
    assert max(p.count_s for p in capped.points) == 20
    assert capped.n_capped == sum(1 for p in capped.points if 12 * p.correction > 20.5)


def test_plan_scan_cap_below_one_second_raises(q200: PlanePrediction) -> None:
    # Arrange
    cap = 0.5

    # Act / Assert
    with pytest.raises(ValueError, match="cap"):
        plan(q200, base_s=12.0, max_s=cap)


def test_plan_scan_scales_time_by_correction(q200: PlanePrediction) -> None:
    # Arrange
    base_s = 10.0

    # Act
    sc = plan(q200, base_s=base_s)

    # Assert
    assert all(p.count_s == count_time(p.correction, base_s=base_s) for p in sc.points)
    assert sc.total_s == sum(p.count_s for p in sc.points)


# ---------------------------------------------------------------- skipped points
def test_plan_scan_skips_points_above_cutoff(q200: PlanePrediction) -> None:
    # Arrange
    keep = np.isfinite(q200.correction) & (q200.correction <= 3.0)

    # Act
    sc = plan(q200)

    # Assert
    assert len(sc.points) == keep.sum()
    assert sc.n_skipped == (~keep).sum() > 0
    assert all(p.correction <= 3.0 for p in sc.points)


def test_plan_scan_keeps_each_point_once(q200: PlanePrediction) -> None:
    # Arrange
    keep = np.isfinite(q200.correction) & (q200.correction <= 3.0)

    # Act
    sc = plan(q200)

    # Assert
    assert sorted((p.chi, p.phi) for p in sc.points) == sorted(zip(q200.chi[keep], q200.phi[keep], strict=True))


def test_plan_scan_higher_cutoff_keeps_more(q200: PlanePrediction) -> None:
    # Arrange
    low = plan(q200)

    # Act
    high = plan(q200, cutoff=10.0)

    # Assert
    assert len(high.points) >= len(low.points)


# ---------------------------------------------------------------- order and chi offset
def test_serpentine_cuts_phi_travel(q200: PlanePrediction) -> None:
    # Arrange
    plain = plan(q200, serpentine=False)

    # Act
    serp = plan(q200)

    # Assert
    assert phi_travel(serp) < 0.6 * phi_travel(plain)


def test_serpentine_alternates_phi_direction(q200: PlanePrediction) -> None:
    # Arrange
    serp = plan(q200)

    # Act
    rings: dict[float, list[float]] = {}
    for p in serp.points:
        rings.setdefault(p.chi, []).append(p.phi)
    dirs = [np.sign(np.diff(v)).mean() for v in rings.values() if len(v) > 1]

    # Assert
    assert len(dirs) >= 2  # at least two rings with several points, so the alternation is tested
    assert all(a * b < 0 for a, b in zip(dirs, dirs[1:], strict=False))  # up on one ring, down on the next


def test_plan_scan_chi_offset_writes_logged_chi(q200: PlanePrediction) -> None:
    """The mount's chi offset turns the predicted cradle chi into the logged chi of the table."""
    # Arrange
    conv = replace(CONV, chi_offset_deg=90.0)
    q = replace(q200, mount=Mount(conv, omega=q200.mount.omega, tth=q200.mount.tth))

    # Act
    sc = plan(q)

    # Assert
    assert min(p.chi for p in sc.points) == pytest.approx(-90.0)
    assert sc.chi_offset_deg == 90.0


# ---------------------------------------------------------------- rows
@pytest.fixture(scope="module")
def scan_222(pred: Prediction) -> tuple[PlaneScan, str, list]:
    sc = plan_scan(pred.planes[1], cutoff=3.0, base_s=12.0)
    title = run_title("IN718 plate", sc)
    return sc, title, table_rows(sc, title, "IN718 222")


def test_run_title_names_plane_and_angles(scan_222: tuple[PlaneScan, str, list]) -> None:
    # Arrange
    _, title, _ = scan_222

    # Act
    head = title[: len("IN718 plate | {222} pole figure | 2theta 112 omega 56")]

    # Assert
    assert head == "IN718 plate | {222} pole figure | 2theta 112 omega 56"


def test_table_rows_header(scan_222: tuple[PlaneScan, str, list]) -> None:
    # Arrange
    _, _, rows = scan_222

    # Act
    header = tuple(rows[0])

    # Assert
    assert header == COLUMNS


def test_table_rows_run_row(scan_222: tuple[PlaneScan, str, list]) -> None:
    # Arrange
    _, title, rows = scan_222

    # Act
    run_row = rows[1]

    # Assert
    assert run_row[:4] == [title, "IN718 222", 56.0, 112.0]
    assert run_row[4:] == [None] * 4


def test_table_rows_first_point(scan_222: tuple[PlaneScan, str, list]) -> None:
    # Arrange
    sc, _, rows = scan_222

    # Act
    first = rows[2]

    # Assert
    assert first[:4] == [None] * 4 and first[6] == "seconds"
    assert (first[4], first[5], first[7]) == (sc.points[0].chi, sc.points[0].phi, sc.points[0].count_s)


def test_table_rows_one_row_per_point(scan_222: tuple[PlaneScan, str, list]) -> None:
    # Arrange
    sc, _, rows = scan_222

    # Act
    n = len(rows)

    # Assert
    assert n == 2 + len(sc.points)
