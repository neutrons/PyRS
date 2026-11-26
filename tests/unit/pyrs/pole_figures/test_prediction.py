"""Predicted corrections on the in-memory slab: grids, weights, free phi, progress and the input checks."""

import numpy as np
import pytest

from pyrs.pole_figures.core.geometry import Mount, MountConventions
from pyrs.pole_figures.core.prediction import (
    Beams,
    Grid,
    Plane,
    Prediction,
    phi_is_free,
    predict,
    solid_angle_weights,
)
from pyrs.pole_figures.core.stl_mesh import SampleMesh
from tests.util.pole_figures_helpers import SLAB_CENTER_MM, SLAB_SIZE_MM, box_triangles, slab_mesh

CONV = MountConventions(incident_from="+vy")


@pytest.fixture(scope="module")
def slab() -> SampleMesh:
    return slab_mesh()


def predict_200(mesh: SampleMesh, **kwargs: object) -> Prediction:
    """A quick {200} prediction on a 30 x 30 deg grid with 500 Monte Carlo points."""
    return predict(mesh, center=SLAB_CENTER_MM, conv=CONV, planes=[Plane.symmetric("200", 57.0)], beams=Beams(),
                   mu=0.14, grid=Grid(30, 30), n_points=500, **kwargs)  # type: ignore[arg-type]  # fmt: skip


# ---------------------------------------------------------------- grids and weights
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


# ---------------------------------------------------------------- free phi
@pytest.mark.parametrize(
    ("omega", "tth", "chi", "free"),
    [(28.5, 57.0, 90.0, True), (28.5, 57.0, 80.0, False), (25.0, 57.0, 90.0, False), (42.86, 85.71, 90.0, True)],
    ids=["symmetric_chi90", "symmetric_chi80", "tilted_q_chi90", "omega_rounded_chi90"],
)
def test_phi_is_free(omega: float, tth: float, chi: float, free: bool) -> None:
    """Phi is free where the phi axis is along Q: chi 90 with omega = theta (to 0.01 deg)."""
    # Arrange
    m = Mount(CONV, omega=omega, tth=tth)

    # Act
    got = phi_is_free(m, chi)

    # Assert
    assert got is free


@pytest.fixture(scope="module")
def free_phi_predictions(slab: SampleMesh) -> tuple[Prediction, Prediction]:
    """PF's default (phi 0 where phi is free) and the best-phi option, for a symmetric and a tilted {200}."""
    args = {"center": SLAB_CENTER_MM, "conv": CONV, "planes": [Plane("200", 57.0, 28.5), Plane("200", 57.0, 25.0)],
            "beams": Beams(), "mu": 0.14, "grid": Grid(30, 30), "n_points": 3000}  # fmt: skip
    return predict(slab, **args), predict(slab, **args, best_phi=True)  # type: ignore[arg-type]


def test_default_keeps_phi0_where_phi_is_free(free_phi_predictions: tuple[Prediction, Prediction]) -> None:
    # Arrange
    pf, _ = free_phi_predictions

    # Act
    q0 = pf.planes[0]

    # Assert
    assert q0.phi[-1] == 0 and not q0.phi_chosen.any()  # PF default unchanged


def test_best_phi_chooses_only_the_free_point(free_phi_predictions: tuple[Prediction, Prediction]) -> None:
    # Arrange
    _, best = free_phi_predictions

    # Act
    q1 = best.planes[0]

    # Assert
    assert q1.phi_chosen[-1] and q1.phi_chosen.sum() == 1


def test_best_phi_takes_the_shortest_sweep_path(free_phi_predictions: tuple[Prediction, Prediction]) -> None:
    # Arrange
    q1 = free_phi_predictions[1].planes[0]

    # Act
    trial, paths = q1.phi_sweeps[len(q1.chi) - 1]

    # Assert
    assert len(trial) == 12
    assert q1.mean_path[-1] == pytest.approx(np.nanmin(paths), rel=0.011)


def test_best_phi_tie_takes_the_lower_phi(free_phi_predictions: tuple[Prediction, Prediction]) -> None:
    # Arrange
    q1 = free_phi_predictions[1].planes[0]

    # Act
    phi = q1.phi[-1]

    # Assert
    assert phi == 90.0  # 90 and 270 tie on the plate; the lower phi wins


def test_best_phi_shortens_path_and_correction(free_phi_predictions: tuple[Prediction, Prediction]) -> None:
    # Arrange
    pf, best = free_phi_predictions

    # Act
    q0, q1 = pf.planes[0], best.planes[0]

    # Assert
    assert q1.mean_path[-1] < q0.mean_path[-1]
    assert q1.correction[-1] < q0.correction[-1]


def test_best_phi_keeps_the_pole(free_phi_predictions: tuple[Prediction, Prediction]) -> None:
    # Arrange
    pf, best = free_phi_predictions

    # Act
    p0, p1 = pf.planes[0].poles()[-1], best.planes[0].poles()[-1]

    # Assert
    np.testing.assert_allclose(p1, p0, atol=1e-12)


def test_best_phi_leaves_other_points(free_phi_predictions: tuple[Prediction, Prediction]) -> None:
    # Arrange
    pf, best = free_phi_predictions

    # Act
    c0, c1 = pf.planes[0].correction[:-1], best.planes[0].correction[:-1]

    # Assert
    np.testing.assert_array_equal(c1, c0)


def test_best_phi_ignores_tilted_plane(free_phi_predictions: tuple[Prediction, Prediction]) -> None:
    # Arrange
    _, best = free_phi_predictions

    # Act
    chosen = best.planes[1].phi_chosen

    # Assert
    assert not chosen.any()  # tilted plane: nothing to choose


# ---------------------------------------------------------------- predict
@pytest.fixture(scope="module")
def progress_run(slab: SampleMesh) -> tuple[Prediction, list[float]]:
    seen: list[float] = []
    res = predict(slab, center=SLAB_CENTER_MM, conv=CONV, planes=[Plane.symmetric("200", 57.0)], beams=Beams(),
                  mu=0.14, grid=Grid(30, 30), n_points=2000, progress=lambda f, _m: seen.append(f))  # fmt: skip
    return res, seen


def test_predict_reports_progress_to_one(progress_run: tuple[Prediction, list[float]]) -> None:
    # Arrange
    _, seen = progress_run

    # Act
    last = seen[-1]

    # Assert
    assert last == pytest.approx(1.0)


def test_predict_reference_correction_is_one(progress_run: tuple[Prediction, list[float]]) -> None:
    # Arrange
    res, _ = progress_run

    # Act
    planes = res.planes

    # Assert
    assert len(planes) == 1
    assert planes[0].correction[0] == pytest.approx(1.0)


def test_predict_poles_one_per_point(progress_run: tuple[Prediction, list[float]]) -> None:
    # Arrange
    q = progress_run[0].planes[0]

    # Act
    poles = q.poles()

    # Assert
    assert poles.shape == (len(q.chi), 3)


def test_predict_without_planes_raises(slab: SampleMesh) -> None:
    # Arrange
    planes: list[Plane] = []

    # Act / Assert
    with pytest.raises(ValueError, match="no planes"):
        predict(slab, center=SLAB_CENTER_MM, conv=MountConventions(), planes=planes, beams=Beams(), mu=0.14)


def test_predict_gauge_outside_sample_raises(slab: SampleMesh) -> None:
    # Arrange
    center = [-100.0, 0.0, 0.0]

    # Act / Assert
    with pytest.raises(ValueError, match="holds no sample at chi 0, phi 0"):
        predict(slab, center=center, conv=CONV, planes=[Plane.symmetric("200", 57.0)], beams=Beams(), mu=0.14,
                grid=Grid(30, 30), n_points=500)  # fmt: skip


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_beams_non_positive_size_raises(bad: float) -> None:
    # Arrange
    width = bad

    # Act / Assert
    with pytest.raises(ValueError, match="positive size"):
        Beams(incident_width=width)


# ---------------------------------------------------------------- open and inconsistent meshes
@pytest.fixture
def open_slab() -> SampleMesh:
    """The slab with one triangle of its +vx face missing (an open mesh)."""
    tri = box_triangles([0.0, 0.0, 0.0], SLAB_SIZE_MM)
    return SampleMesh(tri[:-1], name="open slab")


@pytest.fixture
def flipped_slab(slab: SampleMesh) -> SampleMesh:
    tri = slab.triangles().copy()
    tri[0] = tri[0][[0, 2, 1]]  # one triangle turned inward: still "closed", but the crossing signs are wrong
    return SampleMesh(tri, name="flipped")


def test_one_flipped_triangle_is_inconsistent(flipped_slab: SampleMesh) -> None:
    # Arrange
    mesh = flipped_slab

    # Act
    warnings = mesh.warnings()

    # Assert
    assert mesh.closed and not mesh.consistent
    assert "not all wound the same way" in warnings[0]


def test_predict_inconsistent_winding_raises(flipped_slab: SampleMesh) -> None:
    # Arrange
    mesh = flipped_slab

    # Act / Assert
    with pytest.raises(ValueError, match="allow_open_mesh"):
        predict_200(mesh)


def test_predict_inconsistent_winding_allowed(flipped_slab: SampleMesh) -> None:
    # Arrange
    mesh = flipped_slab

    # Act
    res = predict_200(mesh, allow_open_mesh=True)

    # Assert
    assert len(res.planes) == 1


def test_predict_open_mesh_raises(open_slab: SampleMesh) -> None:
    # Arrange
    mesh = open_slab

    # Act / Assert
    with pytest.raises(ValueError, match="not closed.*allow_open_mesh"):
        predict_200(mesh)


def test_predict_open_mesh_allowed(open_slab: SampleMesh) -> None:
    # Arrange
    mesh = open_slab

    # Act
    res = predict_200(mesh, allow_open_mesh=True)

    # Assert
    assert len(res.planes) == 1
    assert res.planes[0].correction[0] == pytest.approx(1.0)
