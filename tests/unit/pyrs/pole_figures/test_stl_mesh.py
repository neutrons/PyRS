"""SampleMesh on in-memory meshes: STL reading, closed and winding checks, and ray paths against analytic chords."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from pyrs.pole_figures.core.stl_mesh import SampleMesh, read_stl
from tests.util.pole_figures_helpers import SLAB_SIZE_MM, box_triangles, icosphere, slab_mesh

FloatArray = npt.NDArray[np.float64]
#: per direction: winding and path of the inside points, the analytic path, winding of the outside points, direction
BoxRays = list[tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]]


def write_ascii_stl(path: Path, T: FloatArray) -> None:
    lines = ["solid test"]
    for t in T:
        lines += ["facet normal 0 0 0", "outer loop", *[f"vertex {v[0]} {v[1]} {v[2]}" for v in t], "endloop"]
        lines.append("endfacet")
    lines.append("endsolid test")
    path.write_text("\n".join(lines))


def write_binary_stl(path: Path, T: FloatArray) -> None:
    rec = np.zeros(len(T), np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["v"] = T
    path.write_bytes(b"\0" * 80 + np.uint32(len(T)).tobytes() + rec.tobytes())


def unit(v: npt.ArrayLike) -> FloatArray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def box_exit(p: FloatArray, d: FloatArray, lo: npt.ArrayLike, hi: npt.ArrayLike) -> float:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    with np.errstate(divide="ignore"):
        t = np.where(d > 0, (hi - p) / d, np.where(d < 0, (lo - p) / d, np.inf))
    return float(t.min())


def sagitta(mesh: SampleMesh, radius: float) -> float:
    """How far the faceted sphere sits inside the true sphere: radius minus the nearest face-plane distance."""
    n = mesh.n / np.linalg.norm(mesh.n, axis=1)[:, None]
    return float(radius - np.abs(np.einsum("ij,ij->i", n, mesh.v0)).min())


def chord(mesh: SampleMesh, p: FloatArray, d: FloatArray) -> float:
    """Material length along the whole line through p (p inside the material): path ahead plus path behind."""
    return float(mesh.trace(p[None], d)[1][0] + mesh.trace(p[None], -d)[1][0])


# ---------------------------------------------------------------- reading
@pytest.mark.parametrize(("writer", "rtol"), [(write_ascii_stl, 1e-7), (write_binary_stl, 1e-6)],
                         ids=["ascii", "binary"])  # fmt: skip
def test_read_stl_box(tmp_path: Path, writer, rtol: float) -> None:  # type: ignore[no-untyped-def]
    # Arrange
    T = box_triangles([0, 0, 0], [3, 4, 5])
    writer(tmp_path / "box.stl", T)

    # Act
    got = read_stl(tmp_path / "box.stl")

    # Assert
    np.testing.assert_allclose(got, T, rtol=rtol)


def test_read_stl_garbage_raises(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "x.stl"
    path.write_bytes(b"\x00\x01garbage that is neither format" * 3)

    # Act / Assert
    with pytest.raises(ValueError, match="not a valid binary or ASCII STL"):
        read_stl(path)


def test_warnings_mesh_in_meters_says_check_units() -> None:
    # Arrange
    small = SampleMesh(slab_mesh().triangles() / 1000.0)  # the slab in meters

    # Act
    warnings = small.warnings()

    # Assert
    assert any("check the units" in w for w in warnings)


# ---------------------------------------------------------------- winding and closure
def test_inward_winding_keeps_volume() -> None:
    # Arrange
    tri = box_triangles([0, 0, 0], [2, 3, 4], outward=False)

    # Act
    inn = SampleMesh(tri)

    # Assert
    assert inn.volume == pytest.approx(24.0) == SampleMesh(box_triangles([0, 0, 0], [2, 3, 4])).volume


def test_inward_winding_keeps_paths() -> None:
    # Arrange
    out = SampleMesh(box_triangles([0, 0, 0], [2, 3, 4]))
    inn = SampleMesh(box_triangles([0, 0, 0], [2, 3, 4], outward=False))
    p = np.array([[0.7, 1.1, 1.3]])
    d = unit([0.21, 0.17, 0.9])  # exits inside the top face, not on an edge or diagonal (an edge hit counts twice)

    # Act
    path_in = inn.trace(p, d)[1][0]

    # Assert
    assert path_in == pytest.approx(out.trace(p, d)[1][0])


def test_trace_box_exit_is_analytic() -> None:
    # Arrange
    out = SampleMesh(box_triangles([0, 0, 0], [2, 3, 4]))
    p = np.array([[0.7, 1.1, 1.3]])
    d = unit([0.21, 0.17, 0.9])

    # Act
    path = out.trace(p, d)[1][0]

    # Assert
    assert path == pytest.approx(box_exit(p[0], d, [0, 0, 0], [2, 3, 4]))


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_init_non_finite_vertex_raises(value: float) -> None:
    # Arrange
    tri = box_triangles([0, 0, 0], [1, 1, 1])
    tri[0, 0, 0] = value

    # Act / Assert
    with pytest.raises(ValueError, match="non-finite vertex"):
        SampleMesh(tri)


def test_open_mesh_is_not_closed() -> None:
    # Arrange
    tri = box_triangles([0, 0, 0], [1, 1, 1])[:-1]

    # Act
    m = SampleMesh(tri)

    # Assert
    assert not m.closed


def test_open_mesh_warns() -> None:
    # Arrange
    m = SampleMesh(box_triangles([0, 0, 0], [1, 1, 1])[:-1])

    # Act
    warnings = m.warnings()

    # Assert
    assert any("not closed" in w for w in warnings)


# ---------------------------------------------------------------- box paths
@pytest.fixture(scope="module")
def box_rays() -> BoxRays:
    """Traces in a slab-sized box from 500 inside and 200 outside points along 20 random directions."""
    lo, hi = np.array([0.0, 0, 0]), np.array(SLAB_SIZE_MM)
    m = SampleMesh(box_triangles(lo, hi))
    rng = np.random.default_rng(0)
    p_in = lo + rng.uniform(0.01, 0.99, (500, 3)) * (hi - lo)
    p_out = hi + rng.uniform(0.5, 5.0, (200, 3))
    out = []
    for _ in range(20):
        d = unit(rng.normal(size=3))
        wind, L = m.trace(p_in, d)
        want = np.array([box_exit(p, d, lo, hi) for p in p_in])
        out.append((wind, L, want, m.trace(p_out, d)[0], d))
    return out


def test_trace_box_inside_points_wind_once(box_rays: BoxRays) -> None:
    # Arrange
    rays = box_rays

    # Act
    winds = [wind for wind, *_ in rays]

    # Assert
    for wind in winds:
        np.testing.assert_array_equal(wind, 1)


def test_trace_box_paths_match_analytic(box_rays: BoxRays) -> None:
    # Arrange
    rays = box_rays

    # Act
    pairs = [(L, want) for _, L, want, _, _ in rays]

    # Assert
    for L, want in pairs:
        np.testing.assert_allclose(L, want, rtol=1e-9, atol=1e-9)


def test_trace_box_outside_points_do_not_wind(box_rays: BoxRays) -> None:
    # Arrange
    rays = box_rays

    # Act
    winds = [wind_out for *_, wind_out, _ in rays]

    # Assert
    for wind in winds:
        np.testing.assert_array_equal(wind, 0)


# ---------------------------------------------------------------- box with a cavity
@pytest.fixture(scope="module")
def cavity_box() -> SampleMesh:
    """20 mm box with a closed 10 mm cavity in the middle (inner faces wound inward)."""
    return SampleMesh(
        np.concatenate([box_triangles([0, 0, 0], [20, 20, 20]), box_triangles([5, 5, 5], [15, 15, 15], outward=False)])
    )


#: a point in the wall below the cavity, looking up through it
CAVITY_P = np.array([[10.2, 9.7, 2.0]])
CAVITY_D = unit([0.001, 0.002, 1.0])


def test_cavity_box_is_closed(cavity_box: SampleMesh) -> None:
    # Arrange
    m = cavity_box

    # Act
    closed = m.closed

    # Assert
    assert closed


def test_cavity_box_volume(cavity_box: SampleMesh) -> None:
    # Arrange
    m = cavity_box

    # Act
    volume = m.volume

    # Assert
    assert volume == pytest.approx(8000 - 1000)


def test_trace_skips_the_cavity(cavity_box: SampleMesh) -> None:
    # Arrange
    m = cavity_box

    # Act
    _, L = m.trace(CAVITY_P, CAVITY_D)

    # Assert
    assert L[0] == pytest.approx((20 - 2.0 - 10.0) / CAVITY_D[2], rel=1e-9)  # 3 mm + (skip 10) + 5 mm


def test_contains_point_in_cavity_is_false(cavity_box: SampleMesh) -> None:
    # Arrange
    p = [[10.1, 9.9, 10.2]]

    # Act
    inside = cavity_box.contains(p)

    # Assert
    assert not inside[0]


def test_ray_segments_two_walls(cavity_box: SampleMesh) -> None:
    # Arrange
    m = cavity_box

    # Act
    segs = m.ray_segments(CAVITY_P[0], CAVITY_D)

    # Assert
    assert len(segs) == 2


def test_ray_segments_sum_to_trace_in_cavity_box(cavity_box: SampleMesh) -> None:
    # Arrange
    m = cavity_box

    # Act
    segs = m.ray_segments(CAVITY_P[0], CAVITY_D)

    # Assert
    assert sum(b - a for a, b in segs) == pytest.approx(m.trace(CAVITY_P, CAVITY_D)[1][0])


# ---------------------------------------------------------------- analytic sphere chords
# Icosphere of radius R (subdivided 4 times, 5120 faces, vertices on the sphere). Each face lies inside the sphere by
# at most the sagitta s = R - (nearest face-plane distance), about 1.1e-3 R. A chord through the faceted sphere is
# shorter than the true one by at most s / cos(i) at each end, i the angle of incidence. The directions are irrational
# so the rays do not meet a mesh edge or vertex.
R = 10.0
D = unit([0.3141, 0.2718, 0.9])


@pytest.fixture(scope="module")
def sphere() -> SampleMesh:
    return SampleMesh(icosphere(R), name="icosphere")


def test_icosphere_is_closed_and_consistent(sphere: SampleMesh) -> None:
    # Arrange
    m = sphere

    # Act
    ok = (m.closed, m.consistent)

    # Assert
    assert ok == (True, True)


def test_trace_from_sphere_center_is_radius(sphere: SampleMesh) -> None:
    # Arrange
    tol = sagitta(sphere, R)  # normal incidence: the path is short by at most the sagitta

    # Act
    path = sphere.trace(np.zeros((1, 3)), D)[1][0]

    # Assert
    assert R - tol - 1e-9 <= path <= R + 1e-9


@pytest.mark.parametrize("b", [3.0, 6.0, 8.5])
def test_trace_sphere_chord_at_impact_parameter(sphere: SampleMesh, b: float) -> None:
    """A line at distance b from the center crosses the sphere along 2 sqrt(R^2 - b^2)."""
    # Arrange
    p = b * unit(np.cross(D, [1.0, 0.0, 0.0]))  # the closest point to the center, inside the sphere
    cos_i = np.sqrt(1 - (b / R) ** 2)
    want = 2 * np.sqrt(R**2 - b**2)
    tol = 2 * sagitta(sphere, R) / cos_i

    # Act
    got = chord(sphere, p, D)

    # Assert
    assert want - tol - 1e-9 <= got <= want + 1e-9


@pytest.fixture(scope="module")
def hollow_sphere() -> tuple[SampleMesh, float, float]:
    """A shell: outer radius 15, inner radius 9 (the cavity wound inward)."""
    r_out, r_in = 15.0, 9.0
    tri = np.concatenate([icosphere(r_out), icosphere(r_in, outward=False)])
    return SampleMesh(tri, name="hollow icosphere"), r_out, r_in


def test_hollow_icosphere_volume(hollow_sphere: tuple[SampleMesh, float, float]) -> None:
    # Arrange
    m, r_out, r_in = hollow_sphere
    want = 4 / 3 * np.pi * (r_out**3 - r_in**3)

    # Act
    volume = m.volume

    # Assert
    assert volume == pytest.approx(want, rel=5e-3)  # faceting: both spheres are about 0.2 % small


@pytest.mark.parametrize("b", [0.0, 4.0, 8.0])
def test_trace_hollow_sphere_chord_skips_cavity(hollow_sphere: tuple[SampleMesh, float, float], b: float) -> None:
    """A line at distance b < r_in crosses the shell along 2 (sqrt(R^2 - b^2) - sqrt(r^2 - b^2))."""
    # Arrange
    m, r_out, r_in = hollow_sphere
    e = unit(np.cross(D, [1.0, 0.0, 0.0]))
    t_mid = (np.sqrt(r_out**2 - b**2) + np.sqrt(r_in**2 - b**2)) / 2
    p = b * e + t_mid * D  # in the shell wall, on the line
    want = 2 * (np.sqrt(r_out**2 - b**2) - np.sqrt(r_in**2 - b**2))
    # the outer faces shorten the chord, the inner faces lengthen it (the cavity is smaller)
    tol_out = 2 * sagitta(SampleMesh(icosphere(r_out)), r_out) / np.sqrt(1 - (b / r_out) ** 2)
    tol_in = 2 * sagitta(SampleMesh(icosphere(r_in)), r_in) / np.sqrt(1 - (b / r_in) ** 2)

    # Act
    got = chord(m, p, D)

    # Assert
    assert want - tol_out - 1e-9 <= got <= want + tol_in + 1e-9
