"""
SampleMesh on the STL fixture shapes: PF's shape table, the indexed tracer against the brute-force one, and trimesh.

The fixtures are slab, c_channel, hex_tube, hollow_box_cavity, hollow_sphere, plate_windows_pocket, two_blocks and
brass cylinders (the STL fixtures). The trimesh cross-checks skip when trimesh is not installed.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pyrs.pole_figures.core.stl_mesh import SampleMesh
from tests.util.pole_figures_helpers import SLAB_SIZE_MM, STL_DIR, read_csv_rows, slab_mesh

pytestmark = pytest.mark.integration

FloatArray = npt.NDArray[np.float64]
SHAPES = sorted(p.name for p in STL_DIR.glob("*.stl"))
#: the shapes in PF's table (stl_shape_tests.csv)
PF_TABLE_SHAPES = ("plate_windows_pocket", "c_channel", "hollow_box_cavity", "two_blocks", "hex_tube", "hollow_sphere")


def unit(v: npt.ArrayLike) -> FloatArray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------- the reference slab
@pytest.fixture(scope="module")
def slab_stl() -> SampleMesh:
    return SampleMesh.from_stl(STL_DIR / "slab.stl")


def test_slab_stl_is_closed_without_warnings(slab_stl: SampleMesh) -> None:
    # Arrange
    m = slab_stl

    # Act
    warnings = m.warnings()

    # Assert
    assert m.closed
    assert warnings == []


def test_slab_stl_starts_at_origin(slab_stl: SampleMesh) -> None:
    # Arrange
    m = slab_stl

    # Act
    lo = m.lo

    # Assert
    np.testing.assert_allclose(lo, 0, atol=1e-6)


def test_slab_stl_size(slab_stl: SampleMesh) -> None:
    # Arrange
    m = slab_stl

    # Act
    size = m.size

    # Assert
    np.testing.assert_allclose(size, [35.4, 54.6, 3.3], rtol=1e-6)


def test_slab_stl_volume(slab_stl: SampleMesh) -> None:
    # Arrange
    m = slab_stl

    # Act
    volume = m.volume

    # Assert
    np.testing.assert_allclose(volume, 35.4 * 54.6 * 3.3, rtol=1e-5)


def test_in_memory_slab_matches_stl(slab_stl: SampleMesh) -> None:
    """The unit tests use slab_mesh() in place of the file: the same box, so the same paths."""
    # Arrange
    rng = np.random.default_rng(7)
    pts = rng.uniform(0.05, 0.95, (500, 3)) * np.array(SLAB_SIZE_MM)
    d = unit([0.31, -0.62, 0.72])

    # Act
    L_memory = slab_mesh().trace(pts, d)[1]

    # Assert
    np.testing.assert_allclose(L_memory, slab_stl.trace(pts, d)[1], atol=1e-5)


# ---------------------------------------------------------------- PF's shape table
@pytest.fixture(scope="module")
def pf_table() -> dict[str, dict[str, str]]:
    return {r["name"]: r for r in read_csv_rows(STL_DIR / "stl_shape_tests.csv")}


def test_pf_table_lists_the_shapes(pf_table: dict[str, dict[str, str]]) -> None:
    # Arrange
    table = pf_table

    # Act
    names = sorted(table)

    # Assert
    assert names == sorted(PF_TABLE_SHAPES)


@pytest.mark.parametrize("name", PF_TABLE_SHAPES)
def test_triangle_count_matches_pf(pf_table: dict[str, dict[str, str]], name: str) -> None:
    # Arrange
    row = pf_table[name]

    # Act
    m = SampleMesh.from_stl(STL_DIR / f"{name}.stl")

    # Assert
    assert m.ntri == int(row["ntri"])


@pytest.mark.parametrize("name", PF_TABLE_SHAPES)
def test_closed_flag_matches_pf(pf_table: dict[str, dict[str, str]], name: str) -> None:
    # Arrange
    row = pf_table[name]

    # Act
    m = SampleMesh.from_stl(STL_DIR / f"{name}.stl")

    # Assert
    assert m.closed == (row["closed"] == "True")


@pytest.mark.parametrize("name", PF_TABLE_SHAPES)
def test_volume_matches_pf(pf_table: dict[str, dict[str, str]], name: str) -> None:
    # Arrange
    row = pf_table[name]

    # Act
    m = SampleMesh.from_stl(STL_DIR / f"{name}.stl")

    # Assert
    np.testing.assert_allclose(m.volume, float(row["volume"]), rtol=1e-6)


# ---------------------------------------------------------------- ray segments
def test_ray_segments_sum_to_trace_in_c_channel() -> None:
    # Arrange
    m = SampleMesh.from_stl(STL_DIR / "c_channel.stl")
    rng = np.random.default_rng(1)
    pts = m.lo + rng.uniform(0, 1, (400, 3)) * m.size
    d = unit([0.2, -0.7, 0.4])
    inside = pts[m.contains(pts)]

    # Act
    _, L = m.trace(inside, d)

    # Assert
    for p, li in zip(inside[:50], L[:50], strict=True):
        assert sum(b - a for a, b in m.ray_segments(p, d)) == pytest.approx(li, abs=1e-9)


# ---------------------------------------------------------------- indexed tracer against PF's brute-force tracer
@pytest.fixture(scope="module", params=SHAPES)
def indexed_and_brute(request: pytest.FixtureRequest) -> list[tuple[FloatArray, FloatArray, FloatArray, FloatArray]]:
    """Winding and path from trace (spatial index) and trace_brute (PF's tracer) for 3000 points, 4 directions."""
    m = SampleMesh.from_stl(STL_DIR / request.param)
    rng = np.random.default_rng(3)
    p = (m.lo + m.hi) / 2 + rng.uniform(-0.45, 0.45, (3000, 3)) * m.size
    out = []
    for d in rng.normal(size=(4, 3)):
        d /= np.linalg.norm(d)
        w0, l0 = m.trace_brute(p, d)
        w1, l1 = m.trace(p, d)
        out.append((w0, w1, l0, l1))
    return out


def test_indexed_trace_winding_matches_pf(
    indexed_and_brute: list[tuple[FloatArray, FloatArray, FloatArray, FloatArray]],
) -> None:
    # Arrange
    traces = indexed_and_brute

    # Act
    pairs = [(w0, w1) for w0, w1, _, _ in traces]

    # Assert
    for w0, w1 in pairs:
        assert np.array_equal(w0, w1)


def test_indexed_trace_paths_match_pf(
    indexed_and_brute: list[tuple[FloatArray, FloatArray, FloatArray, FloatArray]],
) -> None:
    # Arrange
    traces = indexed_and_brute

    # Act
    pairs = [(l0, l1) for _, _, l0, l1 in traces]

    # Assert
    for l0, l1 in pairs:
        assert np.allclose(l0, l1, atol=1e-9)


# ---------------------------------------------------------------- trimesh cross-check
@pytest.fixture(scope="module", params=SHAPES)
def trimesh_case(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Inside flags and paths from SampleMesh and from trimesh for random points around one shape."""
    trimesh = pytest.importorskip("trimesh")
    name = request.param
    tm = trimesh.load(STL_DIR / name, force="mesh", process=False)
    m = SampleMesh.from_stl(STL_DIR / name)
    rng = np.random.default_rng(3)
    pts = m.lo - 1 + rng.uniform(0, 1, (2000, 3)) * (m.size + 2)
    inside = m.contains(pts)
    d = unit([0.31, -0.62, 0.72])
    p = pts[inside][:300]
    _, L = m.trace(p, d)
    loc, iray, itri = tm.ray.intersects_location(p, np.tile(d, (len(p), 1)), multiple_hits=True)
    t = np.einsum("ij,j->i", loc - p[iray], d)
    s = np.sign(tm.face_normals[itri] @ d)
    if (
        np.dot(np.cross(tm.triangles[0, 1] - tm.triangles[0, 0], tm.triangles[0, 2] - tm.triangles[0, 0]), [1, 1, 1])
        and tm.volume < 0
    ):
        s = -s
    L_tm = np.bincount(iray, weights=s * t, minlength=len(p))
    return {"inside": inside, "inside_tm": tm.contains(pts), "L": L, "L_tm": L_tm}


def test_contains_matches_trimesh(trimesh_case: dict[str, Any]) -> None:
    # Arrange
    case = trimesh_case

    # Act
    inside = case["inside"]

    # Assert
    np.testing.assert_array_equal(inside, case["inside_tm"])


def test_trimesh_check_has_enough_inside_points(trimesh_case: dict[str, Any]) -> None:
    # Arrange
    case = trimesh_case

    # Act
    n = case["inside"].sum()

    # Assert
    assert n > 50


def test_trace_matches_trimesh(trimesh_case: dict[str, Any]) -> None:
    # Arrange
    case = trimesh_case

    # Act
    L = case["L"]

    # Assert
    np.testing.assert_allclose(L, case["L_tm"], atol=1e-6)
