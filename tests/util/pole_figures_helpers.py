"""
Shared test inputs: fixture paths, in-memory meshes, synthetic pole figures and reduced runs.

The pole-figure tests import this module. The fixture paths are relative to this file, so the tests run from any
checkout.
"""

import csv
import os
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import pytest

from pyrs.pole_figures.core.correction import Beams, GaugeCorrections
from pyrs.pole_figures.core.geometry import Mount, MountConventions
from pyrs.pole_figures.core.grid import Grid
from pyrs.pole_figures.core.polefig import PolePeak
from pyrs.pole_figures.core.runs import RunData
from pyrs.pole_figures.core.stl_mesh import SampleMesh, moller_trumbore

FloatArray = npt.NDArray[np.float64]

#: test inputs, all relative to this file so the tests run from any checkout
FIXTURES = Path(__file__).resolve().parents[1] / "data" / "pole_figures"
STL_DIR = FIXTURES / "stl"
REFERENCE = FIXTURES / "reference" / "slab_vx29.4vy19.75"  # the reference script's outputs for the parity tests
#: 13 sub-runs of the reduced run HB2B_6494.h5 (identifying text logs replaced), small enough for git, so CI reads a
#: real run
HB2B_SUBSET = FIXTURES / "hb2b" / "HB2B_6494_subset.h5"
#: the full reduced reference runs HB2B_6494.h5 to HB2B_6497.h5 (not in git): HIDRA_PF_TEST_DATA, else a local folder
TEST_DATA = Path(os.environ.get("HIDRA_PF_TEST_DATA", FIXTURES / "hb2b"))
needs_data = pytest.mark.skipif(not (TEST_DATA / "HB2B_6494.h5").exists(), reason="HB2B test files not present")

#: the reference sample (the slab STL): an axis-aligned box from the origin, vx x vy x vz (mm)
SLAB_SIZE_MM = (35.4, 54.6, 3.3)
#: the reference gauge center in that slab (pole_figure_config.json, sample.gauge_center_mm)
SLAB_CENTER_MM = (29.4, 19.75, 1.65)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """All rows of a CSV file as dicts (the file is closed before returning)."""
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def box_triangles(lo: npt.ArrayLike, hi: npt.ArrayLike, *, outward: bool = True) -> FloatArray:
    """
    The 12 triangles of an axis-aligned box.

    Args:
        lo: Lower corner (mm).
        hi: Upper corner (mm).
        outward: Wind the triangles with outward normals (default: True). False winds them inward, as for a cavity.

    Returns:
        Triangle vertices, shape (12, 3, 3).
    """
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], float)
    c = lo + c * (hi - lo)
    quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    tri = []
    for a, b, cc, d in quads:
        tri += [(a, b, cc), (a, cc, d)]
    T = c[np.array(tri)]
    return T if outward else T[:, ::-1]


def slab_mesh() -> SampleMesh:
    """The reference slab built in memory (no STL file): a closed box of ``SLAB_SIZE_MM`` from the origin."""
    return SampleMesh(box_triangles([0.0, 0.0, 0.0], SLAB_SIZE_MM), name="slab (in memory)")


def icosphere(radius: float, *, subdivisions: int = 4, outward: bool = True) -> FloatArray:
    """
    A closed triangulated sphere about the origin: an icosahedron with each face split in four ``subdivisions`` times.

    Every vertex lies on the sphere, so the faces lie inside it. Shared vertices are computed once, so edges match
    exactly and the mesh is closed and consistently wound.

    Args:
        radius: Sphere radius (mm).
        subdivisions: Number of 4-way splits (default: 4, 5120 faces).
        outward: Wind the triangles with outward normals (default: True). False winds them inward, as for a cavity.

    Returns:
        Triangle vertices, shape (20 * 4**subdivisions, 3, 3).
    """
    g = (1 + np.sqrt(5)) / 2
    corners = [(-1, g, 0), (1, g, 0), (-1, -g, 0), (1, -g, 0), (0, -1, g), (0, 1, g), (0, -1, -g), (0, 1, -g),
               (g, 0, -1), (g, 0, 1), (-g, 0, -1), (-g, 0, 1)]  # fmt: skip
    verts = [np.array(v, float) for v in corners]
    verts = [v / np.linalg.norm(v) for v in verts]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
             (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
             (8, 6, 7), (9, 8, 1)]  # fmt: skip
    for _ in range(subdivisions):
        cache: dict[tuple[int, int], int] = {}

        def mid(i: int, j: int, cache: dict[tuple[int, int], int] = cache) -> int:
            key = (min(i, j), max(i, j))
            if key not in cache:
                m = verts[i] + verts[j]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]

        new = []
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new
    V = np.array(verts) * radius
    T = V[np.array(faces)]
    normal = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    inward = np.einsum("ij,ij->i", normal, T.mean(axis=1)) < 0
    T[inward] = T[inward][:, ::-1]
    return T if outward else T[:, ::-1]


def synthetic_pole_figure(run: str = "1001", hkl: str = "200", chi_step: float = 30.0, phi_step: float = 30.0,
                          rate: float = 10.0, seed: int = 0) -> PolePeak:  # fmt: skip
    """A small pole figure on a chi/phi grid with smooth corrections (some above 3)."""
    chi, phi = Grid(chi_step, phi_step).points()
    rng = np.random.default_rng(seed)
    corr = 1 + 3 * np.sin(np.radians(chi)) ** 4 * np.abs(np.cos(np.radians(phi)))
    n = len(chi)
    m = Mount(MountConventions(incident_from="+vy"), omega=28.5, tth=57.0)
    return PolePeak(
        run=run, hkl=hkl, title=f"run {run}", two_theta=57.0, fwhm=0.5, wavelength_A=1.72,
        beams=Beams(), mount=m, chi=chi, phi=phi, count_s=np.full(n, 10.0),
        rate=rate * (1 + 0.1 * rng.standard_normal(n)), sigma=np.full(n, 0.5),
        gauge=GaugeCorrections(F=40 / corr, F_ref=40.0, volume_in_sample=np.full(n, 40.0), mean_path=4 * corr,
                               vy_offset=np.zeros(n), vy_rms=np.ones(n), correction=corr),
        cutoff=3.0, mu=0.14, method="window",
        h=np.array([m.pole(c, f) for c, f in zip(chi, phi, strict=True)]),
    )  # fmt: skip


def write_reduced_run(path: Path, n: int = 6, position: tuple[float, float, float] | None = None,
                      moving: float = 0.0, short_tail: bool = False, keys: tuple[str, ...] = ("vx", "vy", "vz"),
                      nan_subruns: int = 0) -> Path:  # fmt: skip
    """
    A minimal reduced HB2B file (logs, one Gaussian peak per sub-run) with optional vx, vy, vz logs.

    ``position`` None writes NaN for vx, vy, vz (the virtual frame was not set up). ``moving`` adds a ramp in vx (mm).
    ``short_tail`` adds a final 0.01 s sub-run far from the others (the scan going home), which read_run skips.
    ``keys`` are the position logs written. ``nan_subruns`` sets the first sub-runs' position to NaN.
    """
    x = np.linspace(80.0, 100.0, 200)
    m = n + int(short_tail)
    with h5py.File(path, "w") as h:
        logs = h.create_group("raw data/logs")
        logs["chi"] = np.linspace(0.0, 50.0, m)
        logs["phi"] = np.zeros(m)
        logs["omega"] = np.full(m, 45.0)
        logs["2theta"] = np.full(m, 90.0)
        logs["sub-run duration"] = np.r_[np.full(n, 30.0), [0.01] * int(short_tail)]
        logs["sub-runs"] = np.arange(1, m + 1, dtype=float)
        logs["ISR:X:Gap"] = np.full(m, 4.0)
        logs["ISR:Y:Gap"] = np.full(m, 4.0)
        pos = np.full((m, 3), np.nan) if position is None else np.tile(np.asarray(position, float), (m, 1))
        pos[:n, 0] += np.linspace(0.0, moving, n)
        pos[n:, 0] += 50.0  # the home position of the skipped tail
        pos[:nan_subruns] = np.nan
        for j, k in enumerate(("vx", "vy", "vz")):
            if k in keys:
                logs[k] = pos[:, j]
        h["reduced diffraction data/2theta"] = x[None]
        h["reduced diffraction data/main"] = np.tile(100.0 * np.exp(-0.5 * ((x - 90.0) / 0.5) ** 2) + 5.0, (m, 1))
        h["instrument/monochromator setting/wave length"] = np.array([1.54])
    return path


def synthetic_run_data(n: int = 6) -> RunData:
    """
    A reduced run built in memory: the same sub-runs as :func:`write_reduced_run` with its defaults, without a file.

    Sub-runs: chi 0 to 50 deg, phi 0, omega 45, arm 2theta 90, 30 s each, 4 x 4 mm slits, wavelength 1.54 A, one
    Gaussian peak at 2theta 90 (height 100, sigma 0.5 deg) on a background of 5 counts, no logged position.
    """
    x = np.linspace(80.0, 100.0, 200)
    peak = 100.0 * np.exp(-0.5 * ((x - 90.0) / 0.5) ** 2) + 5.0
    return RunData(
        run="2", path=Path("HB2B_2.h5"), title="", wavelength_A=1.54, chi=np.linspace(0.0, 50.0, n), phi=np.zeros(n),
        omega=np.full(n, 45.0), two_theta_arm=np.full(n, 90.0), count_s=np.full(n, 30.0),
        subruns=np.arange(1, n + 1, dtype=float), slit_x=np.full(n, 4.0), slit_y=np.full(n, 4.0), x=x,
        y=np.tile(peak, (n, 1)), live_skipped=0, short_skipped=0, invalid_skipped=0, position=np.full((n, 3), np.nan),
        notes=[], position_note="",
    )  # fmt: skip


def replace_dataset(path: Path, name: str, values: npt.ArrayLike) -> None:
    """Replace one dataset of an HDF5 file (to make a written reduced run bad in one way)."""
    with h5py.File(path, "a") as h:
        del h[name]
        h[name] = values


def trace_brute(mesh: SampleMesh, points: npt.ArrayLike, d: npt.ArrayLike, *,
                chunk: int = 2_000_000) -> tuple[FloatArray, FloatArray]:  # fmt: skip
    """
    The reference script's tracer: every point against every triangle in the ray bundle's footprint.

    ``SampleMesh.trace`` must give the same windings and paths with its spatial index.

    Args:
        mesh: The sample mesh.
        points: Ray start points, shape (n, 3) (mm).
        d: Ray direction, shape (3,). Any length: the path is in units of ``|d|``.
        chunk: Work-array size limit in points x triangles per step (default: 2,000,000).

    Returns:
        The sum of crossing signs per point (1 inside a closed mesh, 0 outside) and the material path ahead of each
        point.
    """
    p = np.atleast_2d(np.asarray(points, float))
    d = np.asarray(d, float)
    h = np.cross(d, mesh.e2)
    a = np.einsum("ij,ij->i", mesh.e1, h)
    use = np.abs(a) > 1e-12  # skip triangles parallel to the ray
    # skip triangles outside the footprint of the ray bundle (projected on the plane normal to d)
    b1 = np.cross(d, [1.0, 0, 0] if abs(d[0]) < 0.9 else [0, 1.0, 0])
    b1 /= np.linalg.norm(b1)
    B = np.c_[b1, np.cross(d / np.linalg.norm(d), b1)]
    pp = p @ B
    tv = np.stack([mesh.v0, mesh.v0 + mesh.e1, mesh.v0 + mesh.e2]) @ B  # (3, m, 2)
    tol = 1e-9 * (1 + np.abs(pp).max())
    use &= np.all((tv.max(axis=0) >= pp.min(axis=0) - tol) & (tv.min(axis=0) <= pp.max(axis=0) + tol), axis=1)
    v0, e1, e2, h, f = mesh.v0[use], mesh.e1[use], mesh.e2[use], h[use], 1.0 / a[use]
    s_t = np.sign(mesh.n[use] @ d)
    n = len(p)
    wind, length = np.zeros(n), np.zeros(n)
    step = max(1, chunk // max(1, n))
    for k in range(0, len(v0), step):
        sl = slice(k, k + step)
        m = len(v0[sl])

        def pairs(x: FloatArray, m: int = m) -> FloatArray:
            return np.broadcast_to(x, (n, *x.shape)).reshape(n * m, *x.shape[1:])

        s = (p[:, None, :] - v0[None, sl]).reshape(n * m, 3)
        t, hit = moller_trumbore(s, e1=pairs(e1[sl]), e2=pairs(e2[sl]), h=pairs(h[sl]), f=pairs(f[sl]), d=d)
        t, hit = t.reshape(n, m), hit.reshape(n, m)
        wind += (hit * s_t[sl]).sum(axis=1)
        length += (hit * s_t[sl] * t).sum(axis=1)
    return wind, length
