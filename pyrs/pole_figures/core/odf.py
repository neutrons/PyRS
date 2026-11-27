"""
ODF from the corrected pole figures with MTEX for Python (pymtex).

pymtex (``pip install mtex``, mtex-toolbox/pymtex) is an optional dependency, installed separately where the ODF step
is used: :func:`pymtex_installed` says whether it is there, and only :func:`compute` imports it. The steps are those
of the reference script's ODF step: read ``pole_figure_<run>_<hkl>.csv``, keep the rows with a finite
intensity and correction (and ``ignored == 0``), take the specimen directions from pole_vx, pole_vy, pole_vz and the
hkl from the file name, fit the ODF with a de la Vallee Poussin kernel (halfwidth and resolution 5 deg, no sample
symmetry), then the RP errors, the ODF maximum, the texture index and entropy, the color ranges and every grid the
plots and plot-data CSVs need. No plotting here: ``plots/odf.py`` draws from :class:`OdfResult`.

Against the stored reference values (made once with MTEX 7.2.0 on 4 pole figures of an fcc alloy), pymtex 1.0.0
gives RP errors equal to 3 decimals, texture index 1.1508 vs 1.1475, and the ODF on the reference's 5 deg grid within
rms 0.035 mrd (correlation 0.998). The peak height moves by about 0.1 mrd with the solver's iteration limits alone,
and the reference value lies in that range.

Typical workflow:

1. Write the pole-figure CSVs (``io/polefig_csv.py``) and set :class:`~.mtex.MtexOptions`.
2. Call :func:`compute`, which reports each step to ``stage``.
3. Draw with ``plots/odf.py`` and write the outputs with ``io/odf.py``.

Example:
    >>> view_axes("+vz")[0]
    array([0., 0., 1.])
    >>> res = compute(["pole_figure_1234_111.csv", "pole_figure_1234_200.csv"], cell=(3.59, 3.59, 3.59, 90, 90, 90),
    ...               phase_name="IN718", view_from="+vz", opts=MtexOptions(), stage=no_stage)  # doctest: +SKIP
    >>> res.texture_index  # doctest: +SKIP
    1.15
"""

import importlib.util
import re
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .mtex import FROM_DATA, ColorMax, MtexOptions
from .projection import check_view
from .types import FloatArray

Progress = Callable[[str], None]

#: steps of :func:`compute` and the writer in ``io/odf.py``, in order: (key, label, typical seconds on a desktop)
STAGES: tuple[tuple[str, str, float], ...] = (
    ("start", "Loading MTEX for Python", 2.0),
    ("read", "Reading the pole figures", 0.2),
    ("odf", "Fitting the ODF (calcODF)", 1.0),
    ("errors", "RP errors and the ODF maximum", 1.0),
    ("grids", "Pole figure, ODF section and inverse pole figure grids", 1.5),
    ("plots", "Drawing the plots", 6.0),
    ("data", "Writing the plot data (CSV)", 0.5),
    ("summary", "Writing the ODF table and summary", 0.5),
)
STAGE_KEYS = tuple(k for k, _, _ in STAGES)
_PF_NAME = re.compile(r"pole_figure_(\d+)_(\d+)")
#: grid steps (deg) of the plot-data CSVs (as the reference script) and of the plots (finer, for smooth contours)
CSV_STEP, PLOT_STEP = 2.5, 1.0
#: the import name of MTEX for Python (pymtex, the PyPI package ``mtex``)
PYMTEX_MODULE = "mtex"


def pymtex_installed() -> bool:
    """True when MTEX for Python (pymtex) can be imported. It is checked without importing it (that takes seconds)."""
    return importlib.util.find_spec(PYMTEX_MODULE) is not None


def no_stage(key: str) -> None:
    """
    Stage callback that ignores every step (the default where no progress is wanted).

    Args:
        key: The step starting, one of ``STAGE_KEYS``.
    """


def stage_progress(key: str, *, seconds_in_stage: float = 0.0, speed: float = 1.0) -> tuple[float, float]:
    """
    Progress estimate from the typical step durations in ``STAGES``.

    Args:
        key: The current step, one of ``STAGE_KEYS``.
        seconds_in_stage: Time spent in the current step so far (s, default: 0).
        speed: This machine's time per typical second (default: 1).

    Returns:
        The fraction done (it stops at 90 % of a step until the next starts) and the estimated seconds left.

    Raises:
        ValueError: If key is not a step.
    """
    i = STAGE_KEYS.index(key)
    typ = [t for _, _, t in STAGES]
    total = sum(typ)
    in_stage = min(seconds_in_stage / max(speed * typ[i], 1e-9), 0.9) if typ[i] > 0 else 0.0
    frac = (sum(typ[:i]) + in_stage * typ[i]) / total
    left = speed * sum(typ[i + 1 :]) + max(speed * typ[i] - seconds_in_stage, 0.0)
    return frac, left


def range_max(spec: ColorMax, data_max: float) -> float:
    """
    Top of a color range.

    Args:
        spec: The user's maximum (mrd), or ``FROM_DATA``.
        data_max: The data maximum (mrd).

    Returns:
        ``spec``, or for ``FROM_DATA`` the data maximum rounded up to the next 0.5 mrd.
    """
    return float(np.ceil(data_max * 2) / 2) if spec == FROM_DATA else float(spec)


def equal_area(polar_deg: npt.ArrayLike) -> FloatArray:
    """
    Equal-area (Schmidt) radius of a polar angle, 1 at 90 deg (the MTEX default projection).

    Args:
        polar_deg: Polar angle from the view direction (deg).

    Returns:
        ``sqrt(2) sin(polar / 2)``.
    """
    return np.sqrt(2.0) * np.sin(np.radians(np.asarray(polar_deg, float)) / 2.0)


@dataclass
class MeasuredPF:
    """
    One measured pole figure, as kept for the ODF.

    Attributes:
        run: Run number from the file name.
        hkl: hkl from the file name.
        pole: Unit specimen directions (pole_vx, pole_vy, pole_vz), shape (n, 3).
        intensity: Intensity of each kept point (the chosen column, counts/s for corrected_per_s).
        n_in_csv: Rows in the CSV before any were dropped.
    """

    run: str
    hkl: str
    pole: FloatArray  # (n, 3) unit specimen directions
    intensity: FloatArray
    n_in_csv: int

    @property
    def label(self) -> str:
        """E.g. '{111} run 1234'."""
        return f"{{{self.hkl}}} run {self.run}"


def read_pole_figure(path: str | Path, opts: MtexOptions) -> MeasuredPF:
    """
    Read ``pole_figure_<run>_<hkl>.csv`` with the reference script's rules.

    Rows with a non-finite intensity or correction are dropped, and ignored rows when ``opts.drop_ignored``. With
    ``opts.merge_chi90`` the chi 90 rows (chi > 89.9) become one averaged point.

    Args:
        path: The CSV file.
        opts: The ODF options (intensity column, drop_ignored, merge_chi90).

    Returns:
        The kept points.

    Raises:
        ValueError: If the file name is not pole_figure_<run>_<hkl>.csv or the intensity column is absent.
    """
    m = _PF_NAME.search(Path(path).stem)
    if m is None:
        raise ValueError(f"{Path(path).name}: the name must be pole_figure_<run>_<hkl>.csv")
    a = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8"))  # 1 row: 0-d
    col = opts.intensity_column
    if col not in (a.dtype.names or ()):
        raise ValueError(f"{Path(path).name} has no column {col}")
    v = np.asarray(a[col], float)
    keep = np.isfinite(v) & np.isfinite(np.asarray(a["correction"], float))
    if opts.drop_ignored:
        keep &= np.asarray(a["ignored"], float) == 0
    pole = np.column_stack([np.asarray(a[k], float)[keep] for k in ("pole_vx", "pole_vy", "pole_vz")])
    v = v[keep]
    if opts.merge_chi90:  # the chi 90 ring is one pole (+vx): one averaged point
        c90 = np.asarray(a["chi"], float)[keep] > 89.9
        if c90.any():
            pole = np.vstack([pole[~c90], pole[c90].mean(axis=0)])
            v = np.append(v[~c90], v[c90].mean())
    return MeasuredPF(m.group(1), m.group(2), pole, v, len(a))


@dataclass
class PFGrid:
    """
    A recalculated pole figure on a polar x azimuth grid of the upper hemisphere in the plot view.

    Attributes:
        hkl: hkl label.
        polar: Polar angle of each grid point from the view direction (deg).
        azimuth: Azimuth from the right-hand axis toward the up axis (deg).
        pole: Grid points as unit vectors in the specimen frame, shape (n, 3).
        f: Pole density at each grid point (mrd).
    """

    hkl: str
    polar: FloatArray  # deg from the view direction
    azimuth: FloatArray  # deg from the right-hand axis toward the up axis
    pole: FloatArray  # (n, 3) in the specimen frame
    f: FloatArray  # mrd


@dataclass
class OdfResult:
    """
    Everything the plots, the plot-data CSVs and the summary need (no MTEX objects).

    Attributes:
        measured: The measured pole figures as used.
        rp_error: RP error of each measured pole figure.
        odf_max: ODF maximum (mrd).
        odf_max_euler: Bunge Euler angles phi1, Phi, phi2 of the maximum (deg).
        max_near: Per IPF direction: angles from the crystal direction of the maximum to 001, 011 and 111 (deg).
        texture_index: ODF texture index.
        entropy: ODF entropy.
        ipf_corner: Per IPF direction: inverse pole figure density at 001, 011 and 111 (mrd).
        cmax_odf: Top of the ODF section color range (mrd).
        cmax_pf: Top of the pole figure color range (mrd).
        cmax_ipf: Top of the inverse pole figure color range (mrd).
        plotkowski_max: Top of the eight-band color range (mrd), None when those plots are off.
        measured_scaled: Measured points scaled to the ODF (mrd) when the normalization worked, else as read.
        measured_recalc: The recalculated pole density at each measured point (mrd).
        measured_in_mrd: True when the normalization worked.
        section_plot: ODF section at phi2 on the plot grid: phi1, Phi (1D, deg) and f, shape (len(Phi), len(phi1)).
        section_csv: ODF section on the CSV grid: phi1, Phi and f flattened (phi1 fastest).
        pf_plot: Recalculated pole figures on the plot grid, by hkl.
        pf_csv: Recalculated pole figures on the CSV grid, by hkl.
        ipf: Per direction: crystal unit vectors in the fundamental sector, shape (n, 3), and the density (mrd).
        view_axes: Out of the page, right and up of the pole figure view (specimen frame).
        odf_table: The ODF on MTEX's Bunge grid: phi1, Phi, phi2 (deg), f (mrd).
        mtex_version: E.g. "pymtex-1.0.0".
    """

    measured: list[MeasuredPF]
    rp_error: FloatArray
    odf_max: float
    odf_max_euler: tuple[float, float, float]  # Bunge, deg
    max_near: dict[str, dict[str, float]]
    texture_index: float
    entropy: float
    ipf_corner: dict[str, dict[str, float]]
    cmax_odf: float
    cmax_pf: float
    cmax_ipf: float
    plotkowski_max: float | None
    measured_scaled: list[FloatArray]
    measured_recalc: list[FloatArray]
    measured_in_mrd: bool
    section_plot: tuple[FloatArray, FloatArray, FloatArray]  # phi1, Phi (1D, deg), f (len(Phi), len(phi1))
    section_csv: tuple[FloatArray, FloatArray, FloatArray]  # phi1, Phi, f flattened (phi1 fastest)
    pf_plot: dict[str, PFGrid]
    pf_csv: dict[str, PFGrid]
    ipf: dict[str, tuple[FloatArray, FloatArray]]  # direction -> (crystal unit vectors (n, 3), f)
    view_axes: tuple[FloatArray, FloatArray, FloatArray]  # out of the page, right, up (specimen frame)
    odf_table: FloatArray = field(default_factory=lambda: np.zeros((0, 4)))  # phi1, Phi, phi2, f (deg, mrd)
    mtex_version: str = ""


def _scalar(x: Any) -> float:
    return float(np.asarray(x, float).ravel()[0])


def _vec(name: str) -> FloatArray:
    return np.eye(3)[{"vx": 0, "vy": 1, "vz": 2}[name]]


def view_axes(view_from: str) -> tuple[FloatArray, FloatArray, FloatArray]:
    """
    Axes of the pole figure view: +vz with +vx right (the reference script), or +vx with +vy right.

    "-vz" (the 2D-map view of the other plots) also gives the +vz view: the MTEX plots are always seen from +vz
    (plotx2east, plotzOutOfPlane) or +vx.

    Args:
        view_from: "+vz", "-vz" or "+vx".

    Returns:
        Unit vectors out of the page, right and up, in the specimen frame.

    Raises:
        ValueError: If view_from is not one of the pole-figure views.
    """
    check_view(view_from)
    out, east = (np.eye(3)[0], np.eye(3)[1]) if view_from == "+vx" else (np.eye(3)[2], np.eye(3)[0])
    return out, east, np.cross(out, east)


def hemisphere_grid(step: float, axes: tuple[FloatArray, FloatArray, FloatArray]) -> tuple[FloatArray, ...]:
    """
    Grid on the upper hemisphere of a view.

    Args:
        step: Grid step in polar angle and azimuth (deg).
        axes: Out of the page, right and up (see :func:`view_axes`).

    Returns:
        Polar angle (deg, 0 to 90 inclusive), azimuth (deg, 0 to 360 inclusive, varying fastest) and the grid points
        as unit vectors, shape (n, 3).
    """
    out, east, north = axes
    az, po = np.meshgrid(np.arange(0.0, 360.0 + step / 2, step), np.arange(0.0, 90.0 + step / 2, step))
    az, po = az.ravel(), po.ravel()
    s, c = np.sin(np.radians(po)), np.cos(np.radians(po))
    poles = c[:, None] * out + s[:, None] * (np.cos(np.radians(az))[:, None] * east
                                             + np.sin(np.radians(az))[:, None] * north)  # fmt: skip
    return po, az, poles


def all_hkl(opts: MtexOptions, measured: Sequence[MeasuredPF]) -> list[str]:
    """
    Every hkl that is plotted.

    Args:
        opts: The ODF options.
        measured: The measured pole figures.

    Returns:
        pf_hkl, then the measured hkl, then the eight-band (Plotkowski) hkl when those plots are on, without repeats.
    """
    out: list[str] = []
    for h in [*opts.pf_hkl, *(m.hkl for m in measured), *(opts.plotkowski_hkl if opts.plotkowski else [])]:
        if h not in out:
            out.append(h)
    return out


def compute(csv_files: Sequence[str | Path], *, cell: Sequence[float], phase_name: str, view_from: str,
            opts: MtexOptions, stage: Progress) -> OdfResult:  # fmt: skip
    """
    Fit the ODF and evaluate everything the outputs need.

    Args:
        csv_files: Two or more pole_figure_<run>_<hkl>.csv files.
        cell: a, b, c (Å) and alpha, beta, gamma (deg).
        phase_name: Mineral name for MTEX.
        view_from: Pole-figure view, "+vz", "-vz" or "+vx" (see :func:`view_axes`).
        opts: The ODF options.
        stage: ``stage(key)``, called as each step starts (keys in ``STAGE_KEYS``). It may raise to stop.
            ``no_stage`` for no reporting.

    Returns:
        The ODF results.

    Raises:
        ValueError: If there are fewer than two files, a file name or column is wrong, or view_from is unknown.
        ModuleNotFoundError: If MTEX for Python (pymtex) is not installed.
    """
    at = stage
    at("start")
    # deferred: pymtex is an optional dependency (installed separately, not in every environment), and its first
    # import costs seconds (numba compiles its kernels)
    import mtex as mt

    mt.setMTEXpref("threads", 1)  # conda-forge's FINUFFT is single-threaded on Windows; one thread is as fast here
    deg = mt.degree
    if len(csv_files) < 2:
        raise ValueError(f"the ODF needs at least two pole figures, got {len(csv_files)}")

    at("read")
    measured = [read_pole_figure(f, opts) for f in csv_files]
    cs = mt.crystalFrame(opts.crystal_symmetry, [float(c) for c in cell[:3]], [float(c) * deg for c in cell[3:6]],
                         mineral=phase_name)  # fmt: skip
    ss = mt.specimenFrame(opts.specimen_symmetry)

    def miller(hkl: str) -> Any:
        return mt.Miller(*[int(c) for c in hkl], cs)

    def v3(p: FloatArray) -> Any:
        p = np.atleast_2d(p)
        return mt.vector3d(p[:, 0], p[:, 1], p[:, 2])

    pf = mt.PoleFigure([miller(m.hkl) for m in measured], [v3(m.pole) for m in measured],
                       [m.intensity for m in measured], cs, ss)  # fmt: skip

    at("odf")
    odf = mt.calcODF(pf, halfwidth=opts.halfwidth_deg * deg, resolution=opts.resolution_deg * deg)

    at("errors")
    rp = np.asarray(mt.calcError(pf, odf), float).ravel()
    mx, om = mt.max(odf)
    euler = (float(om.phi1) / deg % 360.0, float(om.Phi) / deg, float(om.phi2) / deg)
    ti = float(mt.textureIndex(odf))
    en = float(mt.entropy(odf))
    corners = {"c001": miller("001"), "c011": miller("011"), "c111": miller("111")}
    max_near: dict[str, dict[str, float]] = {}
    ipf_corner: dict[str, dict[str, float]] = {}
    for d in opts.ipf_directions:
        r = v3(_vec(d))
        m_dir = mt.Miller(mt.inv(om) * r, cs)  # crystal direction of the maximum along d
        max_near[d] = {f"angle_to_{k[1:]}_deg": _scalar(mt.angle(m_dir, c)) / deg for k, c in corners.items()}
        ipf_corner[d] = {k: _scalar(odf.radon(c, r)) for k, c in corners.items()}

    try:
        pfn = mt.normalize(pf, odf)
        norm_ok = True
    except Exception:  # noqa: BLE001  (as the script: fall back to counts/s and say so in the CSV)
        pfn, norm_ok = pf, False
    scaled = [np.asarray(pfn.select(k).intensities, float).ravel() for k in range(len(measured))]
    recalc = [np.asarray(odf.radon(miller(m.hkl), v3(m.pole)), float).ravel() for m in measured]

    at("grids")
    # color ranges as the script: maxima on 2 deg grids of the whole sphere
    s2 = mt.equispacedS2Grid(resolution=2 * deg)
    pf_max = max(float(np.nanmax(odf.radon(miller(h), s2))) for h in [*opts.pf_hkl, *(m.hkl for m in measured)])
    ipf_max = max(float(np.nanmax(odf.radon(r=v3(_vec(d))).eval(s2))) for d in opts.ipf_directions)
    k_max: float | None = None
    if opts.plotkowski:
        s1 = mt.equispacedS2Grid(resolution=1 * deg)
        k_data = max(float(np.nanmax(odf.radon(miller(h), s1))) for h in opts.plotkowski_hkl)
        k_max = k_data if opts.plotkowski_max == FROM_DATA else float(opts.plotkowski_max)

    axes = view_axes(view_from)
    hkls = all_hkl(opts, measured)

    def pf_grids(step: float) -> dict[str, PFGrid]:
        po, az, poles = hemisphere_grid(step, axes)
        return {h: PFGrid(h, po, az, poles, np.asarray(odf.radon(miller(h), v3(poles)), float).ravel()) for h in hkls}

    def section(step: float) -> tuple[FloatArray, FloatArray, FloatArray]:
        p1 = np.arange(0.0, 360.0 + step / 2, step)
        PP = np.arange(0.0, 90.0 + step / 2, step)
        g1, gP = np.meshgrid(p1, PP)  # rows: Phi, columns: phi1 (phi1 fastest when flattened)
        ori = mt.orientation.byEuler(g1.ravel() * deg, gP.ravel() * deg, np.full(g1.size, opts.phi2_deg) * deg,
                                     cs, ss)  # fmt: skip
        return p1, PP, np.asarray(odf.eval(ori), float).reshape(g1.shape)

    sp = section(PLOT_STEP)
    p1c, PPc, fc = section(CSV_STEP)
    g1c, gPc = np.meshgrid(p1c, PPc)
    gi = mt.equispacedS2Grid(resolution=1 * deg)
    inside = np.asarray(cs.fundamentalSector().checkInside(gi), bool)
    xyz = np.column_stack([np.asarray(gi.x, float), np.asarray(gi.y, float), np.asarray(gi.z, float)])[inside]
    ipf = {d: (xyz, np.asarray(odf.radon(r=v3(_vec(d))).eval(v3(xyz)), float).ravel()) for d in opts.ipf_directions}

    res = opts.resolution_deg
    table = euler_table(odf, res, mt)

    return OdfResult(
        measured=measured, rp_error=rp, odf_max=float(mx), odf_max_euler=euler, max_near=max_near,
        texture_index=ti, entropy=en, ipf_corner=ipf_corner,
        cmax_odf=range_max(opts.cmax_odf, float(mx)), cmax_pf=range_max(opts.cmax_pf, pf_max),
        cmax_ipf=range_max(opts.cmax_ipf, ipf_max), plotkowski_max=k_max,
        measured_scaled=scaled, measured_recalc=recalc, measured_in_mrd=norm_ok,
        section_plot=sp, section_csv=(g1c.ravel(), gPc.ravel(), fc.ravel()),
        pf_plot=pf_grids(PLOT_STEP), pf_csv=pf_grids(CSV_STEP), ipf=ipf, view_axes=axes,
        odf_table=table, mtex_version=f"pymtex-{getattr(mt, '__version__', '?')}",
    )  # fmt: skip


def euler_table(odf: Any, step: float, mt: Any) -> FloatArray:
    """
    The ODF on MTEX's Bunge grid of the Euler fundamental region (phi1 fastest).

    MTEX's own ``export(odf, file, 'Bunge')`` (72 x 19 x 18 points at 5 deg for m-3m, the grid of the reference).

    Args:
        odf: The MTEX ODF.
        step: Grid resolution (deg).
        mt: The ``mtex`` module.

    Returns:
        Rows of phi1, Phi, phi2 (deg) and f (mrd).
    """
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "odf.txt"
        mt.export(odf, str(f), "Bunge", resolution=step * mt.degree)
        return np.atleast_2d(np.loadtxt(f, comments="%"))
