"""
Run the ODF step with MTEX for Python and write its outputs, with the reference script's file names and columns.

Layout of the work folder: ``<work>/pole_figure_<run>_<hkl>.csv`` and ``<work>/mtex/``:

- the plots (``core.mtex.OUTPUT_PLOTS``), PNG at 200 dpi
- ``odf_summary.json`` and ``odf_summary.txt``: points used, RP errors, ODF maximum, texture index, entropy, ranges
- ``odf_<res>deg.txt``: the ODF on a Bunge Euler grid (MTEX export)
- ``plot_data/*.csv`` and ``plot_data/README.txt``: the numbers behind each plot
- ``odf_settings.json``: the inputs, for the record

Typical workflow:

1. Build the pole figures with ``core.polefig`` and set the options in ``core.mtex.MtexOptions``.
2. Call ``run_odf``. It writes the CSVs, fits the ODF (``core.odf.compute``), and writes the plots, the plot data,
   the ODF table and the summary.

Every file goes through ``io.retry.write_atomic`` (a temporary file moved over the final name).

Example:
    >>> out = run_odf("work", pole_figures, phase_name="IN718", cell=(3.6, 3.6, 3.6, 90, 90, 90), view_from="+vz",
    ...               vz_name="BD", opts=MtexOptions(), on_stage=no_stage, should_stop=never_stop)
    >>> (out / "odf_summary.txt").exists()
    True
"""

import csv
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..core.mtex import MtexOptions, settings, vz_label
from ..core.odf import OdfResult, PFGrid, compute, equal_area
from ..core.polefig import PolePeak
from ..plots import odf as odf_plots
from ..plots.save import save_figure
from .polefig_csv import write_all
from .retry import write_atomic

DPI = 200


class OdfCancelledError(RuntimeError):
    """Raised by ``run_odf`` when ``should_stop`` asks to stop the ODF step."""


def never_stop() -> bool:
    """A ``should_stop`` for ``run_odf`` that never stops the ODF step."""
    return False


def run_odf(work: str | Path, pole_figures: list[PolePeak], *, phase_name: str,
            cell: tuple[float, float, float, float, float, float], view_from: str, vz_name: str, opts: MtexOptions,
            on_stage: Callable[[str], None], should_stop: Callable[[], bool]) -> Path:  # fmt: skip
    """
    Write the CSVs, fit the ODF and write every output.

    Args:
        work: Work folder. The pole-figure CSVs go here and the outputs into ``<work>/mtex``.
        pole_figures: Pole figures to fit, one CSV each.
        phase_name: Phase name for MTEX and the settings record.
        cell: Lattice parameters a, b, c (Å) and alpha, beta, gamma (deg).
        view_from: Pole-figure view, "+vz" (or "-vz", drawn as "+vz") or "+vx".
        vz_name: Name of +vz in the labels, for example "BD".
        opts: ODF and plot options.
        on_stage: Called with the key of each step of ``core.odf.STAGES`` as it starts (``core.odf.no_stage`` for
            no reporting).
        should_stop: Checked before each step (``never_stop`` to run to the end).

    Returns:
        The output folder ``<work>/mtex``.

    Raises:
        OdfCancelledError: If ``should_stop`` returns True.
        ModuleNotFoundError: If MTEX for Python (pymtex) is not installed.
    """
    work = Path(work)
    out = work / "mtex"
    out.mkdir(parents=True, exist_ok=True)

    def stage(key: str) -> None:
        if should_stop():
            raise OdfCancelledError(f"cancelled before the step {key!r}")
        on_stage(key)

    csvs = write_all(work, pole_figures)
    s = settings(csvs, out_dir=out, phase_name=phase_name, cell=cell, view_from=view_from, vz_name=vz_name, opts=opts)
    _write_text(out / "odf_settings.json", json.dumps(s, indent=1))
    res = compute(csvs, cell=cell, phase_name=phase_name, view_from=view_from, opts=opts, stage=stage)
    vzl = vz_label(vz_name)

    stage("plots")
    write_plots(res, out, opts=opts, vzl=vzl)
    if opts.export_data:
        stage("data")
        write_plot_data(res, out, opts=opts, vzl=vzl, view_x=view_from == "+vx")
    stage("summary")
    write_odf_table(res, out / f"odf_{opts.resolution_deg:g}deg.txt", crystal_symmetry=opts.crystal_symmetry,
                    specimen_symmetry=opts.specimen_symmetry)  # fmt: skip
    write_summary(res, out, opts=opts, vzl=vzl)
    return out


def _write_text(path: Path, text: str) -> None:
    write_atomic(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))


def write_plots(res: OdfResult, out: Path, *, opts: MtexOptions, vzl: str) -> None:
    """
    Write the plots of ``core.mtex.OUTPUT_PLOTS`` as PNG at ``DPI``.

    The white-background versions are added when ``opts.white_below_mrd`` is set, and the Plotkowski-style plot when
    ``opts.plotkowski`` is True.

    Args:
        res: ODF result.
        out: Output folder.
        opts: ODF and plot options.
        vzl: Label of +vz in the plots (``core.mtex.vz_label``).
    """
    n = opts.contour_levels
    figs = {
        "pole_figures_recalculated.png": odf_plots.recalculated_figure(res, opts.pf_hkl, n_levels=n, vz_label=vzl,
                                                                       white_below_mrd=None),
        f"odf_phi2_{opts.phi2_deg:g}.png": odf_plots.section_figure(res, phi2_deg=opts.phi2_deg, n_levels=n,
                                                                   white_below_mrd=None),
        "pole_figures_measured_vs_recalculated.png": odf_plots.measured_vs_recalculated_figure(res, n_levels=n,
                                                                                               vz_label=vzl),
        "inverse_pole_figures.png": odf_plots.ipf_figure(res, _ipf_names(vzl), n_levels=n,
                                                         cubic=opts.crystal_symmetry == "m-3m"),
    }  # fmt: skip
    if opts.white_below_mrd is not None:
        wb = opts.white_below_mrd
        figs["pole_figures_recalculated_white.png"] = odf_plots.recalculated_figure(
            res, opts.pf_hkl, n_levels=n, vz_label=vzl, white_below_mrd=wb
        )
        figs[f"odf_phi2_{opts.phi2_deg:g}_white.png"] = odf_plots.section_figure(
            res, phi2_deg=opts.phi2_deg, n_levels=n, white_below_mrd=wb
        )
    if opts.plotkowski:
        figs["pole_figures_plotkowski_style.png"] = odf_plots.plotkowski_figure(res, opts.plotkowski_hkl,
                                                                                vz_label=vzl)  # fmt: skip
    for name, fig in figs.items():
        save_figure(fig, out / name, dpi=DPI, bbox_inches="tight")


def _ipf_names(vzl: str) -> dict[str, str]:
    return {"vx": "+vx", "vy": "+vy", "vz": vzl}


# plot data: the same files and columns as the reference script writes
def _write_csv(path: Path, header: Sequence[str], rows: Any) -> None:
    def write(tmp: Path) -> None:
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)

    write_atomic(path, write)


def _g(v: float) -> str:
    return f"{v:.15g}"


def write_plot_data(res: OdfResult, out: Path, *, opts: MtexOptions, vzl: str, view_x: bool) -> None:
    """
    Write ``plot_data/*.csv`` and ``plot_data/README.txt``: the numbers behind each plot, to redraw them elsewhere.

    Pole positions are on the upper hemisphere in the equal-area projection (radius 1 at the rim) of the plot view.

    Args:
        res: ODF result.
        out: Output folder. ``plot_data`` is made inside it.
        opts: ODF and plot options.
        vzl: Label of +vz (``core.mtex.vz_label``).
        view_x: True when the pole figures are seen down +vx, False when seen down +vz. Only the README text uses it.
    """
    d = out / "plot_data"
    d.mkdir(exist_ok=True)
    p1, PP, f = res.section_csv
    _write_csv(d / f"odf_section_phi2_{opts.phi2_deg:g}.csv", ["phi1_deg", "Phi_deg", "phi2_deg", "f_mrd"],
               ([_g(a), _g(b), _g(opts.phi2_deg), _g(c)] for a, b, c in zip(p1, PP, f, strict=True)))  # fmt: skip

    def pf_rows(g: PFGrid) -> Any:
        r = equal_area(g.polar)
        x, y = r * np.cos(np.radians(g.azimuth)), r * np.sin(np.radians(g.azimuth))
        for k in range(len(g.f)):
            yield [g.hkl, _g(g.polar[k]), _g(g.azimuth[k]), *(_g(v) for v in g.pole[k]), _g(x[k]), _g(y[k]),
                   _g(g.f[k])]  # fmt: skip

    _write_csv(d / "pole_figures_recalculated.csv",
               ["hkl", "polar_deg", "azimuth_deg", "pole_vx", "pole_vy", "pole_vz", "x_plot", "y_plot", "f_mrd"],
               (row for g in res.pf_csv.values() for row in pf_rows(g)))  # fmt: skip

    out_v, east, north = res.view_axes
    rows = []
    for k, m in enumerate(res.measured):
        c, e, n = m.pole @ out_v, m.pole @ east, m.pole @ north
        flip = c < 0
        c, e, n = np.where(flip, -c, c), np.where(flip, -e, e), np.where(flip, -n, n)
        po = np.degrees(np.arccos(np.minimum(c, 1.0)))
        az = np.degrees(np.arctan2(n, e)) % 360.0
        r = equal_area(po)
        for i in range(len(m.intensity)):
            rows.append([m.label, *(_g(v) for v in m.pole[i]), _g(po[i]), _g(az[i]),
                         _g(r[i] * np.cos(np.radians(az[i]))), _g(r[i] * np.sin(np.radians(az[i]))),
                         _g(res.measured_scaled[k][i]), _g(res.measured_recalc[k][i]), int(res.measured_in_mrd),
                         _g(res.rp_error[k])])  # fmt: skip
    _write_csv(d / "pole_figures_measured.csv",
               ["pole_figure", "pole_vx", "pole_vy", "pole_vz", "polar_deg", "azimuth_deg", "x_plot", "y_plot",
                "measured", "recalculated_mrd", "measured_in_mrd", "RP_error"], rows)  # fmt: skip

    names = _ipf_names(vzl)
    rows = []
    for dname, (xyz, fv) in res.ipf.items():
        po = np.degrees(np.arccos(np.clip(xyz[:, 2], -1, 1)))
        az = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])) % 360.0
        r = equal_area(po)
        for i in range(len(fv)):
            x, y = r[i] * np.cos(np.radians(az[i])), r[i] * np.sin(np.radians(az[i]))
            rows.append([names.get(dname, dname), *(_g(v) for v in xyz[i]), _g(po[i]), _g(az[i]), _g(x), _g(y),
                         _g(fv[i])])  # fmt: skip
    _write_csv(d / "inverse_pole_figures.csv",
               ["sample_direction", "crystal_x", "crystal_y", "crystal_z", "polar_deg", "azimuth_deg", "x_plot",
                "y_plot", "f_mrd"], rows)  # fmt: skip

    wb = "NaN" if opts.white_below_mrd is None else _g(opts.white_below_mrd)
    nl = opts.contour_levels
    _write_csv(d / "color_scales.csv", ["plot", "min_mrd", "max_mrd", "contour_levels", "white_below_mrd"],
               [["odf_section", 0, _g(res.cmax_odf), nl, wb], ["pole_figures", 0, _g(res.cmax_pf), nl, wb],
                ["inverse_pole_figures", 0, _g(res.cmax_ipf), nl, "NaN"]])  # fmt: skip
    _write_csv(d / "colormap_WhiteJetColorMap.csv", ["r", "g", "b"],
               ([_g(v) for v in rgb] for rgb in odf_plots.white_jet()))  # fmt: skip
    if opts.plotkowski and res.plotkowski_max is not None:
        k_max = res.plotkowski_max
        edges = k_max ** (np.arange(-1, 6) / 6.0)
        lower, upper = [0.0, *edges], [*edges, k_max]
        kc = np.round(odf_plots.PLOTKOWSKI_COLORS * 255).astype(int)
        _write_csv(d / "plotkowski_bands.csv", ["band", "lower_mrd", "upper_mrd", "r_255", "g_255", "b_255"],
                   ([k + 1, _g(lower[k]), _g(upper[k]), *kc[k]] for k in range(8)))  # fmt: skip
    write_readme(d / "README.txt", view_x=view_x, vzl=vzl)


def write_readme(path: Path, *, view_x: bool, vzl: str) -> None:
    """
    Write ``plot_data/README.txt``: what each CSV holds and how to redraw the plots.

    Args:
        path: Output file.
        view_x: True when the pole figures are seen down +vx, False when seen down +vz.
        vzl: Label of +vz (``core.mtex.vz_label``).
    """
    view = (f"seen down +vx: +vy to the right (x_plot), {vzl} up (y_plot)" if view_x
            else f"seen down {vzl} (+vz): +vx to the right (x_plot), +vy up (y_plot)")  # fmt: skip
    lines = [
        "Data behind the ODF plots (MTEX for Python). Values are in mrd (multiples of a random distribution).",
        "",
        "Pole figures use the upper hemisphere (pole figures are antipodal) in an equal-area (Schmidt) projection,",
        f"the MTEX default, {view}.",
        "x_plot, y_plot: r = sqrt(2) sin(polar/2), 1 at the rim; polar_deg from the view direction, azimuth_deg from",
        "the right-hand axis toward the up axis. pole_vx, pole_vy, pole_vz: unit pole vector in the sample frame.",
        "",
        "odf_section_phi2_<phi2>.csv     ODF f(g) on a 2.5 deg grid of phi1 (0-360, fastest) and Phi (0-90)",
        "                                at fixed phi2, Bunge Euler angles. Plot as a filled contour of f_mrd",
        "                                over phi1 (x) and Phi (y, down).",
        "pole_figures_recalculated.csv   recalculated pole figure of each hkl on a 2.5 deg polar x azimuth grid.",
        "                                Plot a filled contour (or triangulation) of f_mrd at (x_plot, y_plot),",
        "                                one panel per hkl. Covers the recalculated, white-background, measured",
        "                                vs recalculated (bottom row) and Plotkowski-style plots.",
        "pole_figures_measured.csv       measured points used for the ODF (top row of measured vs recalculated):",
        "                                measured intensity (scaled to the ODF in mrd when measured_in_mrd is 1,",
        "                                else counts/s), the recalculated value at the same pole, and the RP error",
        "                                of that pole figure.",
        "inverse_pole_figures.csv        f for crystal directions in the fundamental sector (1 deg grid), one",
        "                                block per sample direction. crystal_x/y/z: unit vector in the MTEX",
        "                                crystal frame. x_plot, y_plot: equal-area from crystal z, azimuth from",
        "                                crystal x (the MTEX IPF layout).",
        "color_scales.csv                color range and contour levels of each plot type, and the white-background",
        "                                level (values below it are white in the *_white plots).",
        "colormap_<name>.csv             the colormap (r, g, b from 0 to 1, low to high).",
        "plotkowski_bands.csv            the eight log-spaced bands and colors of the Plotkowski-style plot.",
    ]
    _write_text(path, "\n".join(lines) + "\n")


def write_odf_table(res: OdfResult, path: Path, *, crystal_symmetry: str, specimen_symmetry: str) -> None:
    """
    Write the ODF on the Bunge Euler grid, in MTEX's export layout.

    A ``%`` header, then one line per grid point: phi1, Phi, phi2 (deg) and the ODF value (mrd).

    Args:
        res: ODF result.
        path: Output text file.
        crystal_symmetry: Crystal symmetry written in the header, for example "m-3m".
        specimen_symmetry: Specimen symmetry written in the header, for example "1".
    """
    head = (f'% MTEX ODF\n% crystal symmetry: "{crystal_symmetry}"\n% specimen symmetry: "{specimen_symmetry}"\n'
            "% phi1    Phi     phi2    value\n")  # fmt: skip
    body = "\n".join(f"{a:.5f} {b:.5f} {c:.5f} {v:.5f}" for a, b, c, v in res.odf_table)
    _write_text(path, head + body + "\n")


def summary_dict(res: OdfResult, opts: MtexOptions) -> dict[str, Any]:
    """
    Contents of ``odf_summary.json``, with the reference script's keys.

    Args:
        res: ODF result.
        opts: ODF options (halfwidth, resolution, intensity column, chi 90 merge).

    Returns:
        JSON-ready dictionary: points used, RP errors, ODF maximum (mrd) and its Bunge angles (deg), texture index,
        entropy, color ranges and settings.
    """
    return {
        "pole_figures": [m.label for m in res.measured],
        "points_used": [len(m.intensity) for m in res.measured],
        "points_in_csv": [m.n_in_csv for m in res.measured],
        "rp_error": [float(v) for v in res.rp_error],
        "odf_max_mrd": res.odf_max,
        "odf_max_euler_bunge_deg": list(res.odf_max_euler),
        "odf_max_crystal_direction_along": res.max_near,
        "texture_index_norm2": res.texture_index,
        "entropy": res.entropy,
        "cmax_odf": res.cmax_odf,
        "cmax_pf": res.cmax_pf,
        "cmax_ipf": res.cmax_ipf,
        "halfwidth_deg": opts.halfwidth_deg,
        "resolution_deg": opts.resolution_deg,
        "intensity_column": opts.intensity_column,
        "merge_chi90": opts.merge_chi90,
        "mtex_version": res.mtex_version,
        "ipf_corner_mrd": res.ipf_corner,
    }


def write_summary(res: OdfResult, out: Path, *, opts: MtexOptions, vzl: str) -> None:
    """
    Write ``odf_summary.json`` and the readable ``odf_summary.txt``.

    Args:
        res: ODF result.
        out: Output folder.
        opts: ODF options.
        vzl: Label of +vz (``core.mtex.vz_label``).
    """
    _write_text(out / "odf_summary.json", json.dumps(summary_dict(res, opts), indent=2))
    names = _ipf_names(vzl)
    lines = [f"ODF from corrected pole figures ({opts.intensity_column}), de la Vallee Poussin kernel, halfwidth "
             f"{opts.halfwidth_deg:g} deg, resolution {opts.resolution_deg:g} deg ({res.mtex_version})"]  # fmt: skip
    for k, m in enumerate(res.measured):
        lines.append(f"{m.label:<20} {len(m.intensity):4d} of {m.n_in_csv:4d} points, RP error {res.rp_error[k]:.3f}")
    e = res.odf_max_euler
    lines.append(f"ODF max {res.odf_max:.2f} mrd at phi1 {e[0]:.1f}, Phi {e[1]:.1f}, phi2 {e[2]:.1f} deg (Bunge)")
    for d, q in res.max_near.items():
        lines.append(f"  along {names.get(d, d):<8} the max orientation has its crystal direction "
                     f"{q['angle_to_001_deg']:.1f} deg from <001>, {q['angle_to_011_deg']:.1f} from <011>, "
                     f"{q['angle_to_111_deg']:.1f} from <111>")  # fmt: skip
    lines.append(f"texture index norm(odf)^2 {res.texture_index:.3f}, entropy {res.entropy:.3f}")
    lines.append(f"color ranges: ODF 0-{res.cmax_odf:g}, pole figures 0-{res.cmax_pf:g}, inverse pole figures "
                 f"0-{res.cmax_ipf:g} mrd")  # fmt: skip
    _write_text(out / "odf_summary.txt", "\n".join(lines) + "\n")
