"""
ODF step settings and checks (the reference script's ``mtex`` config block), and the names of its outputs.

The ODF itself is fitted with MTEX for Python (pymtex, ``core/odf.py``) and written out by ``io/odf.py``. Rules:
specimen directions are pole_vx, pole_vy, pole_vz, the intensity is
corrected_per_s, ignored rows are dropped, the hkl comes from the file name, de la Vallee Poussin kernel with halfwidth
and resolution 5 deg, no sample symmetry, seen from +vz with +vx east (plotx2east, plotzOutOfPlane).

Typical workflow:

1. Keep the pole figures MTEX can use (:func:`usable`) and write their CSVs.
2. Set the options in :class:`MtexOptions` (the crystal symmetry from :func:`laue_group`).
3. Build odf_settings.json with :func:`settings`, fit with ``odf.compute`` and write with ``io/odf.py``.
4. List the outputs with :func:`plot_files` and :func:`plot_data_files`.

Example:
    >>> laue_group("Fm-3m")
    'm-3m'
    >>> opts = MtexOptions(crystal_symmetry=laue_group("Fm-3m"), pf_hkl=parse_hkl_list("111, 200 220"))
    >>> opts.pf_hkl
    ['111', '200', '220']
"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

import gemmi
import numpy as np

from .polefig import PolePeak

#: a color-range maximum: a value in mrd, or FROM_DATA for the data maximum rounded up to the next 0.5 mrd
FROM_DATA: Literal["data"] = "data"
ColorMax: TypeAlias = float | Literal["data"]

#: plots written by io/odf.py (the reference script's file names), in display order:
#: (name, file, white-below-random version or None).
#: {phi2} is filled from the ODF section. "simple" and "Square" are the Peterson 2021 styles, "colorful" is the
#: Plotkowski 2021 eight-band style.
OUTPUT_PLOTS: tuple[tuple[str, str, str | None], ...] = (
    ("Measured vs recalculated", "pole_figures_measured_vs_recalculated.png", None),
    ("Recalculated - simple", "pole_figures_recalculated.png", "pole_figures_recalculated_white.png"),
    ("Recalculated - colorful", "pole_figures_plotkowski_style.png", None),
    ("Square", "odf_phi2_{phi2}.png", "odf_phi2_{phi2}_white.png"),
    ("Inverse pole figures", "inverse_pole_figures.png", None),
)


def laue_group(space_group: str) -> str:
    """
    MTEX point group for antipodal pole figures: the Laue class of the space group (Fm-3m -> m-3m).

    gemmi gives -3m for both trigonal settings. MTEX tells them apart: P-31m, P312 and P31m give -31m (the 2-fold or
    mirror normal along a*), all others (P-3m1, R-3m, ...) give -3m1.

    Args:
        space_group: Hermann-Mauguin symbol in any form gemmi accepts.

    Returns:
        The Laue class in MTEX notation, e.g. 'm-3m', '6/mmm', '-3m1'.
    """
    sg = gemmi.SpaceGroup(space_group)
    laue = str(sg.laue_str())
    if laue == "-3m":
        parts = sg.hm.split()
        return "-31m" if len(parts) == 4 and parts[2] == "1" else "-3m1"
    return laue


def vz_label(vz_name: str) -> str:
    """
    Label of +vz in the MTEX plots, as the reference script.

    Args:
        vz_name: The user's name for vz, e.g. 'BD' (build direction).

    Returns:
        'BD' for 'BD', '+vz' for 'vz' or '+vz', else '+vz (name)'.
    """
    if vz_name == "BD":
        return "BD"
    return "+vz" if vz_name in ("vz", "+vz") else f"+vz ({vz_name})"


def chi_rings(r: PolePeak) -> int:
    """
    Number of chi rings among the points that are kept (MTEX needs two or more for its quadrature).

    Args:
        r: A measured pole figure. Chi values that round to the same 0.1 deg are one ring.

    Returns:
        The number of distinct chi values among the points that are not ignored.
    """
    kept = ~r.ignored
    return int(len(np.unique(np.round(r.chi[kept], 1))))


def usable(r: PolePeak) -> bool:
    """True when the pole figure has the two or more chi rings MTEX needs (see :func:`chi_rings`)."""
    return chi_rings(r) >= 2


def parse_hkl_list(text: str) -> list[str]:
    """
    Read a list of hkl typed as text, split by commas or spaces.

    Args:
        text: The list, e.g. '111, 200 220'.

    Returns:
        The hkl labels, e.g. ['111', '200', '220'].

    Raises:
        ValueError: If an entry is not three digits (each index from 0 to 9).
    """
    out = [t for t in text.replace(",", " ").split() if t]
    for t in out:
        if len(t) != 3 or not t.isdigit():
            raise ValueError(f"hkl '{t}' must be three digits, e.g. 111 or 200")
    return out


@dataclass(kw_only=True)
class MtexOptions:
    """
    The ``mtex`` block of the reference script's config (the defaults are its settings for an fcc alloy).

    Attributes:
        crystal_symmetry: MTEX point group of the crystal (default: "m-3m").
        specimen_symmetry: MTEX specimen symmetry (default: "1", none).
        intensity_column: CSV column used as the intensity (default: "corrected_per_s").
        drop_ignored: Leave out rows with ``ignored == 1`` (default: True).
        halfwidth_deg: de la Vallee Poussin kernel halfwidth (deg, default: 5).
        resolution_deg: ODF grid resolution (deg, default: 5).
        merge_chi90: Average the chi 90 ring into one point (default: False).
        ipf_directions: Specimen directions of the inverse pole figures, from "vx", "vy", "vz".
        pf_hkl: hkl of the recalculated pole figures.
        cmax_odf: Top of the ODF section color range (mrd), or FROM_DATA (default).
        cmax_pf: Top of the pole figure color range (mrd), or FROM_DATA (default).
        cmax_ipf: Top of the inverse pole figure color range (mrd), or FROM_DATA (default).
        contour_levels: Number of contour levels (default: 20).
        phi2_deg: phi2 of the ODF section (deg, default: 45).
        white_below_mrd: Also write plots with white below this level (mrd, default: 1), None for no such plots.
        plotkowski: Also draw the eight-band pole figures (default: True).
        plotkowski_hkl: hkl of the eight-band pole figures.
        plotkowski_max: Top of the eight-band color range (mrd), or FROM_DATA (default).
        export_data: Write plot_data/*.csv, the numbers behind each plot (default: True, not in the reference
            script).

    Example:
        >>> MtexOptions(halfwidth_deg=7.5).block()["halfwidth_deg"]
        7.5
    """

    crystal_symmetry: str = "m-3m"
    specimen_symmetry: str = "1"
    intensity_column: str = "corrected_per_s"
    drop_ignored: bool = True
    halfwidth_deg: float = 5.0
    resolution_deg: float = 5.0
    merge_chi90: bool = False
    ipf_directions: list[str] = field(default_factory=lambda: ["vx", "vy", "vz"])
    pf_hkl: list[str] = field(default_factory=lambda: ["111", "200", "220"])
    cmax_odf: ColorMax = FROM_DATA
    cmax_pf: ColorMax = FROM_DATA
    cmax_ipf: ColorMax = FROM_DATA
    contour_levels: int = 20
    phi2_deg: float = 45.0
    white_below_mrd: float | None = 1.0  # None: no white-background plots
    plotkowski: bool = True
    plotkowski_hkl: list[str] = field(default_factory=lambda: ["100", "110", "111"])
    plotkowski_max: ColorMax = FROM_DATA
    export_data: bool = True  # plot_data/*.csv: the numbers behind each plot (off in the reference script)

    def block(self) -> dict[str, Any]:
        """
        The ``mtex`` dict of odf_settings.json (the reference script's layout).

        Returns:
            The options, with FROM_DATA color-range maxima written as "data", and the fixed plot settings.
        """

        def rng(v: ColorMax) -> float | str:
            return FROM_DATA if v == FROM_DATA else float(v)

        plot: dict[str, Any] = {
            "colormap": "WhiteJetColorMap", "contour_levels": int(self.contour_levels), "contour_lines": False,
            "odf_section": {"phi2_deg": self.phi2_deg, "phi1_ticks_deg": 45, "Phi_ticks_deg": 30},
            "pf_hkl": list(self.pf_hkl), "cmax_odf": rng(self.cmax_odf), "cmax_pf": rng(self.cmax_pf),
            "cmax_ipf": rng(self.cmax_ipf),
            "plotkowski": {"enabled": bool(self.plotkowski), "hkl": list(self.plotkowski_hkl),
                           "max": rng(self.plotkowski_max)},
            "white_background": False if self.white_below_mrd is None else {"below_mrd": float(self.white_below_mrd)},
            "export_data": bool(self.export_data),
        }  # fmt: skip
        d = asdict(self)
        keys = ("crystal_symmetry", "specimen_symmetry", "intensity_column", "drop_ignored", "halfwidth_deg",
                "resolution_deg", "merge_chi90", "ipf_directions")  # fmt: skip
        out = {k: d[k] for k in keys}
        out.update({"enabled": True, "plot_view": "plotx2east, plotzOutOfPlane", "output_subfolder": "mtex",
                    "plot": plot})  # fmt: skip
        return out


def settings(csv_files: Sequence[str | Path], *, out_dir: str | Path, phase_name: str,
             cell: tuple[float, float, float, float, float, float], view_from: str, vz_name: str,
             opts: MtexOptions) -> dict[str, Any]:  # fmt: skip
    """
    The odf_settings.json content (the reference script's layout plus ``cell``).

    Args:
        csv_files: The pole_figure_<run>_<hkl>.csv files (written as absolute POSIX paths).
        out_dir: Folder for the MTEX outputs.
        phase_name: Mineral name shown in MTEX.
        cell: a, b, c (Å) and alpha, beta, gamma (deg). ``lattice_A`` is a.
        view_from: Pole-figure view, "+vz", "-vz" or "+vx".
        vz_name: The user's name for vz (see :func:`vz_label`).
        opts: The ODF options.

    Returns:
        The settings dict.

    Raises:
        ValueError: If there are fewer than two CSV files or no inverse pole figure direction.
    """
    if len(csv_files) < 2:
        raise ValueError(f"MTEX needs at least two pole figures, got {len(csv_files)}")
    if not opts.ipf_directions:
        raise ValueError(f"choose at least one inverse pole figure direction, got {opts.ipf_directions!r}")
    return {
        "csv_files": [Path(f).resolve().as_posix() for f in csv_files],
        "out_dir": Path(out_dir).resolve().as_posix(),
        "lattice_A": float(cell[0]),
        "cell": [float(c) for c in cell],
        "phase_name": phase_name,
        "view_from": view_from,
        "vz_label": vz_label(vz_name),
        "mtex": opts.block(),
    }


def plot_data_files(out_dir: str | Path) -> list[Path]:
    """
    The plot-data CSVs (and README.txt) written to out_dir/plot_data.

    Args:
        out_dir: Folder of the MTEX outputs.

    Returns:
        The CSV files sorted by name, then README.txt if it exists.
    """
    d = Path(out_dir) / "plot_data"
    return sorted(d.glob("*.csv")) + ([d / "README.txt"] if (d / "README.txt").is_file() else [])


def plot_files(out_dir: str | Path, *, phi2_deg: float = 45.0) -> list[tuple[str, Path, Path | None]]:
    """
    The MTEX plots that exist in out_dir, in display order.

    Args:
        out_dir: Folder of the MTEX outputs.
        phi2_deg: phi2 of the ODF section, which is in the section's file name (deg, default: 45).

    Returns:
        (name, path, white-below-random path or None) for each plot that exists.
    """
    out = []
    for name, pattern, white in OUTPUT_PLOTS:
        p = Path(out_dir) / pattern.format(phi2=f"{phi2_deg:g}")
        w = Path(out_dir) / white.format(phi2=f"{phi2_deg:g}") if white else None
        if p.is_file():
            out.append((name, p, w if w is not None and w.is_file() else None))
    return out
