"""
Combine runs that measured the same pole figure (``combine_runs`` of the reference script).

A pole figure split over several reduced files (a scan stopped and restarted, or measured in parts) is merged into
one pole figure per hkl: every point keeps its own pole vector, rate and correction. Scale "none" (default) keeps the
rates as they are, which is right for parts of one scan. Scale "median" multiplies each run by
median(reference) / median(run) of the corrected rates of the points that are kept. That assumes the runs differ only
by a constant efficiency (for example different detector regions, such as eta slices).

Typical workflow:

1. Analyze each run with ``polefig.analyze_run``.
2. Merge the pole figures that share an hkl with :func:`combine_by_hkl` (or two or more chosen ones with
   :func:`combine`).

Example:
    >>> results = [pf for run in runs for pf in analyze_run(run, ...)]  # doctest: +SKIP
    >>> merged = combine_by_hkl(results, scale="median")  # doctest: +SKIP
"""

import typing
from collections.abc import Sequence
from typing import Literal

import numpy as np

from .polefig import PolePeak

Scale = Literal["none", "median"]


def _kept_corrected(r: PolePeak) -> np.ndarray:
    return np.where(r.ignored | ~np.isfinite(r.correction), np.nan, r.rate * r.correction)


def combine(parts: Sequence[PolePeak], *, scale: Scale = "none", reference: str) -> PolePeak:
    """
    Merge pole figures of the same hkl into one.

    Every point keeps its own pole vector, rate, sigma and correction. The first part gives the mount used to draw
    cells, omega, the wavelength and the beams. 2theta and FWHM are the means of the parts.

    Args:
        parts: Two or more pole figures of one hkl, analyzed with the same cutoff, mu, method and wavelength.
        scale: "none" keeps the rates (default). "median" scales each part's rate and sigma by
            median(reference) / median(part) of the kept corrected rates.
        reference: Run name of the part the others are scaled to. It must be one of the parts.

    Returns:
        The combined pole figure, run named "<run>+<run>...", with ``source_runs`` and a note on the scaling.

    Raises:
        ValueError: If scale is unknown, there are fewer than two parts, the parts have different hkl, cutoff, mu,
            method or wavelength, reference is not one of the parts, or (scale "median") a part has no kept point
            with a positive corrected rate.
    """
    if scale not in typing.get_args(Scale):
        raise ValueError(f"scale must be one of {', '.join(typing.get_args(Scale))}, got {scale!r}")
    if len(parts) < 2:
        raise ValueError(f"combining needs at least two pole figures, got {len(parts)}")
    hkls = {r.hkl for r in parts}
    if len(hkls) != 1:
        raise ValueError(f"cannot combine different reflections: {', '.join(sorted(hkls))}")
    if reference not in {r.run for r in parts}:
        names = ", ".join(r.run for r in parts)
        raise ValueError(f"reference run {reference} is not one of the runs combined ({names})")
    ref = next(r for r in parts if r.run == reference)
    for name in ("cutoff", "mu", "method", "wavelength_A"):
        values = {getattr(r, name) for r in parts}
        if len(values) > 1:
            raise ValueError(f"cannot combine runs analyzed with different {name}: {', '.join(map(str, values))}")
    if scale == "median":
        medians = {r.run: float(np.nanmedian(_kept_corrected(r))) if np.isfinite(_kept_corrected(r)).any()
                   else float("nan") for r in parts}  # fmt: skip
        bad = [run for run, m in medians.items() if not (np.isfinite(m) and m > 0)]
        if bad:
            raise ValueError(f"cannot scale by median: run {', '.join(bad)} has no kept points with a positive "
                             "corrected rate")  # fmt: skip
        fac = [medians[ref.run] / medians[r.run] for r in parts]
    else:
        fac = [1.0] * len(parts)
    first = parts[0]

    def cat(name: str) -> np.ndarray:
        return np.concatenate([np.asarray(getattr(r, name), float) for r in parts])

    runs = tuple(r.run for r in parts)
    how = (f"scaled by median: {', '.join(f'{r.run} x{f:.3f}' for r, f in zip(parts, fac, strict=True))}"
           if scale == "median" else "no scaling between runs")  # fmt: skip
    return PolePeak(
        run="+".join(runs), hkl=first.hkl, title=" | ".join(dict.fromkeys(r.title for r in parts)),
        two_theta=float(np.mean([r.two_theta for r in parts])), fwhm=float(np.mean([r.fwhm for r in parts])),
        omega=first.omega, wavelength_A=first.wavelength_A, beams=first.beams, mount=first.mount,
        chi=cat("chi"), phi=cat("phi"), count_s=cat("count_s"),
        rate=np.concatenate([r.rate * f for r, f in zip(parts, fac, strict=True)]),
        sigma=np.concatenate([r.sigma * f for r, f in zip(parts, fac, strict=True)]),
        F=cat("F"), F_ref=first.F_ref, volume_in_sample=cat("volume_in_sample"), mean_path=cat("mean_path"),
        vy_offset=cat("vy_offset"), vy_rms=cat("vy_rms"), correction=cat("correction"), cutoff=first.cutoff,
        mu=first.mu, method=first.method, live=any(r.live for r in parts), chi_offset_deg=first.chi_offset_deg,
        h=np.vstack([r.poles() for r in parts]), source_runs=runs,
        notes=f"combined from runs {', '.join(runs)}, {how}",
    )  # fmt: skip


def combine_by_hkl(results: Sequence[PolePeak], *, scale: Scale = "none") -> list[PolePeak]:
    """
    One pole figure per hkl: runs sharing an hkl are merged, single runs are kept.

    Args:
        results: Pole figures of any hkl.
        scale: "none" (default) or "median", as in :func:`combine`. The first run of each hkl is the reference.

    Returns:
        One pole figure per hkl, in order of first appearance.

    Raises:
        ValueError: If the runs of one hkl cannot be combined (see :func:`combine`).
    """
    groups: dict[str, list[PolePeak]] = {}
    for r in results:
        groups.setdefault(r.hkl, []).append(r)
    return [combine(g, scale=scale, reference=g[0].run) if len(g) > 1 else g[0] for g in groups.values()]
