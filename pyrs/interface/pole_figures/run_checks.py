"""
Checks on loaded runs and the gauge position before the Data and Corrections analysis.

These are GUI-level rules (what the tab accepts), kept out of ``core/``:

* :func:`is_spatial_map`: a run with fewer than two distinct chi, phi orientations is a spatial (strain) map, not a
  pole-figure scan.
* :func:`inside_with_margin`: the gauge center is inside the sample and clear of its surface.
* :func:`mostly_negative`: most net rates of a pole figure are below zero (background above the peak).

Example:
    >>> import numpy as np
    >>> orientation_count(np.zeros(4), np.zeros(4))
    1
    >>> mostly_negative(np.array([-1.0, -2.0, 3.0]))
    True
"""

import numpy as np
import numpy.typing as npt

from pyrs.pole_figures.core.runs import RunData
from pyrs.pole_figures.core.stl_mesh import SampleMesh

ANGLE_DECIMALS = 2  # chi and phi are compared after rounding to 0.01 deg
MIN_ORIENTATIONS = 2  # a pole-figure scan has at least this many distinct (chi, phi) orientations
GAUGE_MARGIN_MM = 0.1  # the gauge center must stay this far inside the sample along vx, vy and vz

MAP_NOTE = (
    "spatial map (one chi, phi orientation). It is not a pole-figure scan and is left out. "
    "Use the PyRS strain/stress tools."
)
MARGIN_NOTE = (
    f"The center and the six points {GAUGE_MARGIN_MM:g} mm from it along vx, vy and vz must all be inside the sample, "
    "so a point on a face, edge or corner is refused."
)
NEGATIVE_NOTE = "net counts below background: check the material and peak."


def orientation_count(chi: npt.ArrayLike, phi: npt.ArrayLike) -> int:
    """
    Number of distinct (chi, phi) orientations, after rounding both to ANGLE_DECIMALS (0.01 deg).

    Args:
        chi: Logged chi per sub-run (deg).
        phi: Logged phi per sub-run (deg), same length as ``chi``.

    Returns:
        The number of distinct pairs (0 for no sub-runs).

    Example:
        >>> orientation_count([0.0, 0.001, 10.0], [5.0, 5.0, 5.0])
        2
    """
    pairs = np.c_[np.round(np.asarray(chi, float), ANGLE_DECIMALS), np.round(np.asarray(phi, float), ANGLE_DECIMALS)]
    return len(np.unique(pairs, axis=0)) if len(pairs) else 0


def is_spatial_map(run: RunData) -> bool:
    """
    True when the run is a spatial map: fewer than MIN_ORIENTATIONS (2) distinct (chi, phi) orientations.

    Such a run moves the sample (vx, vy, vz) at one orientation, for strain or stress. Its points all give the same
    pole, so it is not a pole-figure scan.

    Args:
        run: A loaded run.

    Returns:
        True for a spatial map.
    """
    return orientation_count(run.chi, run.phi) < MIN_ORIENTATIONS


def inside_with_margin(mesh: SampleMesh, center: npt.ArrayLike, *, margin_mm: float = GAUGE_MARGIN_MM) -> bool:
    """
    True when ``center`` and the six points ``margin_mm`` from it along +-vx, +-vy and +-vz are all inside the mesh.

    A center on a face, edge or corner of the sample fails, since some of the six points fall outside. The inside
    test of a point exactly on the surface is not reliable, so this rule avoids it.

    Args:
        mesh: The sample.
        center: Gauge center (mm, STL frame), three values.
        margin_mm: Distance of the six points from the center (mm, default: GAUGE_MARGIN_MM, 0.1).

    Returns:
        True when all seven points are inside.
    """
    c = np.asarray(center, float).reshape(3)
    pts = np.vstack([c, c + margin_mm * np.eye(3), c - margin_mm * np.eye(3)])
    return bool(np.all(mesh.contains(pts)))


def mostly_negative(rates: npt.ArrayLike) -> bool:
    """
    True when the median of the finite net rates is below zero.

    Args:
        rates: Net counts per second of a pole figure.

    Returns:
        True when most rates are negative. False when no rate is finite.
    """
    r = np.asarray(rates, float)
    r = r[np.isfinite(r)]
    return bool(r.size) and float(np.median(r)) < 0
