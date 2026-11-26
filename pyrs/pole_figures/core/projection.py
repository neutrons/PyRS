"""
Equal-area (Schmidt) projection of sample-frame poles, and the pole-figure views.

Ported from the reference script (``project``). Its global ``VIEW`` is the explicit ``view`` argument here.

Views ("pf_view_from" in a reference-script config): "+vz" is seen from +vz with +vx right and +vy up (default),
"-vz" mirrors it as the 2D maps do (vx to the left), and "+vx" is seen down the phi axis with +vy right and +vz up.

Typical workflow:

1. Get the poles of a pole figure (``geometry.Mount.poles``).
2. Place them on the pole figure with :func:`project`.

Example:
    >>> project(np.array([0.0, 0.0, 1.0]))
    array([[0., 0.]])
"""

import typing
from typing import Literal

import numpy as np
import numpy.typing as npt

from .types import FloatArray

ViewFrom = Literal["+vz", "-vz", "+vx"]

# Display frame for "pf_view_from": "+vx": (vy, vz, vx), vx toward the viewer.
_VIEW_PLUS_VX = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])


def check_view(view: str) -> None:
    """
    Check that ``view`` is one of the pole-figure views (``ViewFrom``).

    Args:
        view: "+vz", "-vz" or "+vx".

    Raises:
        ValueError: If view is anything else.
    """
    allowed = typing.get_args(ViewFrom)
    if view not in allowed:
        raise ValueError(f"view must be one of {', '.join(allowed)}, got {view!r}")


def to_view(h: npt.ArrayLike, *, view: ViewFrom = "+vz") -> FloatArray:
    """
    Sample-frame vectors in the display frame (x right, y up, z toward the viewer).

    Args:
        h: Sample-frame vectors, shape (..., 3).
        view: "+vz", "-vz" or "+vx" (default: "+vz"). Only "+vx" changes the frame: (vy, vz, vx). The "-vz" mirror
            is applied in :func:`project`.

    Returns:
        The vectors in the display frame, same shape.

    Raises:
        ValueError: If view is not one of the pole-figure views.
    """
    check_view(view)
    h = np.asarray(h, float)
    return h @ _VIEW_PLUS_VX.T if view == "+vx" else h


def from_view(d: npt.ArrayLike, *, view: ViewFrom = "+vz") -> FloatArray:
    """
    Display-frame vectors back to the sample frame (inverse of :func:`to_view`).

    Args:
        d: Display-frame vectors, shape (..., 3).
        view: "+vz", "-vz" or "+vx" (default: "+vz").

    Returns:
        The vectors in the sample frame, same shape.

    Raises:
        ValueError: If view is not one of the pole-figure views.
    """
    check_view(view)
    d = np.asarray(d, float)
    return d @ _VIEW_PLUS_VX if view == "+vx" else d


def project(h: npt.ArrayLike, *, view: ViewFrom = "+vz", fold: bool = True) -> FloatArray:
    """
    Equal-area (Schmidt) projection of sample-frame unit vectors.

    Args:
        h: Sample-frame unit vectors, shape (3,) or (n, 3).
        view: "+vz": seen from +vz, +vx right, +vy up (default). "-vz": as the 2D maps (vx to the left). "+vx": seen
            down the phi axis, +vy right, +vz up.
        fold: Replace poles on the far hemisphere with their antipodes (default: True).

    Returns:
        Projected points, shape (n, 2), radius 1 at the equator.

    Raises:
        ValueError: If view is not one of the pole-figure views.
    """
    check_view(view)
    sgn = -1.0 if view == "-vz" else 1.0
    h = np.atleast_2d(np.asarray(h, float))
    h = to_view(h, view=view)
    if fold:
        h = np.where(sgn * h[:, 2:3] < -1e-9, -h, h)
    alpha = np.arccos(np.clip(sgn * h[:, 2], -1, 1))
    r = np.sqrt(2) * np.sin(alpha / 2)
    xy = np.c_[sgn * h[:, 0], h[:, 1]]
    n = np.linalg.norm(xy, axis=1)
    n[n < 1e-12] = 1
    return np.asarray(xy / n[:, None] * r[:, None])
