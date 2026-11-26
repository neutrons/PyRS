"""
Rotation helpers of the stage geometry: signed sample axes, rotation matrices and checks that a matrix is a rotation.

Ported from the reference script (``rot``). The signed sample axes ("+vx", "-vz", ...) name how a sample sits on the
cradle (``geometry.MountConventions``).

Example:
    >>> axis_vector("-vz")
    array([ 0.,  0., -1.])
    >>> axis_name([0.0, -0.9, 0.1])
    '-vy'
    >>> np.round(rot([0, 0, 1], 90.0) @ [1.0, 0.0, 0.0], 12) + 0.0
    array([0., 1., 0.])
"""

from typing import Literal

import numpy as np
import numpy.typing as npt

from .types import FloatArray

SampleAxis = Literal["+vx", "-vx", "+vy", "-vy", "+vz", "-vz"]
SAMPLE_AXES: tuple[SampleAxis, ...] = ("+vx", "-vx", "+vy", "-vy", "+vz", "-vz")


def axis_vector(a: str) -> FloatArray:
    """
    Unit vector of a signed sample axis: '+vx' -> (1, 0, 0), '-vz' -> (0, 0, -1).

    Args:
        a: One of ``SAMPLE_AXES``.

    Returns:
        The unit vector in the sample frame.

    Raises:
        ValueError: If ``a`` is not one of ``SAMPLE_AXES``.
    """
    if a not in SAMPLE_AXES:
        raise ValueError(f"sample axis must be one of {', '.join(SAMPLE_AXES)}, not {a!r}")
    v = np.zeros(3)
    v["xyz".index(a[2])] = 1.0 if a[0] == "+" else -1.0
    return v


def check_rotation(m: npt.ArrayLike, *, tol: float = 2e-3) -> None:
    """
    Check that m is a proper rotation (orthonormal rows, determinant +1) within tol.

    Args:
        m: A 3 x 3 sample-to-instrument matrix.
        tol: Largest allowed element of ``|m m^T - I|`` (default: 2e-3, so values typed to 3 or 4 decimals pass).

    Raises:
        ValueError: If m is not 3 x 3 finite numbers, its rows are not orthonormal within tol, or it is a reflection
            (negative determinant).
    """
    m = np.asarray(m, float)
    if m.shape != (3, 3) or not np.all(np.isfinite(m)):
        raise ValueError(f"the sample-to-instrument matrix must be 3 x 3 finite numbers, got {m.tolist()!r}")
    err = float(np.abs(m @ m.T - np.eye(3)).max())
    if err > tol:
        raise ValueError(f"the sample-to-instrument matrix is not a rotation: rows must be unit length and at right "
                         f"angles (off by {err:.3g})")  # fmt: skip
    det = float(np.linalg.det(m))
    if det < 0:
        raise ValueError(f"the sample-to-instrument matrix is a reflection (determinant {det:.3g}): flip the sign of "
                         "one row")  # fmt: skip


def nearest_rotation(m: npt.ArrayLike) -> FloatArray:
    """
    The rotation closest to m (polar decomposition), so a matrix typed to 3 or 4 decimals is exactly orthonormal.

    A matrix that is already orthonormal (to 1e-12) is returned as it is, so the identity stays exact.

    Args:
        m: A 3 x 3 matrix that is nearly a rotation.

    Returns:
        The orthonormal 3 x 3 matrix ``U V^T`` from the singular value decomposition of m.
    """
    m = np.asarray(m, float)
    if np.abs(m @ m.T - np.eye(3)).max() < 1e-12:
        return m.copy()
    u, _, vt = np.linalg.svd(m)
    return np.asarray(u @ vt)


def axis_name(v: npt.ArrayLike) -> str:
    """
    Name of the sample axis nearest a vector: (0, -1, 0) -> '-vy'.

    Args:
        v: A vector in the sample frame. Only its largest component counts.

    Returns:
        The signed axis name, one of ``SAMPLE_AXES``.
    """
    v = np.asarray(v, float)
    i = int(np.argmax(np.abs(v)))
    return ("+" if v[i] > 0 else "-") + "v" + "xyz"[i]


def rot(axis: npt.ArrayLike, deg: float) -> FloatArray:
    """
    Rotation matrix about an axis (right hand, Rodrigues formula).

    Args:
        axis: Rotation axis, any length.
        deg: Rotation angle (deg).

    Returns:
        The 3 x 3 rotation matrix.
    """
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K
