"""
Mount-matrix helpers for the Sample tab: the identity mount, the stage axes and turning the mount about one of them.

Example:
    >>> rotate_mount(IDENTITY9, 2, 90.0)
    (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
"""

import numpy as np
import numpy.typing as npt

from pyrs.pole_figures.core.rotation import nearest_rotation, rot

#: stage (instrument) axes at chi 0, phi 0: e_x up along the phi axis, e_y across (the chi axis), e_z along Q
STAGE_AXES: tuple[str, ...] = ("phi axis (up)", "chi axis (across)", "Q (omega = theta)")
IDENTITY9: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def rotate_mount(m: npt.ArrayLike, stage_axis: int, deg: float) -> tuple[float, ...]:
    """
    Turn the mounted sample about a stage axis by deg (right hand).

    The sample-to-instrument matrix M0 becomes rot(e_axis, deg) @ M0, so presses add up in the stage frame. The
    result is made exactly orthonormal and values below 1e-12 are set to 0.

    Args:
        m: The current M0, 9 values row by row (or 3 x 3).
        stage_axis: Stage axis at chi 0, phi 0: 0 the phi axis (e_x, up), 1 the chi axis (e_y), 2 Q (e_z).
        deg: Rotation angle (deg).

    Returns:
        The new M0 as 9 values, row by row.

    Example:
        >>> rotate_mount(IDENTITY9, 2, 90.0)
        (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    """
    m3 = np.reshape(np.asarray(m, float), (3, 3))
    new = nearest_rotation(rot(np.eye(3)[stage_axis], deg) @ m3)
    new[np.abs(new) < 1e-12] = 0.0
    return tuple(float(x) for x in new.ravel())
