"""
tests/util/transformation_helpers.py

Helpers that apply a NeXus ``NXtransformations`` chain from its fields' attributes alone, independently of the
PyRS code that wrote it, so a test fails when the writer changes rather than tracking it.

Example:
    >>> trans = detector["transformations"]
    >>> placed = (local @ chain_matrix(trans, "shift_x", arm=-90.0).T)[..., :3]
"""

import numpy as np


def rodrigues(axis, degrees: float) -> np.ndarray:
    """Right-handed 3x3 rotation of `degrees` about `axis` (Rodrigues' formula).

    Args:
        axis: Rotation axis; need not be normalised.
        degrees: Rotation angle in degrees.

    Returns:
        The rotation matrix.
    """
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    theta = np.radians(degrees)
    cross = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * cross + (1.0 - np.cos(theta)) * (cross @ cross)


def chain_matrix(transformations, entry: str, arm: float) -> np.ndarray:
    """The 4x4 homogeneous transform of a whole chain, innermost first, built from the fields' attributes.

    Args:
        transformations: The ``NXtransformations`` group holding the chain.
        entry: Name of the first (innermost) transformation, as the detector's ``depends_on`` names it.
        arm: Arm angle in degrees, used for ``arm_rotation`` in place of its stored value(s).

    Returns:
        The transform taking a pixel's local position to the lab frame.
    """
    total = np.eye(4)
    name = entry
    while name != ".":
        link = transformations[name]
        value = arm if name == "arm_rotation" else float(link.nxdata)
        vector = np.asarray(link.attrs["vector"], dtype=float)
        step = np.eye(4)
        if str(link.attrs["transformation_type"]) == "translation":
            step[:3, 3] = value * vector
        else:
            step[:3, :3] = rodrigues(vector, value)
        total = step @ total
        name = str(link.attrs["depends_on"])
    return total
