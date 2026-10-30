"""
Shared plotting helpers used across PyRS fitting UIs.

This module aggregates plotting-preparation logic that was previously duplicated
between the peak-fitting and texture-fitting interfaces. The goal is a single
implementation of the "given x/y/z data and a display mode, build the arrays that
``GeneralDiffDataView.plot_3D_scatter`` expects" decision so that both UIs stay in
sync.

Typical usage:

1. Pull the parameter arrays for the chosen axes from the fit results.
2. Call :func:`prepare_3d_plot_data` with the display mode selected in the UI.
3. Hand the returned arrays to ``ax_object.plot_3D_scatter(...)``.
"""

from typing import Optional, Tuple

import numpy as np
from matplotlib import colormaps
from matplotlib.pyplot import Normalize
from scipy.interpolate import griddata

# Colormap shared by every 3D/contour plot in the fitting UIs.
_COOLWARM = colormaps["coolwarm"]

#: Display modes understood by :func:`prepare_3d_plot_data`.
VALID_MODES = ("contour", "lines", "scatter")


def prepare_3d_plot_data(
    x_data,
    y_data,
    z_data,
    mode: str = "scatter",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], bool]:
    """
    Build the arrays consumed by ``GeneralDiffDataView.plot_3D_scatter``.

    The function reproduces the contour / 3D-line / scatter branching that the
    peak-fitting and texture-fitting controllers previously implemented
    independently. When the x or y axis has all-unique values the data cannot be
    placed on a regular grid, so the result is forced to a scatter plot regardless
    of the requested ``mode``.

    Args:
        x_data: Values for the x axis.
        y_data: Values for the y axis.
        z_data: Values for the z (color) axis.
        mode: One of ``"contour"``, ``"lines"`` or ``"scatter"`` (default
            ``"scatter"``). Unknown values fall back to scatter.

    Returns:
        A tuple ``(vec_x, vec_y, vec_z, colors, plot_scatter)`` where:

        - ``vec_x``, ``vec_y``, ``vec_z`` are the arrays to plot (gridded for
          contour/lines, copies of the inputs for scatter).
        - ``colors`` is an RGBA array for line/scatter modes, or ``None`` when the
          renderer should color the surface itself (contour mode) or when the z
          data cannot be normalized.
        - ``plot_scatter`` indicates whether the renderer should draw a scatter
          plot rather than a surface.

    Example:
        >>> vx, vy, vz, colors, scatter = prepare_3d_plot_data(
        ...     [1, 2], [3, 4], [5, 6], mode="scatter"
        ... )
        >>> scatter
        True
    """
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    z_data = np.asarray(z_data)

    # Data with all-unique x or y values cannot be gridded; fall back to scatter.
    plot_scatter = (x_data.size == np.unique(x_data).size) or (y_data.size == np.unique(y_data).size)

    if mode == "contour" and not plot_scatter:
        vec_x, vec_y = np.meshgrid(np.unique(x_data), np.unique(y_data))
        vec_z = griddata((x_data, y_data), z_data, (vec_x, vec_y), method="nearest")
        return vec_x, vec_y, vec_z, None, False

    if mode == "lines" and not plot_scatter:
        vec_x, vec_y = np.meshgrid(np.unique(x_data), np.unique(y_data))
        vec_z = griddata((x_data, y_data), z_data, (vec_x, vec_y), method="nearest")
        norm = Normalize(vec_z.min(), vec_z.max())
        return vec_x, vec_y, vec_z, _COOLWARM(norm(vec_z)), False

    # Scatter: requested explicitly, used as fallback, or forced by ungriddable data.
    try:
        norm = Normalize(z_data.min(), z_data.max())
        colors = _COOLWARM(norm(z_data))
    except ValueError:
        # Empty z data has no min/max; let the renderer pick a default color.
        colors = None

    return np.copy(x_data), np.copy(y_data), np.copy(z_data), colors, True
