"""
Shared plot colors, as the reference script uses them.

``AXIS_COLORS`` and ``TEXT_COLORS`` mark the sample axes vx, vy and vz (arrows and labels). The other names color
the gauge, the incident and diffracted beams, Q, the sample, the phi dial, ignored cells and cell edges.
``SEQUENTIAL`` is the colormap for measured intensities.

Example:
    >>> from pyrs.pole_figures.plots.style import AXIS_COLORS, SEQUENTIAL
    >>> AXIS_COLORS["vx"]
    '#2a78d6'
"""

from matplotlib.colors import LinearSegmentedColormap

AXIS_COLORS = {"vx": "#2a78d6", "vy": "#eb6834", "vz": "#1baf7a"}
TEXT_COLORS = {"vx": "#1c5cab", "vy": "#b0461a", "vz": "#0f7a55"}
GAUGE = "#c0392b"
INCIDENT = "#4d4d4d"
DIFFRACTED = "#a0522d"
Q_COLOR = "#7b2fa8"
SAMPLE = "#c9a227"
DIAL = "#e6e3dc"
IGNORED = "#c9c6bf"
SURFACE = "#fcfcfb"

#: the reference script's colormap for measured intensities
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
