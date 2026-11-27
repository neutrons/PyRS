"""Helpers for the GUI tests (Qt only, so they stay out of tests/helpers.py, which the core tests share)."""

from pytestqt.qtbot import QtBot

from pyrs.interface.pole_figures.widgets.plot_panel import PlotPanel

FIGURE_TIMEOUT_MS = 60_000


def wait_figure(qtbot: QtBot, panel: PlotPanel) -> None:
    """Wait until the panel shows a figure (figures are built in the background, with a "Drawing ..." note)."""
    qtbot.waitUntil(lambda: panel.kind == "figure", timeout=FIGURE_TIMEOUT_MS)
