"""Figures built off the GUI thread (newest request wins), and the GIF paused while its panel is hidden."""

import threading
from pathlib import Path

import pytest
from matplotlib.figure import Figure
from PIL import Image
from pytestqt.qtbot import QtBot
from qtpy.QtGui import QMovie
from qtpy.QtWidgets import QVBoxLayout, QWidget

from pyrs.interface.pole_figures.figure_cache import FigureCache
from pyrs.interface.pole_figures.figure_jobs import BackgroundFigures
from pyrs.interface.pole_figures.widgets.plot_panel import PlotPanel
from tests.util.pole_figures_gui_helpers import wait_figure

pytestmark = [pytest.mark.gui, pytest.mark.integration]


@pytest.fixture
def panel(qtbot: QtBot) -> PlotPanel:
    p = PlotPanel()
    qtbot.addWidget(p)
    return p


def _gif(path: Path) -> Path:
    frames = [Image.new("RGB", (20, 20), c) for c in ("red", "blue", "green")]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=50, loop=0)
    return path


def test_show_builds_off_gui_thread(panel: PlotPanel, qtbot: QtBot) -> None:
    # Arrange
    jobs = BackgroundFigures(panel, FigureCache())
    threads: list[threading.Thread] = []
    fig = Figure()

    def build() -> Figure:
        threads.append(threading.current_thread())
        return fig

    # Act
    jobs.show("a", build, name="the test plot")
    note = panel.message()
    wait_figure(qtbot, panel)

    # Assert
    assert note == "Drawing the test plot ..."
    assert threads and threads[0] is not threading.main_thread()
    assert panel.canvas is not None and panel.canvas.figure is fig and jobs.cache.cached("a") is fig


def test_show_newer_request_drops_stale_figure(panel: PlotPanel, qtbot: QtBot) -> None:
    # Arrange: the first build waits until the second request is made
    jobs = BackgroundFigures(panel, FigureCache())
    started, release = threading.Event(), threading.Event()
    old, new = Figure(), Figure()

    def slow() -> Figure:
        started.set()
        release.wait(10)
        return old

    # Act
    jobs.show("old", slow, name="old")
    started.wait(10)
    jobs.show("new", lambda: new, name="new")
    release.set()
    qtbot.waitUntil(lambda: not jobs.busy, timeout=10_000)

    # Assert: the newer figure is shown, the stale one is only cached
    assert panel.canvas is not None and panel.canvas.figure is new
    assert jobs.cache.cached("old") is old


def test_clear_drops_figures_of_old_inputs(panel: PlotPanel, qtbot: QtBot) -> None:
    # Arrange
    jobs = BackgroundFigures(panel, FigureCache())
    release = threading.Event()

    def slow() -> Figure:
        release.wait(10)
        return Figure()

    jobs.show("a", slow, name="a")

    # Act: the inputs change while "a" is drawn
    jobs.clear()
    release.set()
    qtbot.waitUntil(lambda: not jobs.busy, timeout=10_000)

    # Assert
    assert jobs.cache.cached("a") is None and panel.kind == "none"


def test_cancel_keeps_message_shown(panel: PlotPanel, qtbot: QtBot) -> None:
    # Arrange
    jobs = BackgroundFigures(panel, FigureCache())
    jobs.show("a", Figure, name="a")

    # Act: the presenter shows something else (a GIF note) before the figure is ready
    jobs.cancel()
    panel.show_message("No GIF yet.")
    qtbot.waitUntil(lambda: not jobs.busy, timeout=10_000)

    # Assert
    assert panel.message() == "No GIF yet."


def test_show_gif_paused_while_hidden(qtbot: QtBot, tmp_path: Path) -> None:
    # Arrange: the panel inside a page, as on a tab
    page = QWidget()
    qtbot.addWidget(page)
    panel = PlotPanel()
    QVBoxLayout(page).addWidget(panel)
    page.show()
    panel.show_gif(_gif(tmp_path / "a.gif"))
    assert panel.movie is not None
    playing = panel.movie.state()

    # Act
    page.hide()
    hidden = panel.movie.state()
    page.show()
    shown = panel.movie.state()

    # Assert
    assert playing == QMovie.MovieState.Running
    assert hidden == QMovie.MovieState.Paused and shown == QMovie.MovieState.Running


def test_show_gif_on_hidden_panel_starts_paused(panel: PlotPanel, tmp_path: Path) -> None:
    # Arrange: the panel was never shown (its tab is not the current one)
    path = _gif(tmp_path / "a.gif")

    # Act
    panel.show_gif(path)

    # Assert
    assert panel.movie is not None and panel.movie.state() == QMovie.MovieState.Paused
    assert panel.movie.frameCount() == 3
