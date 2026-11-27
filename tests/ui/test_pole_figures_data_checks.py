"""Data and Corrections presenter: spatial-map runs, moving logs, the gauge margin, negative rates, table refreshes."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

import pyrs.interface.pole_figures.presenters.data as data_presenter
from pyrs.interface.pole_figures.main_window import MainWindow
from pyrs.interface.pole_figures.tabs.data_tab import RUN_COLUMNS
from tests.util.pole_figures_helpers import STL_DIR, replace_dataset, synthetic_pole_figure, write_reduced_run

pytestmark = [pytest.mark.gui, pytest.mark.integration]

STL = STL_DIR / "slab.stl"
CENTER = {"gauge_vx": 29.4, "gauge_vy": 19.75, "gauge_vz": 1.65}
SUBRUNS = RUN_COLUMNS.index("Sub-runs")


@pytest.fixture
def window(qtbot: QtBot) -> MainWindow:
    """A new main window."""
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _ready(w: MainWindow) -> None:
    """Material, STL and a small Monte Carlo sample: everything but the runs and the gauge position."""
    w.material_tab.preset_combo.setCurrentText("IN718 (fcc)")
    w.sample.load_stl(STL)
    w.data_tab.settings_form.set_values({"mc_points": 1000})
    w.data._on_form_changed()


def _map_run(path: Path) -> Path:
    """A spatial map: one chi, phi orientation, vx moving 2 mm over the sub-runs."""
    write_reduced_run(path, position=(29.4, 19.75, 1.65), moving=2.0)
    replace_dataset(path, "raw data/logs/chi", np.zeros(6))
    return path


def test_add_files_map_run_noted_analyze_off(window: MainWindow, tmp_path: Path) -> None:
    # Arrange
    _ready(window)
    d, tab = window.data, window.data_tab

    # Act
    d.add_files([_map_run(tmp_path / "HB2B_2246.h5")])

    # Assert
    assert "Run 2246: spatial map (one chi, phi orientation)" in tab.map_note.text()
    assert "PyRS strain/stress tools" in tab.map_note.text()
    assert tab.runs_table.item(0, SUBRUNS).text() == "6 (spatial map)"
    assert d.pf_runs() == [] and not tab.analyze_button.isEnabled()
    assert "the loaded runs are spatial maps" in tab.ready_label.text()


def test_run_map_run_left_out(
    window: MainWindow, qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    seen: list[str] = []

    def spy(run: Any, *args: Any, **kwargs: Any) -> list[Any]:
        seen.append(run.run)
        return [synthetic_pole_figure(run.run, "200")]

    monkeypatch.setattr(data_presenter, "analyze_run", spy)
    _ready(window)
    d, tab = window.data, window.data_tab
    d.add_files([_map_run(tmp_path / "HB2B_2246.h5"), write_reduced_run(tmp_path / "HB2B_31.h5")])
    tab.gauge_form.set_values(CENTER)
    d._on_form_changed()

    # Act
    with qtbot.waitSignal(d.analyzed, timeout=60_000):
        tab.analyze_button.click()

    # Assert
    assert seen == ["31"]
    assert "1 spatial map left out" in tab.ready_label.text()


def test_add_files_moving_logs_no_gauge_fill(window: MainWindow, tmp_path: Path) -> None:
    # Arrange
    d, tab = window.data, window.data_tab

    # Act
    d.add_files([write_reduced_run(tmp_path / "HB2B_21.h5", position=(29.4, 19.75, 1.65), moving=0.3)])

    # Assert
    assert d.gauge_center() is None
    assert all(tab.gauge_form.widgets[k].text() == "" for k in CENTER)
    note = tab.gauge_note.text()
    assert "changed during run 21 (0.30 mm)" in note and "does not fill vx, vy, vz" in note


def test_gauge_inside_slab_corner_refused(window: MainWindow) -> None:
    # Arrange
    _ready(window)
    d, tab = window.data, window.data_tab

    # Act
    tab.gauge_form.set_values({"gauge_vx": 0.0, "gauge_vy": 0.0, "gauge_vz": 0.0})
    d._on_form_changed()

    # Assert
    assert not d.gauge_inside()
    assert "outside the sample" in tab.ready_label.text()
    assert "too close to its surface" in tab.gauge_note.text() and "0.1 mm" in tab.gauge_note.text()


def test_show_results_negative_rates_warned(window: MainWindow) -> None:
    # Arrange
    d, tab = window.data, window.data_tab
    d.results = [synthetic_pole_figure("1001", "200", rate=-10.0), synthetic_pole_figure("1002", "111")]

    # Act
    d._show_results()

    # Assert
    text = tab.results_label.text()
    assert "1001 {200}: net counts below background: check the material and peak." in text
    assert "1002 {111}: net counts below" not in text


def test_form_changed_gauge_edit_no_reread(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    calls: list[str] = []
    real = data_presenter.find_peaks

    def counting(run: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(run.run)
        return real(run, *args, **kwargs)

    monkeypatch.setattr(data_presenter, "find_peaks", counting)
    _ready(window)
    d, tab = window.data, window.data_tab
    d.add_files([write_reduced_run(tmp_path / "HB2B_41.h5")])
    calls.clear()

    # Act
    tab.gauge_form.set_values({"gauge_vx": 29.0})
    d._on_form_changed()
    after_gauge = len(calls)
    tab.settings_form.set_values({"central_hw": 4.5})
    d._on_form_changed()

    # Assert
    assert after_gauge == 0 and len(calls) == 1
