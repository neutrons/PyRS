"""The gauge position of the Data and Corrections tab: what the analysis uses, typed input, fills and refusals."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pytestqt.qtbot import QtBot
from qtpy.QtCore import Qt  # type: ignore[attr-defined]

import pyrs.interface.pole_figures.presenters.data as data_presenter
from pyrs.interface.pole_figures.main_window import MainWindow
from pyrs.interface.pole_figures.tabs.data_tab import RUN_COLUMNS
from tests.util.pole_figures_helpers import STL_DIR, synthetic_pole_figure, write_reduced_run

pytestmark = [pytest.mark.gui, pytest.mark.integration]

STL = STL_DIR / "slab.stl"
SAMPLE_CENTER = {"gauge_vx": 20.0, "gauge_vy": 15.0, "gauge_vz": 1.65}  # what Predict uses
DATA_CENTER = {"gauge_vx": 29.4, "gauge_vy": 19.75, "gauge_vz": 1.65}  # where the runs were measured
GAUGE_KEYS = ("gauge_vx", "gauge_vy", "gauge_vz")


@pytest.fixture
def window(qtbot: QtBot) -> MainWindow:
    """A new main window."""
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _two_centers(w: MainWindow, tmp_path: Path) -> None:
    """STL, material, one unlogged run, and different gauge points on the Sample and Data tabs."""
    w.material_tab.preset_combo.setCurrentText("IN718 (fcc)")
    w.sample.load_stl(STL)
    w.sample_tab.gauge_form.set_values(SAMPLE_CENTER)
    w.sample._on_form_changed()
    w.data.add_files([write_reduced_run(tmp_path / "HB2B_31.h5", position=None)])
    w.data_tab.gauge_form.set_values(DATA_CENTER)
    w.data._on_form_changed()


def _type(qtbot: QtBot, box: Any, text: str) -> None:
    """Select all in ``box`` and type ``text`` as the user would (an empty text clears it)."""
    box.setFocus()
    box.selectAll()
    if text:
        qtbot.keyClicks(box, text)
    else:
        qtbot.keyClick(box, Qt.Key.Key_Delete)


def test_run_uses_data_tab_position(
    window: MainWindow, qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    seen: list[np.ndarray] = []

    def spy(run: Any, cands: Any, mesh: Any, center: Any, *args: Any, **kwargs: Any) -> list[Any]:
        seen.append(np.asarray(center))
        return [synthetic_pole_figure(run.run, "200")]

    monkeypatch.setattr(data_presenter, "analyze_run", spy)
    _two_centers(window, tmp_path)
    d, tab = window.data, window.data_tab
    assert tab.analyze_button.isEnabled(), tab.ready_label.text()

    # Act
    with qtbot.waitSignal(d.analyzed, timeout=60_000):
        tab.analyze_button.click()

    # Assert: not the Sample tab's (20, 15), for the analysis and the GIFs
    assert len(seen) == 1 and np.allclose(seen[0], [29.4, 19.75, 1.65])
    assert np.allclose(d._gif_job_parts()[1], [29.4, 19.75, 1.65])

    # Act: moving the Sample tab center (Predict)
    window.sample_tab.gauge_form.set_values({"gauge_vx": 21.0})
    window.sample._on_form_changed()

    # Assert: the data analysis is not stale
    assert not d.stale()

    # Act: moving the Data tab center
    tab.gauge_form.set_values({"gauge_vx": 28.0})
    d._on_form_changed()

    # Assert
    assert d.stale() and "Analyze again" in tab.ready_label.text()


def test_add_files_logged_values_kept_as_shown(window: MainWindow, tmp_path: Path) -> None:
    """Values with more decimals than the box, and encoder noise near zero, must not read back as blank."""
    # Arrange
    d, tab = window.data, window.data_tab
    path = write_reduced_run(tmp_path / "HB2B_41.h5", position=(0.523812, 19.123456, -2.3e-7))

    # Act
    d.add_files([path])

    # Assert
    c = d.gauge_center()
    assert c is not None and np.allclose(c, [0.5238, 19.1235, 0.0])
    assert [tab.gauge_form.widgets[k].text() for k in GAUGE_KEYS] == ["0.5238", "19.1235", "0"]


def test_typing_cleared_or_invalid_reaches_model(window: MainWindow, qtbot: QtBot, tmp_path: Path) -> None:
    # Arrange
    _two_centers(window, tmp_path)
    window.show()
    d, tab = window.data, window.data_tab
    vx = tab.gauge_form.widgets["gauge_vx"]

    # Act: cleared, the old 29.4 must not stay in the model
    _type(qtbot, vx, "")

    # Assert
    assert d.gauge_center() is None and not tab.analyze_button.isEnabled()
    assert "Not set" in tab.gauge_note.text()

    # Act: the comma is refused as it is typed (no crash)
    _type(qtbot, vx, "1,000")

    # Assert
    assert vx.text() == "1000" and np.allclose(np.asarray(d.gauge_center(), dtype=float), [1000.0, 19.75, 1.65])

    # Act: pasted or set text that is not a valid number
    vx.setText("1,000")
    vx.textEdited.emit(vx.text())

    # Assert: blank in the model, and said so
    assert d.gauge_center() is None and "vx: not a valid number" in tab.gauge_note.text()
    assert "vx is not a valid number" in tab.ready_label.text()

    # Act
    _type(qtbot, vx, "28.5")

    # Assert
    assert (
        np.allclose(np.asarray(d.gauge_center(), dtype=float), [28.5, 19.75, 1.65]) and tab.analyze_button.isEnabled()
    )

    # Act: a change on another form
    tab.settings_form.set_values({"cutoff": 2.5})
    d._on_form_changed()

    # Assert: the typed value is kept
    assert np.allclose(np.asarray(d.gauge_center(), dtype=float), [28.5, 19.75, 1.65])


def test_remove_runs_refills_blanks_or_keeps(window: MainWindow, tmp_path: Path) -> None:
    # Arrange
    d, tab = window.data, window.data_tab
    d.add_files([write_reduced_run(tmp_path / "HB2B_51.h5", position=(10.0, 5.0, 1.0)),
                 write_reduced_run(tmp_path / "HB2B_52.h5", position=(12.0, 5.0, 1.0))])  # fmt: skip
    assert (
        np.allclose(np.asarray(d.gauge_center(), dtype=float), [10.0, 5.0, 1.0])
        and "different positions" in tab.gauge_note.text()
    )

    # Act: remove run 51, whose position was used
    tab.runs_table.selectRow(0)
    tab.remove_button.click()

    # Assert: refilled from run 52
    assert (
        np.allclose(np.asarray(d.gauge_center(), dtype=float), [12.0, 5.0, 1.0])
        and "From the logs of run 52" in tab.gauge_note.text()
    )
    assert "different positions" not in tab.gauge_note.text()

    # Act: add an unlogged run and remove run 52, the logged one
    d.add_files([write_reduced_run(tmp_path / "HB2B_53.h5", position=None)])
    tab.runs_table.selectRow(0)
    tab.remove_button.click()

    # Assert: only an unlogged run is left
    assert d.gauge_center() is None

    # Act: typed values, then every run removed
    tab.gauge_form.set_values({"gauge_vx": 1.0, "gauge_vy": 2.0, "gauge_vz": 1.0})
    d._on_form_changed()
    d.clear_runs()

    # Assert: typed values survive removing runs
    assert np.allclose(np.asarray(d.gauge_center(), dtype=float), [1.0, 2.0, 1.0])


def test_add_files_earlier_run_keeps_position(window: MainWindow, tmp_path: Path) -> None:
    # Arrange
    d, tab = window.data, window.data_tab
    d.add_files([write_reduced_run(tmp_path / "HB2B_62.h5", position=(12.0, 5.0, 1.0))])

    # Act: run 61 sorts first
    d.add_files([write_reduced_run(tmp_path / "HB2B_61.h5", position=(10.0, 5.0, 1.0))])

    # Assert
    assert (
        np.allclose(np.asarray(d.gauge_center(), dtype=float), [12.0, 5.0, 1.0])
        and "From the logs of run 62" in tab.gauge_note.text()
    )
    assert "different positions" in tab.gauge_note.text()


def test_add_files_prediction_source_kept(window: MainWindow, tmp_path: Path) -> None:
    # Arrange
    d, tab = window.data, window.data_tab
    tab.gauge_from_prediction_button.click()
    assert d.gauge_center() is None  # no prediction: nothing happens
    window.predict.result = SimpleNamespace(center=np.array([12.0, 3.5, 0.75]))  # type: ignore[assignment]
    d.inherit_prediction_center()

    # Act
    d.add_files([write_reduced_run(tmp_path / "HB2B_71.h5", position=(29.4, 19.75, 1.65))])

    # Assert
    assert np.allclose(np.asarray(d.gauge_center(), dtype=float), [12.0, 3.5, 0.75])
    note = tab.gauge_note.text()
    assert "From the prediction" in note and "Differs from the position logged in run 71" in note


def test_add_files_out_of_range_position_refused(window: MainWindow, tmp_path: Path) -> None:
    # Arrange
    d, tab = window.data, window.data_tab
    path = write_reduced_run(tmp_path / "HB2B_81.h5", position=(20000.0, 5.0, 1.0))

    # Act
    d.add_files([path])

    # Assert: blank, the boxes in step with the model, and the reason shown
    assert d.gauge_center() is None
    assert all(tab.gauge_form.widgets[k].text() == "" for k in GAUGE_KEYS)
    assert "position from run 81 (20000, 5, 1) was refused" in tab.gauge_note.text()


def test_add_files_incomplete_position_flagged(window: MainWindow, tmp_path: Path) -> None:
    # Arrange
    d, tab = window.data, window.data_tab
    path = write_reduced_run(tmp_path / "HB2B_91.h5", position=(10.0, 5.0, 1.0), keys=("vx", "vy"))

    # Act
    d.add_files([path])

    # Assert
    assert d.gauge_center() is None and "Run 91: vz not logged." in tab.gauge_note.text()
    assert "(incomplete)" in tab.runs_table.item(0, RUN_COLUMNS.index("vx, vy, vz (mm)")).text()


def test_form_changed_clears_refused_fill_note(window: MainWindow, tmp_path: Path) -> None:
    # Arrange
    d, tab = window.data, window.data_tab
    d.add_files([write_reduced_run(tmp_path / "HB2B_82.h5", position=(20000.0, 5.0, 1.0))])
    assert "was refused" in tab.gauge_note.text()

    # Act
    tab.gauge_form.set_values({"gauge_vx": 10.0, "gauge_vy": 5.0, "gauge_vz": 1.0})
    d._on_form_changed()

    # Assert
    assert "was refused" not in tab.gauge_note.text() and np.allclose(
        np.asarray(d.gauge_center(), dtype=float), [10.0, 5.0, 1.0]
    )
