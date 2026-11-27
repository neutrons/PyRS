"""
Data and Corrections tab.

Loading reduced runs, readiness, analysis, plots, the style toggle, cutoff, export, beam-path GIFs of the measured
points and the gauge position.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.image import AxesImage
from pytestqt.qtbot import QtBot

from pyrs.interface.pole_figures.main_window import TAB_NAMES, MainWindow
from pyrs.interface.pole_figures.presenters.data import TYPE_ALL, TYPE_GIF, TYPE_ONE, gif_name
from pyrs.interface.pole_figures.tabs.data_tab import RUN_COLUMNS
from pyrs.interface.pole_figures.widgets.toggle_switch import LabeledToggle
from tests.util.pole_figures_gui_helpers import wait_figure
from tests.util.pole_figures_helpers import STL_DIR, TEST_DATA, needs_data, synthetic_pole_figure, write_reduced_run

pytestmark = [pytest.mark.gui, pytest.mark.integration]

STL = STL_DIR / "slab.stl"
POS = RUN_COLUMNS.index("vx, vy, vz (mm)")
CENTER = {"gauge_vx": 29.4, "gauge_vy": 19.75, "gauge_vz": 1.65}
GAUGE_KEYS = ("gauge_vx", "gauge_vy", "gauge_vz")


@pytest.fixture
def window(qtbot: QtBot) -> MainWindow:
    """A new main window."""
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _ready(w: MainWindow) -> None:
    """IN718, the slab, the gauge at the reference center on both tabs (not logged in the reference runs)."""
    w.material_tab.preset_combo.setCurrentText("IN718 (fcc)")
    w.sample.load_stl(STL)
    w.sample_tab.gauge_form.set_values(CENTER)
    w.sample._on_form_changed()
    w.data_tab.settings_form.set_values({"mc_points": 1000})
    w.data_tab.gauge_form.set_values(CENTER)
    w.data._on_form_changed()


def _with_results(w: MainWindow) -> None:
    """Two synthetic measured pole figures in place of an analysis."""
    _ready(w)
    w.data.results = [synthetic_pole_figure("1001", "200"), synthetic_pole_figure("1002", "111", seed=1)]
    w.data._new_pole_figures()
    w.data.update_readiness()


def test_tab_names_include_data_tab() -> None:
    # Arrange
    names = list(TAB_NAMES)

    # Act
    found = "Data and Corrections" in names

    # Assert
    assert found


def test_labeled_toggle_click_and_set(qtbot: QtBot) -> None:
    # Arrange
    t = LabeledToggle("Point cells", "Interpolated")
    qtbot.addWidget(t)
    seen: list[bool] = []
    t.toggled.connect(seen.append)
    assert not t.isChecked()

    # Act
    t.switch.click()

    # Assert
    assert t.isChecked() and seen == [True]

    # Act
    t.setChecked(False)

    # Assert
    assert seen == [True, False]


def test_readiness_needs_runs_sample_and_material(window: MainWindow) -> None:
    # Arrange
    tab = window.data_tab

    # Act
    text = tab.ready_label.text()

    # Assert
    assert "add reduced runs" in text and "load the STL" in text and "choose a material" in text
    assert not tab.analyze_button.isEnabled()


def test_add_files_bad_file_reported(window: MainWindow, tmp_path: Path) -> None:
    # Arrange
    bad = tmp_path / "HB2B_1.h5"
    bad.write_text("not hdf5")

    # Act
    errors = window.data.add_files([bad])

    # Assert
    assert errors and "cannot read" in errors[0] and window.data.runs == []


@needs_data
def test_run_reference_runs_plot_cutoff_export(window: MainWindow, qtbot: QtBot, tmp_path: Path) -> None:
    # Arrange
    _ready(window)
    d, tab = window.data, window.data_tab
    errors = d.add_files([TEST_DATA / "HB2B_6497.h5", TEST_DATA / "HB2B_6494.h5", TEST_DATA / "HB2B_6494.h5"])
    assert errors == []
    assert [r.run for r in d.runs] == ["6494", "6497"]  # sorted, duplicate skipped
    assert tab.runs_table.rowCount() == 2
    assert tab.runs_table.item(0, 7).text() == "{311} (edge), {222}"  # PF also lists and skips the edge {311}
    assert tab.runs_table.item(0, 6).text().startswith("0.14")  # mu at the run's wavelength
    assert tab.analyze_button.isEnabled(), tab.ready_label.text()
    panel = tab.plot_panel

    # Act
    with qtbot.waitSignal(d.analyzed, timeout=120_000):
        tab.analyze_button.click()
    wait_figure(qtbot, panel)

    # Assert
    assert [r.label for r in d.results] == ["6494 {222}", "6497 {200}"]
    assert all(r.beams.incident_width == 4.0 and r.beams.diffracted_width == 3.0 for r in d.results)
    assert panel.current() == (TYPE_ALL, "")
    assert panel.canvas is not None and len(panel.canvas.figure.axes) == 2 * 5  # 3 pole figures + 2 color bars

    # Act: one pole figure
    panel.select(TYPE_ONE, "6497 {200}")
    wait_figure(qtbot, panel)

    # Assert
    assert panel.canvas is not None and len(panel.canvas.figure.axes) == 5
    assert not any(isinstance(a, AxesImage) for a in panel.canvas.figure.axes[0].get_children())

    # Act: the interpolated style
    tab.style_toggle.setChecked(True)
    wait_figure(qtbot, panel)

    # Assert
    assert panel.canvas is not None
    assert any(isinstance(a, AxesImage) for a in panel.canvas.figure.axes[0].get_children())

    # Act: a lower cutoff ignores more points without a new analysis
    before = int(d.results[1].ignored.sum())
    tab.settings_form.set_values({"cutoff": 2.0})
    d._on_form_changed()

    # Assert
    assert int(d.results[1].ignored.sum()) > before and not d.stale()

    # Act
    paths = d.export(tmp_path)

    # Assert
    names = sorted(p.name for p in paths)
    assert names == sorted(["pole_figure_6494_222.csv", "pole_figure_6497_200.csv", "pole_figures.png",
                            "pole_figures_interpolated.png"])  # fmt: skip
    header = (tmp_path / "pole_figure_6494_222.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("run,hkl,chi,phi,pole_vx")
    assert header.endswith("corrected_per_s,corrected_sigma_per_s,ignored")

    # Act: a changed receiving slit
    tab.settings_form.set_values({"receiving_slit_mm": 2.0})
    d._on_form_changed()

    # Assert: the analysis is stale
    assert "Analyze again" in tab.ready_label.text()


def test_update_readiness_before_results_no_gif(window: MainWindow) -> None:
    # Arrange
    _ready(window)
    tab = window.data_tab

    # Act
    window.data.update_readiness()

    # Assert
    assert not tab.gif_button.isEnabled() and "Analyze runs first" in tab.gif_label.text()
    assert tab.style_toggle.isHidden()  # no plot yet


def test_new_pole_figures_list_gif_choices(window: MainWindow) -> None:
    # Arrange
    tab = window.data_tab

    # Act
    _with_results(window)

    # Assert
    assert [tab.gif_pf.itemText(i) for i in range(tab.gif_pf.count())] == ["1001 {200}", "1002 {111}"]
    assert tab.gif_button.isEnabled() and "frames for 1001 {200}" in tab.gif_label.text()
    assert tab.plot_panel.current()[0] == TYPE_ALL and not tab.style_toggle.isHidden()


def test_show_plot_gif_type_asks_for_gif(window: MainWindow) -> None:
    # Arrange
    _with_results(window)
    tab = window.data_tab

    # Act
    tab.plot_panel.select(TYPE_GIF, "1002 {111}")

    # Assert: a note until a GIF is made; the cells / interpolated toggle does not change a GIF
    assert "Make GIF for this pole figure" in tab.plot_panel.message()
    assert tab.gif_pf.currentText() == "1002 {111}" and tab.style_toggle.isHidden()

    # Act
    tab.plot_panel.select(TYPE_ONE, "1001 {200}")

    # Assert
    assert not tab.style_toggle.isHidden()


def test_make_gif_one_all_save_and_clear(window: MainWindow, qtbot: QtBot, tmp_path: Path) -> None:
    """GIFs of a measured pole figure (synthetic here): one, all, save, cleared by a cutoff change."""
    # Arrange
    _with_results(window)
    d, tab = window.data, window.data_tab
    tab.gif_every.setValue(6)
    tab.gif_dpi.setValue(40)
    tab.plot_panel.select(TYPE_GIF, "1002 {111}")

    # Act
    with qtbot.waitSignal(d.status, timeout=120_000, check_params_cb=lambda m: "ready" in m):
        tab.gif_button.click()

    # Assert
    assert list(d.gifs) == ["1002 {111}"] and tab.plot_panel.kind == "gif"
    assert "GIF done" in tab.gif_progress_label.text() and tab.progress_label.text() == ""

    # Act
    with qtbot.waitSignal(d.status, timeout=120_000, check_params_cb=lambda m: "GIFs ready" in m):
        tab.gif_all_button.click()
    names = sorted(p.name for p in d.save_all_gifs(tmp_path))

    # Assert
    assert sorted(d.gifs) == ["1001 {200}", "1002 {111}"]
    assert names == ["beam_paths_1001_200.gif", "beam_paths_1002_111.gif"]

    # Act
    tab.settings_form.set_values({"cutoff": 2.0})
    d._on_form_changed()

    # Assert
    assert d.gifs == {} and not tab.gif_save_all_button.isEnabled()


def test_gif_name_label_made_safe() -> None:
    # Arrange
    label = "6494 {222}"

    # Act
    name = gif_name(label)

    # Assert
    assert name == "beam_paths_6494_222.gif"


def test_gauge_center_starts_blank_and_required(window: MainWindow) -> None:
    # Arrange
    window.material_tab.preset_combo.setCurrentText("IN718 (fcc)")
    window.sample.load_stl(STL)
    d, tab = window.data, window.data_tab
    assert d.gauge_center() is None
    assert all(tab.gauge_form.widgets[k].text() == "" for k in GAUGE_KEYS)
    assert "enter the gauge position" in tab.ready_label.text() and "Not set" in tab.gauge_note.text()

    # Act: two of three
    tab.gauge_form.set_values({"gauge_vx": 29.4, "gauge_vy": 19.75})
    d._on_form_changed()

    # Assert: all three are needed
    assert d.gauge_center() is None

    # Act
    tab.gauge_form.set_values({"gauge_vz": 500.0})
    d._on_form_changed()

    # Assert
    assert "outside the sample" in tab.ready_label.text()

    # Act
    tab.gauge_form.set_values({"gauge_vz": 1.65})
    d._on_form_changed()

    # Assert
    assert (
        np.allclose(np.asarray(d.gauge_center(), dtype=float), [29.4, 19.75, 1.65])
        and "gauge position" not in tab.ready_label.text()
    )


def test_add_files_gauge_fills_from_run_logs(window: MainWindow, tmp_path: Path) -> None:
    # Arrange
    d, tab = window.data, window.data_tab

    # Act
    d.add_files([write_reduced_run(tmp_path / "HB2B_11.h5", position=(29.4, 19.75, 1.65))])

    # Assert
    assert (
        np.allclose(np.asarray(d.gauge_center(), dtype=float), [29.4, 19.75, 1.65])
        and "From the logs of run 11" in tab.gauge_note.text()
    )
    assert tab.runs_table.item(0, POS).text() == "29.4, 19.75, 1.65"

    # Act: a second run at another position and one without logs
    d.add_files([write_reduced_run(tmp_path / "HB2B_12.h5", position=(30.0, 19.75, 1.65)),
                 write_reduced_run(tmp_path / "HB2B_13.h5", position=None)])  # fmt: skip

    # Assert: both flagged, the first run's position kept
    note = tab.gauge_note.text()
    assert "logged different positions" in note and "11 (29.4, 19.75, 1.65); 12 (30, 19.75, 1.65)" in note
    assert "No vx, vy, vz logged in run 13" in note and tab.runs_table.item(2, POS).text() == "not logged"
    assert np.allclose(np.asarray(d.gauge_center(), dtype=float), [29.4, 19.75, 1.65])

    # Act: removing the runs that gave the values
    d.clear_runs()

    # Assert: blank again
    assert d.gauge_center() is None

    # Act
    d.add_files([write_reduced_run(tmp_path / "HB2B_13.h5", position=None)])

    # Assert: NaN logs leave it blank, to be filled
    assert d.gauge_center() is None


def test_add_files_typed_gauge_kept_and_compared(window: MainWindow, tmp_path: Path) -> None:
    # Arrange
    d, tab = window.data, window.data_tab
    tab.gauge_form.set_values({"gauge_vx": 28.0, "gauge_vy": 19.75, "gauge_vz": 1.65})
    d._on_form_changed()

    # Act
    d.add_files([write_reduced_run(tmp_path / "HB2B_21.h5", position=(29.4, 19.75, 1.65), moving=0.3)])

    # Assert: the user's values win, and the logs are compared with them
    assert np.allclose(np.asarray(d.gauge_center(), dtype=float), [28.0, 19.75, 1.65])
    note = tab.gauge_note.text()
    assert "Differs from the position logged in run 21" in note and "changed during run 21 (0.30 mm)" in note
    assert "(moves 0.30)" in tab.runs_table.item(0, POS).text()


def test_inherit_prediction_center_fills_gauge(window: MainWindow) -> None:
    # Arrange
    d, tab = window.data, window.data_tab
    assert not tab.gauge_from_prediction_button.isEnabled()  # no prediction yet
    window.predict.result = SimpleNamespace(center=np.array([12.0, 3.5, 0.75]))  # type: ignore[assignment]
    d.update_readiness()  # what the predicted signal triggers (the other tabs need a full Prediction)
    assert tab.gauge_from_prediction_button.isEnabled()

    # Act
    tab.gauge_from_prediction_button.click()

    # Assert
    assert (
        np.allclose(np.asarray(d.gauge_center(), dtype=float), [12.0, 3.5, 0.75])
        and "From the prediction" in tab.gauge_note.text()
    )
