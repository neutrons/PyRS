"""
Predict tab: readiness, planes, grid and sampling, a run, plots, cutoff, saving and beam-path GIFs.

The beams come from the Experiment setup tab.
"""

import csv
from pathlib import Path

import pytest
from pytestqt.qtbot import QtBot
from qtpy.QtWidgets import QSpinBox

from pyrs.interface.pole_figures.main_window import MainWindow
from pyrs.interface.pole_figures.presenters.predict import (
    PLOT_TYPES,
    TYPE_GIF,
    TYPE_PLANE,
    TYPE_PROFILE,
    TYPE_SUMMARY,
)
from tests.util.pole_figures_gui_helpers import wait_figure
from tests.util.pole_figures_helpers import SLAB_CENTER_MM, STL_DIR

pytestmark = [pytest.mark.gui, pytest.mark.integration]

STL = STL_DIR / "slab.stl"
CENTER = dict(zip(("gauge_vx", "gauge_vy", "gauge_vz"), SLAB_CENTER_MM, strict=True))


@pytest.fixture
def window(qtbot: QtBot) -> MainWindow:
    """The main window with the IN718 preset chosen."""
    w = MainWindow()
    qtbot.addWidget(w)
    w.material_tab.preset_combo.setCurrentText("IN718 (fcc)")
    return w


def _fast(w: MainWindow) -> None:
    """Coarse grid and few points so a prediction runs in about a second."""
    w.predict_tab.grid_form.set_values({"chi_step": 30.0, "phi_step": 45.0})
    w.predict_tab.mc_form.set_values({"mc_points": 1000})
    w.predict._on_form_changed()


def _ready(w: MainWindow) -> None:
    """STL, gauge at the reference center and a fast grid."""
    w.sample.load_stl(STL)
    w.sample_tab.gauge_form.set_values(CENTER)
    w.sample._on_form_changed()
    _fast(w)


@pytest.fixture
def predicted(window: MainWindow, qtbot: QtBot) -> MainWindow:
    """The window after a fast prediction on the slab."""
    _ready(window)
    with qtbot.waitSignal(window.predict.predicted, timeout=60_000):
        window.predict_tab.predict_button.click()
    return window


def _labels(w: MainWindow) -> list[str]:
    assert w.predict.result is not None
    return [f"{{{q.plane.hkl}}}" for q in w.predict.result.planes]


def test_init_beams_on_setup_grid_on_predict(window: MainWindow) -> None:
    # Arrange
    spin = window.predict_tab.mc_form.widgets["mc_points"]
    assert isinstance(spin, QSpinBox)
    m = window.predict.model
    defaults = (m.chi_step, m.phi_step, m.mc_points)

    # Act
    spin.setValue(20_000)

    # Assert
    b = window.material.beams()
    assert (b.incident_width, b.incident_height, b.diffracted_width) == (4.0, 4.0, 3.0)
    assert "beam_form" in vars(window.material_tab)
    assert defaults == (10.0, 5.0, 5000) and window.predict.model.mc_points == 20_000
    assert "649 orientations" in window.predict_tab.grid_label.text()


def test_beam_change_shown_in_prediction_inputs(window: MainWindow) -> None:
    # Arrange
    window.sample.load_stl(STL)

    # Act
    window.material_tab.beam_form.widgets["incident_width"].setValue(2.0)  # type: ignore[attr-defined]

    # Assert
    assert window.material.beams().incident_width == 2.0
    assert "2 x 4 mm incident" in window.predict_tab.ready_label.text()


def test_readiness_without_stl_not_ready(window: MainWindow) -> None:
    # Arrange
    tab = window.predict_tab

    # Act
    text = tab.ready_label.text()

    # Assert
    assert not tab.predict_button.isEnabled() and "load an STL" in text


def test_planes_follow_material_selection(window: MainWindow) -> None:
    # Arrange
    p = window.predict
    planes = p.planes()

    # Act
    window.material_tab.set_all_checked(False)
    window.material._on_table_changed()

    # Assert
    assert [pl.hkl for pl in planes] == ["200", "220", "311", "222"]  # 2theta 50 to 120 at 1.73 A
    assert all(50 <= pl.two_theta <= 120 for pl in planes)
    assert all(pl.omega == pytest.approx(pl.two_theta / 2, abs=0.006) for pl in planes)
    assert p.planes() == []
    assert "select at least one reflection" in window.predict_tab.ready_label.text()


def test_run_predicts_every_plane(window: MainWindow, qtbot: QtBot) -> None:
    # Arrange
    _ready(window)
    p = window.predict
    assert window.predict_tab.predict_button.isEnabled(), window.predict_tab.ready_label.text()

    # Act
    with qtbot.waitSignal(p.predicted, timeout=60_000):
        window.predict_tab.predict_button.click()

    # Assert
    assert p.result is not None and len(p.result.planes) == len(p.planes())
    assert p.result.planes[0].correction[0] == pytest.approx(1.0)


def test_show_plot_summary_first_at_figure_shape(predicted: MainWindow, qtbot: QtBot) -> None:
    # Arrange
    tab = predicted.predict_tab
    panel = tab.plot_panel

    # Act
    wait_figure(qtbot, panel)

    # Assert: the summary, no plane dropdown, no correction-parts box
    assert panel.types() == PLOT_TYPES and panel.planes() == _labels(predicted)
    assert panel.current() == (TYPE_SUMMARY, "")
    assert panel.plane_combo.isHidden() and tab.parts_box.isHidden()
    assert panel.canvas is not None and len(panel.canvas.figure.axes) == 2 * len(_labels(predicted)) + 2
    w_in, h_in = panel.canvas.figure.get_size_inches()  # the canvas keeps the figure's designed proportions
    assert panel.canvas.width() / panel.canvas.height() == pytest.approx(w_in / h_in, rel=0.02)


def test_show_plot_plane_dropdown_switches_plane(predicted: MainWindow, qtbot: QtBot) -> None:
    # Arrange
    tab = predicted.predict_tab
    panel = tab.plot_panel
    labels = _labels(predicted)

    # Act
    panel.select(TYPE_PLANE, labels[1])
    wait_figure(qtbot, panel)

    # Assert
    assert not panel.plane_combo.isHidden()
    assert panel.canvas is not None and labels[1] in panel.canvas.figure.texts[0].get_text()

    # Act: the plane dropdown alone switches the plot
    panel.plane_combo.setCurrentText(labels[0])
    wait_figure(qtbot, panel)

    # Assert
    assert panel.canvas is not None and labels[0] in panel.canvas.figure.texts[0].get_text()
    assert tab.parts_box.isHidden()


def test_show_plot_profile_parts_checkboxes(predicted: MainWindow, qtbot: QtBot) -> None:
    # Arrange
    tab = predicted.predict_tab
    panel = tab.plot_panel
    panel.select(TYPE_PLANE, _labels(predicted)[0])

    # Act
    panel.type_combo.setCurrentText(TYPE_PROFILE)
    wait_figure(qtbot, panel)

    # Assert
    assert not tab.parts_box.isHidden()
    assert panel.canvas is not None and len(panel.canvas.figure.axes) == 2
    n_lines = len(panel.canvas.figure.axes[0].lines)

    # Act
    tab.part_boxes["volume"].setChecked(False)
    wait_figure(qtbot, panel)

    # Assert
    assert panel.canvas is not None and len(panel.canvas.figure.axes[0].lines) < n_lines


def test_cutoff_lower_ignores_more(predicted: MainWindow, qtbot: QtBot) -> None:
    # Arrange
    p, tab = predicted.predict, predicted.predict_tab
    before = p.sphere_fractions()

    # Act
    tab.cutoff_spin.setValue(1.5)
    wait_figure(qtbot, tab.plot_panel)

    # Assert
    after = p.sphere_fractions()
    assert all(after[h] >= before[h] for h in before)
    assert "Ignored" in tab.sphere_label.text()


def test_save_csv_and_active_plot(predicted: MainWindow, qtbot: QtBot, tmp_path: Path) -> None:
    # Arrange
    p, panel = predicted.predict, predicted.predict_tab.plot_panel
    wait_figure(qtbot, panel)
    panel.zoom.setCurrentText("50%")

    # Act
    panel.save_to(tmp_path / "active.png")
    p.save_csv(tmp_path / "pred.csv")

    # Assert
    assert panel.canvas is not None
    assert panel.canvas.width() == round(panel.canvas.figure.get_size_inches()[0] * 50)
    assert (tmp_path / "active.png").stat().st_size > 10_000
    with open(tmp_path / "pred.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert p.result is not None and len(rows) == len(p.result.planes) * len(p.result.planes[0].chi)
    assert {r["ignored"] for r in rows} <= {"0", "1"}


def test_make_gif_from_preview_note(predicted: MainWindow, qtbot: QtBot, tmp_path: Path) -> None:
    # Arrange: the GIF preview before the GIF exists points to the left panel and sets its plane
    p, tab = predicted.predict, predicted.predict_tab
    panel = tab.plot_panel
    labels = _labels(predicted)
    panel.select(TYPE_GIF, labels[1])
    assert panel.kind == "none" and "Make beam-path GIF" in panel.message()
    assert tab.gif_plane.currentText() == labels[1] and tab.parts_box.isHidden()
    assert tab.gif_button.isEnabled()
    tab.gif_every.setValue(4)
    assert "frames" in tab.gif_label.text()
    run_msg = tab.progress_label.text()

    # Act
    with qtbot.waitSignal(p.gif_made, timeout=120_000):
        tab.gif_button.click()
    panel.save_to(tmp_path / "beam.gif")

    # Assert: progress shows in the GIF section, not the prediction's Run section
    assert tab.progress_label.text() == run_msg
    assert "GIF done" in tab.gif_progress_label.text() and tab.gif_progress.value() == 1000
    assert panel.current() == (TYPE_GIF, labels[1]) and panel.kind == "gif"
    assert (tmp_path / "beam.gif").read_bytes()[:3] == b"GIF"

    # Act
    tab.cutoff_spin.setValue(3.0)

    # Assert: a cutoff change keeps the GIF preview
    assert panel.kind == "gif"


def test_stale_after_gauge_edit(predicted: MainWindow) -> None:
    # Arrange
    sample_form = predicted.sample_tab.gauge_form

    # Act
    sample_form.set_values({"gauge_vx": 20.0})
    predicted.sample._on_form_changed()

    # Assert
    assert "Inputs changed" in predicted.predict_tab.ready_label.text()


def test_readiness_gauge_outside_blocks(window: MainWindow) -> None:
    # Arrange
    window.sample.load_stl(STL)

    # Act
    window.sample_tab.gauge_form.set_values({"gauge_vz": 50.0})
    window.sample._on_form_changed()

    # Assert
    assert not window.predict_tab.predict_button.isEnabled()
    assert "inside the sample" in window.predict_tab.ready_label.text()


def test_cancel_running_prediction_no_result(window: MainWindow, qtbot: QtBot) -> None:
    # Arrange
    window.sample.load_stl(STL)
    window.predict_tab.mc_form.set_values({"mc_points": 20000})
    window.predict._on_form_changed()
    p = window.predict
    p.run()

    # Act
    p.cancel()
    qtbot.waitUntil(lambda: p._worker is None, timeout=60_000)

    # Assert
    assert p.result is None
    assert window.predict_tab.progress_label.text() == "Cancelled."
    assert window.predict_tab.gif_progress_label.text() == ""


def test_readiness_no_material_blocks(qtbot: QtBot) -> None:
    # Arrange
    w = MainWindow()
    qtbot.addWidget(w)

    # Act
    w.sample.load_stl(STL)

    # Assert
    assert not w.predict_tab.predict_button.isEnabled()
    assert "choose a material" in w.predict_tab.ready_label.text()


def test_make_all_gifs_every_plane(predicted: MainWindow, qtbot: QtBot, tmp_path: Path) -> None:
    # Arrange
    p, tab = predicted.predict, predicted.predict_tab
    assert p.result is not None
    n_pl = len(p.result.planes)
    tab.gif_every.setValue(5)
    assert f"for all {n_pl} planes" in tab.gif_label.text() and "typical speed" in tab.gif_label.text()
    assert p.gif_seconds(n_pl) == pytest.approx(n_pl * p.gif_seconds())
    assert tab.gif_all_button.isEnabled() and not tab.gif_save_all_button.isEnabled()
    statuses: list[str] = []
    p.status.connect(statuses.append)

    # Act
    tab.gif_all_button.click()
    started = tab.gif_progress_label.text()
    qtbot.waitUntil(lambda: p._worker is None, timeout=180_000)
    saved = p.save_all_gifs(tmp_path)

    # Assert
    assert "about" in started
    assert len(p.gifs) == n_pl and "All" in tab.gif_progress_label.text()
    assert any(s.startswith(f"Plane {n_pl} of {n_pl}") for s in statuses)  # status bar follows the job
    assert "measured on the last GIF" in tab.gif_label.text()
    assert tab.plot_panel.kind == "gif" and tab.gif_save_all_button.isEnabled()
    assert sorted(x.name for x in saved) == sorted(f"beam_paths_{h}.gif" for h in p.gifs)


def test_cancel_all_gifs_keeps_finished(predicted: MainWindow, qtbot: QtBot) -> None:
    # Arrange
    p, tab = predicted.predict, predicted.predict_tab
    tab.gif_every.setValue(5)
    p.make_all_gifs()
    qtbot.waitUntil(lambda: tab.gif_progress_label.text().startswith("Plane 2"), timeout=120_000)

    # Act
    p.cancel()
    qtbot.waitUntil(lambda: p._worker is None, timeout=120_000)

    # Assert
    assert p.result is not None
    assert "Cancelled" in tab.gif_progress_label.text()
    assert 1 <= len(p.gifs) < len(p.result.planes)


def test_equal_spacing_reaches_table_scan(window: MainWindow, qtbot: QtBot) -> None:
    """The Predict toggle cuts the points per plane, and Table scan plans on the same grid."""
    # Arrange
    w = window
    w.sample.load_stl(STL)
    w.predict_tab.grid_form.set_values({"chi_step": 10.0, "phi_step": 5.0, "equal_spacing": True})
    w.predict_tab.mc_form.set_values({"mc_points": 500})
    w.predict._on_form_changed()
    assert "454 orientations" in w.predict_tab.grid_label.text()
    w.predict_tab.grid_form.set_values({"chi_step": 30.0, "phi_step": 30.0})
    w.predict._on_form_changed()

    # Act
    with qtbot.waitSignal(w.predict.predicted, timeout=120_000):
        w.predict_tab.predict_button.click()

    # Assert
    res = w.predict.result
    assert res is not None and res.grid.equal_spacing
    n = len(res.grid.points()[0])
    assert all(len(q.chi) == n for q in res.planes) and n < len(type(res.grid)(30.0, 30.0).points()[0])
    assert "equal spacing" in w.table_scan_tab.ready_label.text()
    scans = w.table_scan.scans()
    assert scans and all(s.n_grid == n and len(s.points) <= n for s in scans)
