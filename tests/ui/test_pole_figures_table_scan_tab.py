"""Table scan tab: waits for a prediction, plans per plane, settings, preview, writing, stale inputs."""

from pathlib import Path

import openpyxl
import pytest
from pytestqt.qtbot import QtBot

from pyrs.interface.pole_figures.main_window import TAB_NAMES, MainWindow
from pyrs.interface.pole_figures.models.table_scan import FIXED
from tests.util.pole_figures_helpers import SLAB_CENTER_MM, STL_DIR

pytestmark = [pytest.mark.gui, pytest.mark.integration]

STL = STL_DIR / "slab.stl"
CENTER = dict(zip(("gauge_vx", "gauge_vy", "gauge_vz"), SLAB_CENTER_MM, strict=True))


@pytest.fixture
def window(qtbot: QtBot) -> MainWindow:
    """IN718, the slab with the gauge at the reference center, and a coarse grid (no prediction yet)."""
    w = MainWindow()
    qtbot.addWidget(w)
    w.material_tab.preset_combo.setCurrentText("IN718 (fcc)")
    w.sample.load_stl(STL)
    w.sample_tab.gauge_form.set_values(CENTER)
    w.sample._on_form_changed()
    w.predict_tab.grid_form.set_values({"chi_step": 30.0, "phi_step": 30.0})
    w.predict_tab.mc_form.set_values({"mc_points": 1000})
    w.predict._on_form_changed()
    return w


def _predict(w: MainWindow, qtbot: QtBot) -> None:
    with qtbot.waitSignal(w.predict.predicted, timeout=60_000):
        w.predict_tab.predict_button.click()


@pytest.fixture
def planned(window: MainWindow, qtbot: QtBot) -> MainWindow:
    """The window after a prediction at cutoff 2.5."""
    window.predict_tab.cutoff_spin.setValue(2.5)
    _predict(window, qtbot)
    return window


def test_tab_names_table_scan_after_predict() -> None:
    # Arrange
    names = list(TAB_NAMES)

    # Act
    index = names.index("Table scan")

    # Assert
    assert index == names.index("Predict") + 1


def test_refresh_without_prediction_waits(window: MainWindow) -> None:
    # Arrange
    tab = window.table_scan_tab

    # Act
    scans = window.table_scan.scans()

    # Assert
    assert "Predict first" in tab.ready_label.text()
    assert not tab.write_button.isEnabled() and tab.table.rowCount() == 0
    assert scans == []


def test_predicted_lists_planes_and_preview(planned: MainWindow) -> None:
    # Arrange
    t, tab = planned.table_scan, planned.table_scan_tab
    assert planned.predict.result is not None

    # Act
    hkls = [q.plane.hkl for q in planned.predict.result.planes]

    # Assert
    assert tab.hkls() == hkls and tab.write_button.isEnabled()
    assert t.model.cutoff == 2.5  # copied from Predict
    assert " of " in tab.table.item(0, 4).text() and "files" in tab.summary.text()
    panel = tab.plot_panel
    assert panel.current()[0] == "Table scan points" and panel.kind == "figure"
    assert panel.canvas is not None and "table scan" in panel.canvas.figure.axes[0].get_title(loc="left")
    assert not hasattr(planned.predict_tab, "write_scans_button")  # the Predict tab has no table-scan controls


def test_scans_follow_settings(planned: MainWindow) -> None:
    """Base time, fixed mode, cap, cutoff and order all change the plan."""
    # Arrange
    t, tab = planned.table_scan, planned.table_scan_tab
    form = tab.settings_form

    # Act
    tab.base_spins()[0].setValue(20.0)

    # Assert
    assert t.scans()[0].base_s == 20.0
    scaled_total = t.scans()[0].total_s

    # Act
    form.widgets["time_mode"].setCurrentText(FIXED)  # type: ignore[attr-defined]

    # Assert
    sc = t.scans()[0]
    assert not sc.scaled and all(p.count_s == 20 for p in sc.points) and sc.total_s <= scaled_total
    assert not form.widgets["max_s"].isEnabled()

    # Act
    form.widgets["cap_on"].setChecked(True)  # type: ignore[attr-defined]
    form.widgets["max_s"].setValue(15.0)  # type: ignore[attr-defined]

    # Assert
    assert form.widgets["max_s"].isEnabled() and max(p.count_s for p in t.scans()[0].points) == 15
    n = len(t.scans()[0].points)

    # Act
    form.widgets["cutoff"].setValue(10.0)  # type: ignore[attr-defined]
    form.widgets["serpentine"].setChecked(False)  # type: ignore[attr-defined]

    # Assert
    assert len(t.scans()[0].points) >= n and not t.scans()[0].serpentine


def test_write_one_workbook_per_plane(planned: MainWindow, tmp_path: Path) -> None:
    # Arrange: fixed 20 s, capped at 15 s, with a run title
    t, tab = planned.table_scan, planned.table_scan_tab
    form = tab.settings_form
    tab.base_spins()[0].setValue(20.0)
    form.widgets["time_mode"].setCurrentText(FIXED)  # type: ignore[attr-defined]
    form.widgets["cap_on"].setChecked(True)  # type: ignore[attr-defined]
    form.widgets["max_s"].setValue(15.0)  # type: ignore[attr-defined]
    form.widgets["run_title"].setText("IN718 plate")  # type: ignore[attr-defined]
    form.widgets["run_title"].editingFinished.emit()  # type: ignore[attr-defined]
    hkls = tab.hkls()

    # Act
    paths = t.write(tmp_path)

    # Assert
    assert [p.name for p in paths] == [f"IN718_plate_{h}.xlsx" for h in hkls]
    rows = list(openpyxl.load_workbook(paths[0]).active.iter_rows(values_only=True))
    assert str(rows[1][0]).startswith("IN718 plate | {" + hkls[0] + "} pole figure")
    assert all(r[7] == 15 for r in rows[2:])


def test_refresh_stale_prediction_blocks_write(planned: MainWindow) -> None:
    # Arrange
    tab = planned.table_scan_tab

    # Act
    planned.sample_tab.gauge_form.set_values({"gauge_vx": 20.0})
    planned.sample._on_form_changed()

    # Assert
    assert "Predict again" in tab.ready_label.text() and not tab.write_button.isEnabled()


def test_scans_chi90_plate_uses_phi_90(window: MainWindow, qtbot: QtBot) -> None:
    """On the reference plate at chi 90 the phi axis is along Q, and phi 90 has the shortest path (the +vx pole)."""
    # Arrange
    assert window.predict.model.best_phi  # on by default in the GUI

    # Act
    _predict(window, qtbot)

    # Assert
    for sc in window.table_scan.scans():
        last = sc.points[-1]
        assert (last.chi, last.phi) == (90.0, 90.0)
        assert last.correction < 1.5
