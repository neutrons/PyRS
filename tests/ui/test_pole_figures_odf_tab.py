"""
MTEX tab: pole figures from Data and Corrections, readiness, a run (fake ODF step), plots, summary and saving.

The tab comes from ``gui/odf_tab.py``, the optional ODF part. Most tests replace the ODF step (``run_odf``) with a fake
that writes the same files, so they run without MTEX for Python (pymtex). The fit against the stored reference values
is in ``tests/integration/test_pole_figures_odf_reference.py``. ``test_run_real_odf_shows_plots`` runs the real step
through the tab and is skipped where pymtex is not installed.
"""

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from matplotlib.figure import Figure
from pytestqt.qtbot import QtBot

from pyrs.pole_figures.core import mtex as core_mtex
from pyrs.pole_figures.core import odf as core_odf
from pyrs.pole_figures.core.polefig import PolePeak
from pyrs.interface.pole_figures.main_window import TAB_NAMES, MainWindow, odf_tab_installed
from pyrs.interface.pole_figures.odf_tab import ODF_TAB_NAME
from pyrs.interface.pole_figures.presenters import mtex as mtex_presenter
from pyrs.interface.pole_figures.presenters.mtex import PYMTEX_MISSING, MtexPresenter
from pyrs.interface.pole_figures.tabs.mtex_tab import MtexTab
from pyrs.pole_figures.io.odf import OdfCancelledError
from pyrs.pole_figures.io.polefig_csv import write_all
from tests.util.pole_figures_helpers import synthetic_pole_figure

pytestmark = [pytest.mark.gui, pytest.mark.integration]

MakeWindow = Callable[..., MainWindow]


@pytest.fixture
def make_window(qtbot: QtBot, monkeypatch: pytest.MonkeyPatch) -> MakeWindow:
    """Build the main window with the IN718 preset, as if pymtex were installed (``pymtex=True``) or not."""

    def make(*, pymtex: bool) -> MainWindow:
        monkeypatch.setattr(core_odf, "pymtex_installed", lambda: pymtex)
        w = MainWindow()
        qtbot.addWidget(w)
        w.material_tab.preset_combo.setCurrentText("IN718 (fcc)")
        return w

    return make


@pytest.fixture
def window(make_window: MakeWindow) -> MainWindow:
    """The main window with the IN718 preset chosen, as if pymtex were installed."""
    return make_window(pymtex=True)


def _odf(w: MainWindow) -> tuple[MtexTab, MtexPresenter]:
    """The MTEX tab and its presenter."""
    tab = w.findChild(MtexTab)
    assert tab is not None
    presenter = tab.findChild(MtexPresenter)
    assert presenter is not None
    return tab, presenter


def _two_pole_figures() -> list[PolePeak]:
    return [synthetic_pole_figure("1001", "200"), synthetic_pole_figure("1002", "111", seed=1)]


def fake_odf(lines: list[str]) -> Callable[..., Path]:
    """Stand-in for io.odf.run_odf: writes three PNGs, the summary and plot data, like the real step."""

    def run(work: Path, pole_figures: list[PolePeak], *, phase_name: str, cell: Any, view_from: str, vz_name: str,
            opts: core_mtex.MtexOptions, on_stage: Callable[[str], None],
            should_stop: Callable[[], bool]) -> Path:  # fmt: skip
        for key in ("start", "read", "odf", "summary"):
            on_stage(key)
        out = Path(work) / "mtex"
        out.mkdir(parents=True)
        csvs = write_all(work, pole_figures)
        s = core_mtex.settings(csvs, out_dir=out, phase_name=phase_name, cell=cell, view_from=view_from,
                               vz_name=vz_name, opts=opts)  # fmt: skip
        lines.append(json.dumps(s))
        for name in ("odf_phi2_45.png", "odf_phi2_45_white.png", "inverse_pole_figures.png"):
            fig = Figure(figsize=(2, 2))
            fig.add_subplot().plot([0, 1], [0, 1])
            fig.savefig(out / name)
        (out / "odf_summary.txt").write_text("ODF max 2.80 mrd\n", encoding="utf-8")
        if s["mtex"]["plot"].get("export_data"):
            (out / "plot_data").mkdir()
            (out / "plot_data" / "pole_figures_recalculated.csv").write_text("hkl,f_mrd\n111,1.0\n", encoding="utf-8")
            (out / "plot_data" / "README.txt").write_text("plot data\n", encoding="utf-8")
        return out

    return run


def test_init_odf_tab_last(window: MainWindow) -> None:
    # Arrange
    tabs = window.tabs

    # Act
    names = [tabs.tabText(i) for i in range(tabs.count())]

    # Assert
    assert odf_tab_installed() and names == [*TAB_NAMES, ODF_TAB_NAME]


def test_init_without_pymtex_note_and_run_off(make_window: MakeWindow) -> None:
    """Without pymtex the tab is there, says how pymtex is installed, and Run stays off even with pole figures."""
    # Arrange
    w = make_window(pymtex=False)
    tab, presenter = _odf(w)

    # Act
    w.data.pole_figures_changed.emit(_two_pole_figures())

    # Assert
    assert w.tabs.tabText(w.tabs.count() - 1) == ODF_TAB_NAME
    assert not tab.install_note.isHidden() and tab.install_note.text() == PYMTEX_MISSING
    assert "installed separately on the analysis cluster" in PYMTEX_MISSING
    assert not tab.run_button.isEnabled() and "install MTEX for Python (pymtex)" in tab.ready_label.text()
    assert presenter.missing() == ["install MTEX for Python (pymtex)"]


def test_init_with_pymtex_no_note(window: MainWindow) -> None:
    # Arrange
    tab, _presenter = _odf(window)

    # Act
    window.data.pole_figures_changed.emit(_two_pole_figures())

    # Assert
    assert tab.install_note.isHidden() and tab.run_button.isEnabled(), tab.ready_label.text()


def test_readiness_needs_pole_figures(window: MainWindow) -> None:
    # Arrange
    tab, _presenter = _odf(window)

    # Act
    text = tab.ready_label.text()

    # Assert
    assert "analyze runs" in text and not tab.run_button.isEnabled()
    assert "m-3m" in tab.symmetry_label.text()


def test_crystal_symmetry_manual_choice_flagged(window: MainWindow) -> None:
    """Auto takes the Laue class from the space group. A manual choice is flagged when it differs."""
    # Arrange
    tab, m = _odf(window)
    assert m.model.crystal_symmetry == "Auto (from space group)" and m.crystal_symmetry() == "m-3m"

    # Act
    tab.odf_form.set_values({"crystal_symmetry": "6/mmm"})
    m._on_form_changed()

    # Assert
    assert m.crystal_symmetry() == "6/mmm" and "override" in tab.symmetry_label.text()


def test_run_fake_odf_plots_and_saves(
    window: MainWindow, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    seen: list[str] = []
    monkeypatch.setattr(mtex_presenter, "run_odf", fake_odf(seen))
    good = _two_pole_figures()
    third = synthetic_pole_figure("1003", "220", seed=2)
    one_ring = replace(third, gauge=replace(third.gauge, correction=np.where(np.abs(good[0].chi) < 1e-9, 1.0, 99.0)))
    tab, m = _odf(window)
    window.data.pole_figures_changed.emit([*good, one_ring])
    assert tab.pf_table.rowCount() == 3
    assert [r.label for r in m.selected()] == ["1001 {200}", "1002 {111}"]  # the one-ring PF is left out
    assert "fewer than 2 chi rings" in tab.pf_table.item(2, 4).text()
    assert tab.run_button.isEnabled(), tab.ready_label.text()
    tab.plot_form.set_values({"pf_hkl": "111, 200", "ipf_vy": False, "cmax_pf": 3.0})
    m._on_form_changed()

    # Act
    with qtbot.waitSignal(m.finished, timeout=30_000):
        tab.run_button.click()

    # Assert
    s = json.loads(seen[0])
    assert [Path(f).name for f in s["csv_files"]] == ["pole_figure_1001_200.csv", "pole_figure_1002_111.csv"]
    assert s["mtex"]["plot"]["pf_hkl"] == ["111", "200"] and s["mtex"]["ipf_directions"] == ["vx", "vz"]
    assert s["mtex"]["plot"]["cmax_pf"] == 3.0 and s["mtex"]["plot"]["cmax_odf"] == "data"
    assert s["mtex"]["crystal_symmetry"] == "m-3m" and s["cell"][:3] == [3.597, 3.597, 3.597]
    assert "ODF max 2.80" in tab.summary.toPlainText() and "Fitting the ODF" in tab.log.toPlainText()
    assert tab.progress.value() == 1000 and "MTEX finished: 2 plots, 1 plot-data CSVs" in tab.status_label.text()
    assert s["mtex"]["plot"]["export_data"] is True and tab.save_data_button.isEnabled()
    panel = tab.plot_panel
    assert panel.types() == ["Square", "Inverse pole figures"]
    assert panel.kind == "image" and panel.current()[0] == "Square"
    assert not tab.white_toggle.isHidden()  # Square has a white-below-random version

    # Act
    copied = sorted(p.name for p in m.save_plot_data(tmp_path / "data"))
    tab.white_toggle.setChecked(True)

    # Assert
    assert copied == ["README.txt", "pole_figures_recalculated.csv"]
    assert panel.image_path is not None and panel.image_path.name == "odf_phi2_45_white.png"

    # Act
    panel.select("Inverse pole figures")

    # Assert
    assert panel.image_path is not None and panel.image_path.name == "inverse_pole_figures.png"
    assert tab.white_toggle.isHidden()  # no white version of the IPFs

    # Act
    saved = m.save_results(tmp_path / "res")
    window.data.pole_figures_changed.emit(good + [synthetic_pole_figure("1004", "220", seed=3)])

    # Assert: results saved, and new pole figures from Data and Corrections are flagged against the last run
    assert (saved / "pole_figure_1001_200.csv").is_file() and (saved / "mtex" / "odf_summary.txt").is_file()
    assert "changed since the last MTEX run" in tab.ready_label.text()


def test_run_cancel_shows_cancelled(window: MainWindow, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancel asks the step to stop. The step's OdfCancelledError ends the run as Cancelled, not as a failure."""

    # Arrange
    def cancelled(*_args: Any, should_stop: Callable[[], bool], **_kw: Any) -> Path:
        raise OdfCancelledError("cancelled before the step 'start'")

    monkeypatch.setattr(mtex_presenter, "run_odf", cancelled)
    tab, m = _odf(window)
    window.data.pole_figures_changed.emit(_two_pole_figures())

    # Act
    tab.run_button.click()
    qtbot.waitUntil(lambda: "Cancelled" in tab.status_label.text(), timeout=10_000)

    # Assert
    assert m.out_dir is None and tab.run_button.isEnabled() and not tab.cancel_button.isEnabled()


def test_run_real_odf_shows_plots(window: MainWindow, qtbot: QtBot) -> None:
    """The real ODF step through the tab on two synthetic pole figures: plots appear (needs pymtex)."""
    # Arrange
    pytest.importorskip("mtex", reason="MTEX for Python (pymtex) is an optional dependency, installed separately")
    tab, m = _odf(window)
    window.data.pole_figures_changed.emit(_two_pole_figures())
    assert tab.run_button.isEnabled(), tab.ready_label.text()

    # Act
    with qtbot.waitSignal(m.finished, timeout=120_000):
        tab.run_button.click()

    # Assert
    assert m.out_dir is not None and (m.out_dir / "odf_summary.txt").is_file()
    assert tab.plot_panel.types() == [n for n, _, _ in core_mtex.OUTPUT_PLOTS]
    assert "pymtex" in tab.summary.toPlainText()
