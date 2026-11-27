"""Experiment setup tab: presets, custom edits, CIF, wavelength, mu override, reflection choice."""

import typing
from pathlib import Path

import pytest
from pytestqt.qtbot import QtBot
from qtpy.QtCore import Qt  # type: ignore[attr-defined]
from qtpy.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit

from pyrs.interface.pole_figures.models.material import (
    CUSTOM,
    SELECT,
    SETTING_WAVELENGTHS,
    MaterialModel,
    WavelengthSetting,
)
from pyrs.interface.pole_figures.presenters.material import MaterialPresenter
from pyrs.interface.pole_figures.tabs.material_tab import MaterialTab
from tests.util.pole_figures_helpers import CU_CIF

pytestmark = [pytest.mark.gui, pytest.mark.integration]


@pytest.fixture
def blank(qtbot: QtBot) -> MaterialPresenter:
    """The presenter as the window starts it: no material chosen."""
    tab = MaterialTab()
    qtbot.addWidget(tab)
    return MaterialPresenter(tab, model=MaterialModel())


@pytest.fixture
def presenter(blank: MaterialPresenter) -> MaterialPresenter:
    """The presenter with the IN718 preset chosen."""
    blank.view.preset_combo.setCurrentText("IN718 (fcc)")
    return blank


def _spin(p: MaterialPresenter, form: str, name: str) -> QDoubleSpinBox:
    w = getattr(p.view, form).widgets[name]
    assert isinstance(w, QDoubleSpinBox)
    return w


def _inputs(tab: MaterialTab) -> tuple[typing.Any, ...]:
    return (*tab.forms, tab.table, tab.all_button, tab.none_button)


def test_init_no_material_inputs_grayed(blank: MaterialPresenter) -> None:
    # Arrange
    tab = blank.view

    # Act
    enabled = [w.isEnabled() for w in _inputs(tab)]

    # Assert
    assert tab.preset_combo.currentText() == SELECT and not blank.has_material()
    assert tab.table.rowCount() == 0 and blank.model.selected_hkl == []
    assert not any(enabled) and tab.cif_button.isEnabled()


def test_preset_chosen_then_cleared_grays_inputs(blank: MaterialPresenter) -> None:
    # Arrange
    tab = blank.view

    # Act
    tab.preset_combo.setCurrentText("Nickel (fcc)")

    # Assert
    assert blank.has_material() and tab.table.rowCount() > 0
    assert all(w.isEnabled() for w in _inputs(tab))

    # Act
    tab.preset_combo.setCurrentText(SELECT)

    # Assert
    assert not blank.has_material() and not tab.table.isEnabled() and blank.model.selected_hkl == []


def test_preset_in718_allowed_reflections(presenter: MaterialPresenter) -> None:
    """At 1.73 A, {111} (49.2 deg) and {400} and up are outside 50 to 120 deg: the four planes of runs 6494-6497."""
    # Arrange
    model = presenter.model

    # Act
    selected = presenter.selected_reflections()

    # Assert
    assert model.preset == "IN718 (fcc)" and model.selected_hkl == ["200", "220", "311", "222"]
    assert all(50 <= r.two_theta <= 120 for r in selected)
    assert presenter.view.table.rowCount() > len(model.selected_hkl)  # out-of-range rows still listed
    assert presenter.mu() == pytest.approx(0.140, abs=0.003)


def test_set_reflections_out_of_range_not_checkable(presenter: MaterialPresenter) -> None:
    # Arrange
    t = presenter.view.table
    row = [t.item(i, 1).text() for i in range(t.rowCount())].index("{111}")
    item = t.item(row, 0)

    # Act
    presenter.view.all_button.click()

    # Assert
    assert not item.flags() & Qt.ItemFlag.ItemIsUserCheckable
    assert item.checkState() == Qt.CheckState.Unchecked and "outside" in item.toolTip()
    assert "111" not in presenter.model.selected_hkl
    assert "allowed 2theta range 50 to 120" in presenter.view.table_note.text()


def test_wavelength_longer_brings_111_into_range(presenter: MaterialPresenter) -> None:
    # Arrange: 1.89 A puts {111} at 54 deg (at 1.54 A it is at 43.8 deg, still out)
    spin = _spin(presenter, "wavelength_form", "wavelength_A")
    presenter.view.all_button.click()

    # Act
    spin.setValue(1.89)

    # Assert
    assert "111" in presenter.model.selected_hkl and "222" not in presenter.model.selected_hkl


def test_preset_switch_bcc_reflections(presenter: MaterialPresenter) -> None:
    # Arrange
    combo = presenter.view.preset_combo

    # Act
    combo.setCurrentText("Ferritic steel (bcc)")

    # Assert
    assert presenter.model.space_group == "Im-3m"
    assert presenter.model.selected_hkl[:3] == ["110", "200", "211"]


def test_wavelength_setting_lists_core_table() -> None:
    """The dropdown lists every monochromator setting of core.hb2b.HB2B_WAVELENGTHS, then Custom."""
    # Arrange
    expected = (*SETTING_WAVELENGTHS, CUSTOM)

    # Act
    listed = typing.get_args(WavelengthSetting)

    # Assert
    assert listed == expected


def test_wavelength_setting_then_custom_value(presenter: MaterialPresenter) -> None:
    # Arrange
    combo = presenter.view.wavelength_form.widgets["wavelength_setting"]
    assert isinstance(combo, QComboBox)

    # Act
    combo.setCurrentText("Si422 (1.54 A)")

    # Assert
    assert presenter.model.wavelength_A == pytest.approx(1.54)

    # Act
    _spin(presenter, "wavelength_form", "wavelength_A").setValue(1.7213)

    # Assert
    assert presenter.model.wavelength_setting == CUSTOM and combo.currentText() == CUSTOM


def test_phase_edit_preset_becomes_custom(presenter: MaterialPresenter) -> None:
    # Arrange
    spin = _spin(presenter, "phase_form", "a")

    # Act
    spin.setValue(3.6)

    # Assert
    assert presenter.model.preset == CUSTOM
    assert presenter.view.preset_combo.currentText() == CUSTOM
    assert presenter.phase().cell[0] == pytest.approx(3.6)


def test_table_unchecked_survives_wavelength_change(presenter: MaterialPresenter) -> None:
    # Arrange
    presenter.view.table.item(1, 0).setCheckState(Qt.CheckState.Unchecked)  # {200}
    assert "200" not in presenter.model.selected_hkl

    # Act
    _spin(presenter, "wavelength_form", "wavelength_A").setValue(1.54)

    # Assert
    assert "200" not in presenter.model.selected_hkl and "220" in presenter.model.selected_hkl


def test_all_and_none_buttons_check_rows(presenter: MaterialPresenter) -> None:
    # Arrange
    presenter.view.table.item(1, 0).setCheckState(Qt.CheckState.Unchecked)  # {200}

    # Act
    presenter.view.all_button.click()

    # Assert
    assert "200" in presenter.model.selected_hkl

    # Act
    presenter.view.none_button.click()

    # Assert
    assert presenter.model.selected_hkl == []


def test_mu_override_starts_from_computed(presenter: MaterialPresenter) -> None:
    # Arrange
    box = presenter.view.mu_form.widgets["mu_override"]
    assert isinstance(box, QCheckBox)

    # Act
    box.setChecked(True)

    # Assert
    assert presenter.model.mu_value == pytest.approx(presenter.mu())

    # Act
    _spin(presenter, "mu_form", "mu_value").setValue(0.2)

    # Assert
    assert presenter.mu() == pytest.approx(0.2)


def test_composition_unknown_element_shows_error(presenter: MaterialPresenter) -> None:
    # Arrange
    line = presenter.view.phase_form.widgets["composition"]
    assert isinstance(line, QLineEdit)

    # Act
    line.setText("Xx:100")
    line.editingFinished.emit()

    # Assert
    assert "not in the neutron" in presenter.view.error_label.text()
    assert presenter.view.table.rowCount() == 0


def test_load_cif_copper_custom_phase(presenter: MaterialPresenter, tmp_path: Path) -> None:
    # Arrange
    f = tmp_path / "cu.cif"
    f.write_text(CU_CIF)

    # Act
    loaded = presenter.load_cif(f)

    # Assert
    assert loaded
    assert presenter.model.preset == CUSTOM and presenter.model.cif_path == str(f)
    assert "cu.cif" in presenter.view.source_label.text()
    assert presenter.model.selected_hkl[:2] == ["200", "220"]  # {111} of Cu is at 49 deg


def test_load_cif_bad_file_shows_error(presenter: MaterialPresenter, tmp_path: Path) -> None:
    # Arrange
    bad = tmp_path / "bad.cif"
    bad.write_text("data_x\n")

    # Act
    loaded = presenter.load_cif(bad)

    # Assert
    assert not loaded and "Could not read" in presenter.view.error_label.text()
