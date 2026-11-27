"""
Experiment setup presenter: presets, CIF, custom edits, wavelength, beams, attenuation and reflections.

Typical workflow:

1. The user picks a preset, loads a CIF with :meth:`MaterialPresenter.load_cif`, or chooses Custom.
2. Edits to a phase field turn the phase into Custom. A monochromator setting fills in its wavelength.
3. :meth:`MaterialPresenter.refresh` recomputes the reflections and mu at the wavelength.
4. The predict and data presenters read :meth:`MaterialPresenter.phase`, :meth:`MaterialPresenter.mu`,
   :meth:`MaterialPresenter.beams` and :meth:`MaterialPresenter.selected_reflections`.

Example:
    >>> presenter = MaterialPresenter(MaterialTab())  # doctest: +SKIP
    >>> presenter.load_cif("phase.cif")  # doctest: +SKIP
    True
    >>> presenter.mu()  # 1/mm at the current wavelength  # doctest: +SKIP
"""

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from qtpy.QtCore import QObject, Signal  # type: ignore[attr-defined]

from pyrs.pole_figures.core.correction import Beams
from pyrs.pole_figures.core.hb2b import TWO_THETA_MAX, TWO_THETA_MIN, two_theta_allowed
from pyrs.pole_figures.core.neutron_xs import format_composition, parse_composition
from pyrs.pole_figures.core.phase import AVERAGE, Phase, presets
from pyrs.pole_figures.core.reflections import Reflection
from ..models.material import CUSTOM, PHASE_FIELDS, SELECT, SETTING_WAVELENGTHS, MaterialModel
from ..tabs.material_tab import MaterialTab


def _phase_values(p: Phase) -> dict[str, Any]:
    a, b, c, al, be, ga = p.cell
    return {
        "name": p.name, "space_group": p.space_group, "a": a, "b": b, "c": c, "alpha": al, "beta": be, "gamma": ga,
        "composition": format_composition({k: round(v, 4) for k, v in p.composition.items()}),
        "density_g_cm3": round(p.density_g_cm3, 4),
    }  # fmt: skip


def _sites_text(p: Phase) -> str:
    parts = [
        ("average atom" if s.element == AVERAGE else s.element) + f" at ({s.x:.4g}, {s.y:.4g}, {s.z:.4g})"
        for s in p.sites
    ]
    return f"{len(parts)} site{'s' if len(parts) > 1 else ''}: " + "; ".join(parts)


class MaterialPresenter(QObject):
    """
    Logic of the Experiment setup tab.

    Args:
        view: The Experiment setup tab.
        model: Starting inputs (``MaterialModel()`` for the defaults).

    Attributes:
        changed: Signal: the phase, wavelength, beams, mu or reflection choice changed.
        status: Signal with a short message for the status bar.
        view: The Experiment setup tab.
        model: The current inputs.
    """

    changed = Signal()
    status = Signal(str)

    def __init__(self, view: MaterialTab, *, model: MaterialModel) -> None:
        super().__init__(view)
        self.view = view
        self.model = model
        self._base: Phase = presets().get(self.model.preset) or next(iter(presets().values()))
        self._unchecked: set[str] = set()
        self._reflections: list[Reflection] = []
        self._mu_computed = float("nan")
        self._last_dir = str(Path.cwd())

        view.preset_combo.addItems([SELECT, *presets(), CUSTOM])
        view.preset_combo.setCurrentText(self.model.preset)
        for form in view.forms:
            form.set_values(self.model.model_dump())
            form.changed.connect(self._on_form_changed)
        view.preset_combo.currentTextChanged.connect(self._on_preset)
        view.cif_button.clicked.connect(self._on_cif)
        view.table.itemChanged.connect(self._on_table_changed)
        view.all_button.clicked.connect(lambda: self._check_all(True))
        view.none_button.clicked.connect(lambda: self._check_all(False))
        if self.model.preset in presets():
            self._apply_phase(self._base, preset=self.model.preset)
        else:
            self.refresh()

    def _apply_phase(self, phase: Phase, *, preset: str, cif_path: str = "") -> None:
        self._base = phase
        self._unchecked = set()
        vals = _phase_values(phase)
        self.model = self.model.model_copy(update={**vals, "preset": preset, "cif_path": cif_path})
        self.view.phase_form.set_values(vals)
        self.view.preset_combo.blockSignals(True)
        self.view.preset_combo.setCurrentText(preset)
        self.view.preset_combo.blockSignals(False)
        self.refresh()

    def _on_preset(self, name: str) -> None:
        if name in presets():
            self._apply_phase(presets()[name], preset=name)
        else:
            self.model = self.model.model_copy(update={"preset": SELECT if name == SELECT else CUSTOM, "cif_path": ""})
            self.refresh()

    def _on_cif(self) -> None:
        path = self.view.ask_cif_path(self._last_dir)
        if path:
            self.load_cif(path)

    def load_cif(self, path: str | Path) -> bool:
        """
        Use the phase of a CIF file (as Custom).

        Args:
            path: The CIF file.

        Returns:
            True when loaded, False (after showing the error on the tab) when it cannot be read.
        """
        try:
            phase = Phase.from_cif(path)
        except (OSError, ValueError, RuntimeError) as exc:
            self.view.error_label.setText(f"Could not read the CIF {Path(path).name}: {exc}")
            return False
        self._last_dir = str(Path(path).parent)
        self._apply_phase(phase, preset=CUSTOM, cif_path=str(path))
        self.status.emit(f"Loaded {phase.name} from {Path(path).name}")
        return True

    def _on_form_changed(self) -> None:
        vals: dict[str, Any] = {}
        for form in self.view.forms:
            vals.update(form.values())
        old = self.model
        # editing a phase field turns a preset into a custom phase (its sites are kept)
        if old.preset != CUSTOM and any(vals[k] != getattr(old, k) for k in PHASE_FIELDS):
            vals["preset"] = CUSTOM
            self.view.preset_combo.blockSignals(True)
            self.view.preset_combo.setCurrentText(CUSTOM)
            self.view.preset_combo.blockSignals(False)
        # a monochromator setting fills in its wavelength; typing another wavelength means Custom
        if vals["wavelength_setting"] != old.wavelength_setting and vals["wavelength_setting"] in SETTING_WAVELENGTHS:
            vals["wavelength_A"] = SETTING_WAVELENGTHS[vals["wavelength_setting"]]
        elif (
            vals["wavelength_A"] != old.wavelength_A
            and SETTING_WAVELENGTHS.get(vals["wavelength_setting"]) != vals["wavelength_A"]
        ):
            vals["wavelength_setting"] = CUSTOM
        # switching the override on starts from the computed value
        if vals["mu_override"] and not old.mu_override and math.isfinite(self._mu_computed):
            vals["mu_value"] = round(self._mu_computed, 4)
        try:
            self.model = MaterialModel.model_validate({**old.model_dump(), **vals})
        except ValidationError as exc:
            self.view.error_label.setText(f"Invalid input: {exc.errors()[0]['msg']}")
            return
        self.view.wavelength_form.set_values(self.model.model_dump())
        self.view.mu_form.set_values(self.model.model_dump())
        self.refresh()

    def _on_table_changed(self) -> None:
        checked = set(self.view.checked_labels())
        usable = {r.label for r in self._reflections if two_theta_allowed(r.two_theta)}
        self._unchecked = (self._unchecked - usable) | (usable - checked)  # out-of-range choices are kept
        self._store_selection()

    def _check_all(self, on: bool) -> None:
        self.view.set_all_checked(on)
        self._on_table_changed()

    def _store_selection(self) -> None:
        sel = [r.label for r in self._reflections if r.label not in self._unchecked and two_theta_allowed(r.two_theta)]
        self.model = self.model.model_copy(update={"selected_hkl": sel})
        self.changed.emit()

    def phase(self) -> Phase:
        """
        The phase as currently edited: the preset or CIF phase (with its sites) with the form values.

        Returns:
            The phase.

        Raises:
            ValueError: If the composition text cannot be parsed or the edited values do not make a valid phase.
        """
        m = self.model
        return replace(
            self._base,
            name=m.name,
            space_group=m.space_group,
            cell=(m.a, m.b, m.c, m.alpha, m.beta, m.gamma),
            composition=parse_composition(m.composition),
            density_g_cm3=m.density_g_cm3,
        )

    def has_material(self) -> bool:
        """False until the user picks a preset, loads a CIF or chooses Custom."""
        return self.model.preset != SELECT

    def mu(self) -> float:
        """Attenuation mu (1/mm) used for the corrections: the override, or the computed value (NaN if none)."""
        return self.model.mu_value if self.model.mu_override else self._mu_computed

    def beams(self) -> Beams:
        """Incident beam and receiving slit sizes (mm)."""
        return self.model.beams()

    def selected_reflections(self) -> list[Reflection]:
        """The checked reflections whose 2theta is in the allowed range, in table order."""
        return [r for r in self._reflections if r.label in set(self.model.selected_hkl)]

    def refresh(self) -> None:
        """Recompute the reflections and mu, refill the table and notes, and store the selection."""
        m = self.model
        self.view.error_label.setText("")
        self.view.set_inputs_enabled(self.has_material())
        if not self.has_material():
            self._reflections, self._mu_computed = [], float("nan")
            self.view.set_reflections([], checked=set(), disabled=set())
            self.view.source_label.setText("Choose a preset, load a CIF, or choose Custom to enter a phase.")
            self.view.source_label.setToolTip("")
            self.view.mu_label.setText("mu: choose a material first.")
            self.view.table_note.setText("")
            self._store_selection()
            return
        try:
            phase = self.phase()
            refl = phase.reflections(m.wavelength_A)
            self._mu_computed = phase.mu_per_mm(m.wavelength_A)
        except (ValueError, RuntimeError, KeyError) as exc:
            self._reflections, self._mu_computed = [], float("nan")
            self.view.set_reflections([], checked=set(), disabled=set())
            self.view.error_label.setText(str(exc).strip("'\""))
            self.view.mu_label.setText("mu: not available")
            self._store_selection()
            return
        self._reflections = refl
        disabled = {r.label for r in refl if not two_theta_allowed(r.two_theta)}
        checked = {r.label for r in refl} - self._unchecked - disabled
        self.view.set_reflections(refl, checked=checked, disabled=disabled)
        src = "CIF " + Path(m.cif_path).name if m.cif_path else (phase.source or "custom")
        where = f"From {Path(m.cif_path).name}. " if m.cif_path else ""
        self.view.source_label.setText(f"{where}{_sites_text(phase)} <i>(hover for the data source)</i>")
        self.view.source_label.setToolTip(src)
        used = "override in use" if m.mu_override else "in use"
        self.view.mu_label.setText(
            f"Computed mu = <b>{self._mu_computed:.4f} /mm</b> at {m.wavelength_A:g} A "
            f"(scattering + absorption, NIST cross sections). {used.capitalize()}: {self.mu():.4f} /mm"
        )
        n_ok = len(refl) - len(disabled)
        self.view.table_note.setText(
            f"{len(refl)} reflection families at {m.wavelength_A:g} A, up to 2theta 179 deg. {n_ok} are inside the "
            f"allowed 2theta range {TWO_THETA_MIN:g} to {TWO_THETA_MAX:g} deg. Gray rows are outside it and cannot be "
            "used. The HB-2B detector covers about 17 deg of 2theta around the arm angle."
        )
        self._store_selection()
