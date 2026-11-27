"""
A form built from Pydantic fields: titles become labels, bounds become spin-box limits, Literals become combos.

An optional float (``float | None``) is a text box that starts blank and takes numbers only (C locale: a point for
the decimal, no group separators). Blank reads as None. Text that is not a valid number in range also reads as None,
and ``invalid()`` names those fields so the presenter can say so. Every edit is sent, so the model never keeps a value
the box no longer shows.

Typical workflow:

1. Build a form for some fields of a tab model.
2. Show the model's values with :meth:`ModelForm.set_values`.
3. On ``changed``, validate ``model.model_copy(update=form.values())`` and report :meth:`ModelForm.invalid`.

Example:
    >>> from pyrs.interface.pole_figures.models.predict import GRID_FIELDS, PredictModel
    >>> form = ModelForm(PredictModel, GRID_FIELDS, decimals=1)  # doctest: +SKIP
    >>> form.set_values(PredictModel().model_dump(include=set(GRID_FIELDS)))  # doctest: +SKIP
    >>> form.values()  # doctest: +SKIP
    {'chi_step': 10.0, 'phi_step': 5.0, 'equal_spacing': False}
"""

import types
import typing
from typing import Any

import annotated_types as at
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from qtpy.QtCore import QLocale, Signal  # type: ignore[attr-defined]
from qtpy.QtGui import QDoubleValidator
from qtpy.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QLineEdit, QSpinBox, QWidget


def _bounds(info: FieldInfo, decimals: int) -> tuple[float, float]:
    """Spin-box range from the field bounds. A strict bound moves in by one display step."""
    lo, hi = -1e9, 1e9
    step = 10.0**-decimals
    for m in info.metadata:
        if isinstance(m, at.Ge):
            lo = float(m.ge)  # type: ignore[arg-type]
        elif isinstance(m, at.Gt):
            lo = float(m.gt) + step  # type: ignore[arg-type]
        elif isinstance(m, at.Le):
            hi = float(m.le)  # type: ignore[arg-type]
        elif isinstance(m, at.Lt):
            hi = float(m.lt) - step  # type: ignore[arg-type]
    return lo, hi


def _optional_float(ann: Any) -> bool:
    """True for ``float | None`` (or ``Optional[float]``)."""
    return typing.get_origin(ann) in (typing.Union, types.UnionType) and set(typing.get_args(ann)) == {
        float,
        type(None),
    }


def _number_locale() -> QLocale:
    """C locale without group separators, so the validator accepts exactly what the parser reads."""
    loc = QLocale.c()
    loc.setNumberOptions(QLocale.NumberOption.RejectGroupSeparator)
    return loc


def format_optional(value: float, decimals: int) -> str:
    """
    Fixed-point text with at most ``decimals`` places (no exponent), trailing zeros removed.

    Args:
        value: The number.
        decimals: Most decimal places shown.

    Returns:
        The text, "0" for zero or negative zero.

    Example:
        >>> format_optional(29.400, 3)
        '29.4'
        >>> format_optional(-0.0001, 3)
        '0'
    """
    text = f"{round(float(value), decimals):.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


class ModelForm(QWidget):
    """
    Edit a subset of a Pydantic model's fields.

    Widgets by field type: Literal -> combo box, bool -> check box, float -> spin box, ``float | None`` -> text
    box that takes numbers only, int -> integer spin box, anything else -> text box. The field ``title`` is the
    label and its ``description`` the tooltip.

    Args:
        model_cls: Model whose fields are shown.
        names: Field names, in display order.
        decimals: Decimals for float fields (default: 3).
        parent: Parent widget, or None (default: None).

    Attributes:
        changed: Emitted when the user edits any field.
        decimals: Decimals for float fields.
        widgets: Input widget by field name.
    """

    changed = Signal()

    def __init__(self, model_cls: type[BaseModel], names: list[str], *, decimals: int = 3,
                 parent: QWidget | None = None) -> None:  # fmt: skip
        super().__init__(parent)
        self.decimals = decimals
        self.widgets: dict[str, QWidget] = {}
        layout = QFormLayout(self)
        for name in names:
            info = model_cls.model_fields[name]
            ann = info.annotation
            w: QWidget
            if typing.get_origin(ann) is typing.Literal:
                combo = QComboBox()
                combo.addItems([str(a) for a in typing.get_args(ann)])
                combo.currentTextChanged.connect(self.changed)
                w = combo
            elif ann is bool:
                box = QCheckBox()
                box.toggled.connect(self.changed)
                w = box
            elif ann is float:
                spin = QDoubleSpinBox()
                spin.setDecimals(decimals)
                spin.setRange(*_bounds(info, decimals))
                spin.setKeyboardTracking(False)  # emit when the user finishes typing, not on every key
                spin.valueChanged.connect(self.changed)
                w = spin
            elif _optional_float(ann):
                opt = QLineEdit()
                lo, hi = _bounds(info, decimals)
                validator = QDoubleValidator(lo, hi, decimals, opt)
                validator.setNotation(QDoubleValidator.Notation.StandardNotation)
                validator.setLocale(_number_locale())
                opt.setValidator(validator)
                opt.setPlaceholderText("required")
                opt.setProperty("optional_float", True)
                # textEdited, not editingFinished: Qt sends editingFinished only for acceptable text, so clearing a
                # box or typing an out-of-range value would leave the old value in the model
                opt.textEdited.connect(self.changed)
                w = opt
            elif ann is int:
                ispin = QSpinBox()
                lo, hi = _bounds(info, 0)
                ispin.setRange(int(max(lo, -(2**31) + 1)), int(min(hi, 2**31 - 1)))
                ispin.setGroupSeparatorShown(True)
                ispin.setKeyboardTracking(False)
                ispin.valueChanged.connect(self.changed)
                w = ispin
            else:
                line = QLineEdit()
                line.editingFinished.connect(self.changed)
                w = line
            if info.description:
                w.setToolTip(info.description)
            layout.addRow(info.title or name, w)
            self.widgets[name] = w

    def values(self) -> dict[str, Any]:
        """The shown values by field name. A blank or invalid optional-float box gives None."""
        out: dict[str, Any] = {}
        for name, w in self.widgets.items():
            if isinstance(w, QComboBox):
                out[name] = w.currentText()
            elif isinstance(w, QCheckBox):
                out[name] = w.isChecked()
            elif isinstance(w, (QDoubleSpinBox, QSpinBox)):
                out[name] = w.value()
            elif isinstance(w, QLineEdit) and w.property("optional_float"):
                out[name] = _read_optional(w)
            elif isinstance(w, QLineEdit):
                out[name] = w.text()
        return out

    def set_values(self, values: dict[str, Any]) -> None:
        """
        Show values without emitting ``changed``.

        Args:
            values: Values by field name. Names without a widget in this form are skipped.
        """
        for name, v in values.items():
            w = self.widgets.get(name)
            if w is None:
                continue
            w.blockSignals(True)
            if isinstance(w, QComboBox):
                w.setCurrentText(str(v))
            elif isinstance(w, QCheckBox):
                w.setChecked(bool(v))
            elif isinstance(w, QDoubleSpinBox):
                w.setValue(float(v))
            elif isinstance(w, QSpinBox):
                w.setValue(int(v))
            elif isinstance(w, QLineEdit) and w.property("optional_float"):
                w.setText("" if v is None else format_optional(v, self.decimals))
            elif isinstance(w, QLineEdit):
                w.setText(str(v))
            w.blockSignals(False)

    def invalid(self) -> list[str]:
        """Optional-float fields whose text is not blank but is not a valid number in range."""
        return [name for name, w in self.widgets.items()
                if isinstance(w, QLineEdit) and w.property("optional_float") and w.text().strip()
                and _read_optional(w) is None]  # fmt: skip


def _read_optional(w: QLineEdit) -> float | None:
    """The number in an optional-float box, or None when it is blank or not acceptable (parsed in the C locale)."""
    text = w.text().strip()
    if not text or not w.hasAcceptableInput():
        return None
    res = _number_locale().toDouble(text)
    value, ok = res if isinstance(res, tuple) else (res, True)
    return float(value) if ok else None
