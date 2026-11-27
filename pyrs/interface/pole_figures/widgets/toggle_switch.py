"""
A sliding toggle switch with a label on each side: left = off, right = on.

Use :class:`LabeledToggle` for a choice between two named options and connect its ``toggled`` signal.

Example:
    >>> toggle = LabeledToggle("Raw", "Corrected")  # doctest: +SKIP
    >>> toggle.toggled.connect(print)  # doctest: +SKIP
    >>> toggle.setChecked(True)  # doctest: +SKIP
    True
"""

from qtpy.QtCore import Property, QPropertyAnimation, QRectF, QSize, Qt, Signal  # type: ignore[attr-defined]
from qtpy.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent
from qtpy.QtWidgets import QAbstractButton, QHBoxLayout, QLabel, QWidget

TRACK_OFF = "#9aa6b2"
TRACK_ON = "#2a78d6"
KNOB = "#ffffff"


class ToggleSwitch(QAbstractButton):
    """
    Checkable switch drawn as a track with a sliding knob.

    Args:
        parent: Parent widget, or None (default: None).

    Attributes:
        knob: Knob position from 0 (off, left) to 1 (on, right). Animated when the switch toggles.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pos = 0.0
        self._anim = QPropertyAnimation(self, b"knob")
        self._anim.setDuration(140)
        self.toggled.connect(self._animate)

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        """Preferred size: 46 x 24 px."""
        return QSize(46, 24)

    def _get_knob(self) -> float:
        return self._pos

    def _set_knob(self, v: float) -> None:
        self._pos = float(v)
        self.update()

    knob = Property(float, _get_knob, _set_knob)

    def _animate(self, on: bool) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._pos)
        self._anim.setEndValue(1.0 if on else 0.0)
        self._anim.start()

    def paintEvent(self, _event: QPaintEvent) -> None:  # noqa: N802 (Qt override)
        """Draw the track, colored between off and on by the knob position, and the knob."""
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        r = h / 2
        off, on = QColor(TRACK_OFF), QColor(TRACK_ON)
        t = self._pos
        track = QColor(int(off.red() + t * (on.red() - off.red())), int(off.green() + t * (on.green() - off.green())),
                       int(off.blue() + t * (on.blue() - off.blue())))  # fmt: skip
        if not self.isEnabled():
            track.setAlpha(110)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(QRectF(0, 0, w, h), r, r)
        d = h - 4
        x = 2 + t * (w - d - 4)
        p.setBrush(QColor(KNOB))
        p.drawEllipse(QRectF(x, 2, d, d))
        p.end()


class LabeledToggle(QWidget):
    """
    ``left [switch] right``. ``toggled(bool)`` is True when the right option is chosen. Labels are clickable.

    Args:
        left: Label of the off option.
        right: Label of the on option.
        parent: Parent widget, or None (default: None).

    Attributes:
        switch: The :class:`ToggleSwitch`.
        left_label: Label of the off option (bold while chosen).
        right_label: Label of the on option (bold while chosen).
    """

    toggled = Signal(bool)

    def __init__(self, left: str, right: str, *, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.switch = ToggleSwitch()
        self.left_label, self.right_label = QLabel(left), QLabel(right)
        for lab, value in ((self.left_label, False), (self.right_label, True)):
            lab.setCursor(Qt.CursorShape.PointingHandCursor)
            lab.mousePressEvent = lambda _e, v=value: self.setChecked(v)  # type: ignore[assignment,misc]
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.left_label)
        lay.addWidget(self.switch)
        lay.addWidget(self.right_label)
        self.switch.toggled.connect(self._on_toggled)
        self._on_toggled(False)

    def _on_toggled(self, on: bool) -> None:
        self.left_label.setStyleSheet("font-weight: bold" if not on else "color: #666")
        self.right_label.setStyleSheet("font-weight: bold" if on else "color: #666")
        self.toggled.emit(on)

    def isChecked(self) -> bool:  # noqa: N802 (Qt naming)
        """True when the right option is chosen."""
        return bool(self.switch.isChecked())

    def setChecked(self, on: bool) -> None:  # noqa: N802 (Qt naming)
        """Choose the right option (True) or the left one (False)."""
        self.switch.setChecked(on)

    def mousePressEvent(self, _event: QMouseEvent) -> None:  # noqa: N802 (Qt override)
        """Flip the choice on a click anywhere on the widget."""
        self.setChecked(not self.isChecked())
