"""
Keep the mouse wheel from changing spin boxes and combo boxes while the user scrolls a panel.

Qt sends a wheel event to the widget under the mouse. When the panel scrolls, a spin box or combo
box slides under the mouse and its value starts to change. The guard makes these inputs take the
wheel only after the user clicks into them (or reaches them with Tab). Otherwise the event goes on
to the parent, so the scroll area keeps scrolling.

Install one guard per top-level window, after its widgets are built. Widgets added later are prepared when Qt
polishes them.

Example:
    >>> guard = WheelGuard(main_window)  # doctest: +SKIP
"""

from qtpy.QtCore import QEvent, QObject, Qt  # type: ignore[attr-defined]
from qtpy.QtWidgets import QAbstractSpinBox, QApplication, QComboBox, QWidget

GUARDED = (QAbstractSpinBox, QComboBox)


class WheelGuard(QObject):
    """
    Application event filter, limited to the widgets inside ``window``.

    Args:
        window: The top-level window to guard. Widgets in other windows (for example PyRS's own) are not touched.
    """

    def __init__(self, window: QWidget) -> None:
        super().__init__(window)
        self._window = window
        for widget in window.findChildren(QWidget):
            self._prepare(widget)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)  # removed by Qt when the guard is deleted with the window

    def _mine(self, obj: QObject) -> bool:
        return isinstance(obj, GUARDED) and self._window.isAncestorOf(obj)

    @staticmethod
    def _prepare(widget: QObject) -> None:
        # StrongFocus: focus from a click or Tab, not from the wheel (WheelFocus is the spin box default)
        if isinstance(widget, GUARDED):
            widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802 (Qt name)
        """
        Pass a wheel event on to the parents when a guarded input of this window has no focus.

        Args:
            obj: The object the event is for.
            event: The event.

        Returns:
            True when the event was filtered (a wheel event on an unfocused guarded input), else False.
        """
        kind = event.type()
        if kind == QEvent.Type.Polish and self._mine(obj):
            self._prepare(obj)  # widgets added after the window was built
        elif kind == QEvent.Type.Wheel and self._mine(obj) and not obj.hasFocus():  # type: ignore[attr-defined]
            # Ignored and filtered: Qt then offers the event to the parents, so the scroll area scrolls
            event.ignore()
            return True
        return False
