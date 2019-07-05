# This module contains methods for PyQt UI files

try:
    from PyQt5 import QtCore
    from PyQt5.QtWidgets import QWidget
    from PyQt5.QtWidgets import QVBoxLayout
except ImportError:
    from PyQt4 import QtCore
    from PyQt4.QtGui import QWidget
    from PyQt4.QtGui import QVBoxLayout


def promote_widget(parent, ui_frame, widget_class):
    """
    Promote widget by
    (1) Create a Layout on existing QFrame
    (2) Create a new widget
    (3) Set the new widget with layout

    Usage: new_widget = ui_util.promote_widget(self, self.ui.XXXX_frame, Promoted_Widget)
           self.ui.whatever = new_widget

    :param parent: parent class calling UI build
    :param ui_frame: QFrame instance to add promoted widget
    :param widget_class: widget class to promote to
    :return: instance of widget
    """
    frame_layout = QVBoxLayout()
    ui_frame.setLayout(frame_layout)
    promoted_widget = widget_class(parent)
    frame_layout.addWidget(promoted_widget)

    return promoted_widget
