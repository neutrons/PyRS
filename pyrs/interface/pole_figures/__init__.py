"""
Qt desktop GUI, launched locally or on the analysis cluster like PyRS.

Qt is used through qtpy, so PyQt5, PyQt6 or PySide6 works, whichever the environment has. This package module
imports nothing, so the Qt-free helpers (``run_checks``, ``mount_matrix``, ``models``) import without Qt.

Typical workflow:

1. Start it on its own with :func:`app.main` (``python -m pyrs.interface.pole_figures``), or open it from a
   running Qt application, such as the PyRS main window, with :func:`app.open_window`.
2. Work through the tabs: sample, experiment setup, prediction, table scan, data and, when installed, the ODF.

Example:
    >>> from pyrs.interface.pole_figures.app import main
    >>> main()  # doctest: +SKIP
"""
