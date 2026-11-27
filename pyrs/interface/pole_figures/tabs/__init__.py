"""
One QWidget per workflow step. Widgets only, the presenters hold the logic.

Each tab builds its forms (from the tab model), tables, buttons and plot panel, and offers small helpers to show
values and ask for files. The matching presenter in ``gui.presenters`` connects the signals.
"""
