"""
The optional ODF tab (MTEX for Python): the main window adds it after Data and Corrections when this module exists.

It is kept apart from ``main_window`` so a copy without the ODF modules (PyRS before its ODF stage) still opens with
the other tabs. The tab takes the pole figures from the Data and Corrections presenter.

Example:
    >>> tab = build_odf_tab(window)  # doctest: +SKIP
    >>> window.tabs.addTab(tab, ODF_TAB_NAME)  # doctest: +SKIP
"""

from typing import TYPE_CHECKING

from .models.mtex import MtexModel
from .presenters.mtex import MtexPresenter
from .tabs.mtex_tab import MtexTab

if TYPE_CHECKING:  # main_window imports this module (when it is installed), so only the type is needed here
    from .main_window import MainWindow

ODF_TAB_NAME = "MTEX"


def build_odf_tab(window: "MainWindow") -> MtexTab:
    """
    Build the MTEX tab and its presenter, fed by the window's data, sample and material presenters.

    The presenter is a child of the tab (``tab.findChild(MtexPresenter)``). Its status messages go to the window's
    status bar.

    Args:
        window: The main window, with its other presenters built.

    Returns:
        The tab, for ``window.tabs.addTab``.
    """
    tab = MtexTab()
    presenter = MtexPresenter(tab, window.data, sample=window.sample, material=window.material, model=MtexModel())
    presenter.status.connect(window.show_status)
    return tab
