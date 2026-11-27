"""
Main window: one tab per workflow step.

The window builds each tab and its presenter, wires the presenters together (sample and material feed the
prediction, the prediction feeds the table scan and the data) and shows their status messages. The ODF tab (MTEX for
Python) is added after the data tab when its module, ``odf_tab``, is installed. A copy without it (PyRS until its ODF
stage) has the other five tabs.

Example:
    >>> window = MainWindow(guide="standalone")  # doctest: +SKIP
    >>> window.show()  # doctest: +SKIP
"""

import importlib
import importlib.util

from qtpy.QtCore import Qt  # type: ignore[attr-defined]
from qtpy.QtWidgets import QAction, QMainWindow, QMessageBox, QTabWidget, QToolButton, QWidget

from pyrs import __version__
from .help import START, TAB_SECTIONS, GuideResult, GuideVariant, check_variant, guide_pdf, open_guide
from .models.data import DataModel
from .models.material import MaterialModel
from .models.predict import PredictModel
from .models.sample import SampleModel
from .models.table_scan import TableScanModel
from .presenters.data import DataPresenter
from .presenters.material import MaterialPresenter
from .presenters.predict import PredictPresenter
from .presenters.sample import SamplePresenter
from .presenters.table_scan import TableScanPresenter
from .tabs.data_tab import DataTab
from .tabs.material_tab import MaterialTab
from .tabs.predict_tab import PredictTab
from .tabs.sample_tab import SampleTab
from .tabs.table_scan_tab import TableScanTab
from .widgets.wheel_guard import WheelGuard

#: the workflow tabs every copy has, in order
TAB_NAMES = ("Sample", "Experiment setup", "Predict", "Table scan", "Data and Corrections")
#: the optional ODF tab: a module with ``ODF_TAB_NAME`` and ``build_odf_tab(window)``. It needs MTEX for Python
#: (pymtex), so it is installed only with the ODF stage.
ODF_TAB_MODULE = f"{__package__}.odf_tab"
#: how long a status message stays (ms)
STATUS_MS = 8000


def odf_tab_installed() -> bool:
    """True when this copy has the ODF tab module (``ODF_TAB_MODULE``)."""
    return importlib.util.find_spec(ODF_TAB_MODULE) is not None


class MainWindow(QMainWindow):
    """
    Top-level window with the workflow tabs (each tab shows its own plots).

    Args:
        parent: Owner window, or None for a window without an owner (default: None).
        guide: User guide the ? button and F1 open: "standalone" or "pyrs" (default: "standalone").

    Attributes:
        guide: User guide variant.
        tabs: The tab widget.
        sample: Sample tab presenter.
        material: Experiment setup tab presenter.
        predict: Predict tab presenter.
        table_scan: Table scan tab presenter.
        data: Data and Corrections tab presenter.
        help_button: The ? button at the right end of the tab bar.
    """

    def __init__(self, parent: QWidget | None = None, *, guide: GuideVariant = "standalone") -> None:
        super().__init__(parent)
        self.guide = check_variant(guide)
        self.setWindowTitle(f"HIDRA Path-Length-Corrected Pole Figures {__version__}")
        self.resize(1500, 850)

        self.sample_tab = SampleTab()
        self.sample = SamplePresenter(self.sample_tab, model=SampleModel())
        self.material_tab = MaterialTab()
        self.material = MaterialPresenter(self.material_tab, model=MaterialModel())
        self.predict_tab = PredictTab()
        self.predict = PredictPresenter(self.predict_tab, self.sample, material=self.material, model=PredictModel())
        self.table_scan_tab = TableScanTab()
        self.table_scan = TableScanPresenter(self.table_scan_tab, self.predict, model=TableScanModel())
        self.data_tab = DataTab()
        self.data = DataPresenter(self.data_tab, self.sample, material=self.material, predict=self.predict,
                                  model=DataModel())  # fmt: skip
        for presenter in (self.sample, self.material, self.predict, self.table_scan, self.data):
            presenter.status.connect(self.show_status)

        self.tabs = QTabWidget()
        built = (self.sample_tab, self.material_tab, self.predict_tab, self.table_scan_tab, self.data_tab)
        for name, tab in zip(TAB_NAMES, built, strict=True):
            self.tabs.addTab(tab, name)
        if odf_tab_installed():
            # deferred: the ODF tab is optional (it needs pymtex, and copies before the ODF stage do not have it)
            odf_tab = importlib.import_module(ODF_TAB_MODULE)
            self.tabs.addTab(odf_tab.build_odf_tab(self), odf_tab.ODF_TAB_NAME)
        self.setCentralWidget(self.tabs)
        self.help_button = QToolButton()
        self.help_button.setText("?")
        self.help_button.setAutoRaise(True)
        self.help_button.setStyleSheet("QToolButton { font-weight: bold; font-size: 14px; padding: 0 8px; }")
        self.help_button.setToolTip("Open the user guide at this step (F1)")
        self.help_button.clicked.connect(lambda _checked=False: self.show_help())
        self.tabs.setCornerWidget(self.help_button, Qt.Corner.TopRightCorner)

        self._build_menus()
        # the wheel scrolls the panels and changes a spin box or combo box only after a click into it
        self.wheel_guard = WheelGuard(self)
        self.statusBar().showMessage("Ready")

    def show_status(self, message: str) -> None:
        """Show a presenter's message in the status bar for STATUS_MS."""
        self.statusBar().showMessage(message, STATUS_MS)

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        open_action = QAction("Open &STL...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_stl)
        file_menu.addAction(open_action)
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        help_menu = self.menuBar().addMenu("&Help")
        step_action = QAction("User guide for this &step", self)
        step_action.setShortcut("F1")
        step_action.triggered.connect(lambda _checked=False: self.show_help())
        help_menu.addAction(step_action)
        guide_action = QAction("User &guide (start)", self)
        guide_action.triggered.connect(lambda _checked=False: self.show_guide(START))
        help_menu.addAction(guide_action)
        help_menu.addSeparator()
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def current_section(self) -> str:
        """Guide section of the current tab (START for a tab the guide has no section for)."""
        return TAB_SECTIONS.get(self.tabs.tabText(self.tabs.currentIndex()), START)

    def show_help(self) -> None:
        """Open the user guide at the current tab's section."""
        self.show_guide(self.current_section())

    def show_guide(self, section: str) -> None:
        """
        Open the user guide at ``section``. Tell the user when the PDF is missing or nothing can open it.

        Args:
            section: Section key (``help.START`` for the start).
        """
        result = open_guide(section, variant=self.guide)
        if result == GuideResult.MISSING:
            QMessageBox.information(self, "User guide", "The user guide PDF is not installed with this copy.")
        elif result == GuideResult.NO_VIEWER:
            box = QMessageBox(QMessageBox.Icon.Information, "User guide",
                              "No web browser or PDF viewer was found to open the user guide. "
                              f"Open it by hand:\n\n{guide_pdf(self.guide)}", parent=self)  # fmt: skip
            box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            box.exec_()

    def _open_stl(self) -> None:
        self.tabs.setCurrentWidget(self.sample_tab)
        self.sample_tab.open_button.click()

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About",
            f"HIDRA path-length-corrected pole figures {__version__}\nHB-2B, High Flux Isotope Reactor, ORNL",
        )
