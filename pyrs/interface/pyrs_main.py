try:
    from PyQt5.QtWidgets import QMainWindow
except ImportError:
    from PyQt4.QtGui import QMainWindow
from ui import ui_pyrsmain as ui_pyrsmain
from pyrs.core import pyrscore
import fitpeakswindow
import textureanalysiswindow
import strainstresscalwindow
import manualreductionwindow


class PyRSLauncher(QMainWindow):
    """
    The main window launched for PyRS
    """
    def __init__(self):
        """
        initialization
        """
        super(PyRSLauncher, self).__init__(None)

        # core
        self._reduction_core = pyrscore.PyRsCore()

        # set up main UI widgets
        self.ui = ui_pyrsmain.Ui_MainWindow()
        self.ui.setupUi(self)

        # define
        self.ui.pushButton_manualReduction.clicked.connect(self.do_reduce_manually)
        self.ui.pushButton_fitPeaks.clicked.connect(self.do_launch_fit_peak_window)
        self.ui.pushButton_launchTextureAnalysis.clicked.connect(self.do_launch_texture_window)
        self.ui.pushButton_launchStrainStressCalculation.clicked.connect(self.do_launch_strain_stress_window)

        self.ui.actionQuit.triggered.connect(self.do_quit)

        # child windows
        self.peak_fit_window = None
        self.texture_analysis_window = None
        self.strain_stress_window = None
        self.manual_reduction_window = None
        self.instrument_calibration_window = None

        return
    
    @property
    def core(self):
        """
        offer the access of the reduction core
        :return:
        """
        return self._reduction_core

    def do_launch_fit_peak_window(self):
        """
        launch peak fit window
        :return:
        """
        if self.peak_fit_window is None:
            self.peak_fit_window = fitpeakswindow.FitPeaksWindow(self)
            self.peak_fit_window.setup_window(self._reduction_core)
        self.peak_fit_window.show()

        # # optionally close the main window
        # if self.ui.checkBox_keepWindowOpen.isChecked() is False:
        #     self.hide()

        return

    def do_launch_strain_stress_window(self):
        """
        launch the strain/stress calculation and visualization window
        :return:
        """
        if self.strain_stress_window is None:
            self.strain_stress_window = strainstresscalwindow.StrainStressCalculationWindow(self, self._reduction_core)

        # launch
        self.strain_stress_window.show()

        # optionally close the main window
        # if self.ui.checkBox_keepWindowOpen.isChecked() is False:
        #     self.close()

        return

    def do_launch_texture_window(self):
        """
        launch texture analysis home
        :return:
        """
        if self.texture_analysis_window is None:
            self.texture_analysis_window = textureanalysiswindow.TextureAnalysisWindow(self)
            self.texture_analysis_window.setup_window(self._reduction_core)

        # show
        self.texture_analysis_window.show()

        # optionally close the main window
        # if self.ui.checkBox_keepWindowOpen.isChecked() is False:
        #     self.close()

        return

    def do_reduce_manually(self):
        """
        launch manual data reduction window
        :return:
        """
        if self.manual_reduction_window is None:
            self.manual_reduction_window = manualreductionwindow.ManualReductionWindow(self)
            self.manual_reduction_window.setup_window(self._reduction_core)

        # show
        self.manual_reduction_window.show()

        # # optionally close the main window
        # if self.ui.checkBox_keepWindowOpen.isChecked() is False:
        #     self.close()

        return

    def do_quit(self):
        """
        close window
        :return:
        """
        # close all 5 child windows
        if self.peak_fit_window is not None:
            self.peak_fit_window.close()

        if self.texture_analysis_window is not None:
            self.texture_analysis_window.close()

        if self.strain_stress_window is not None:
            self.strain_stress_window.close()

        if self.manual_reduction_window is not None:
            self.manual_reduction_window.close()

        if self.instrument_calibration_window is not None:
            self.instrument_calibration_window.close()

        self.close()

        return
