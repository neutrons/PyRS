import sys
from pyrs.core import pyrscore
from qtpy.QtWidgets import QApplication  # type:ignore  # noqa: E402
from pyrs.interface.detector_calibration.detector_calibration_viewer import DetectorCalibrationViewer  # noqa: E402
from pyrs.interface.detector_calibration.detector_calibration_model import DetectorCalibrationModel  # noqa: E402
from pyrs.interface.detector_calibration.detector_calibration_crtl import DetectorCalibrationCrtl  # noqa: E402


class App(QApplication):
    def __init__(self, sys_argv):
        super().__init__(sys_argv)
        self.model = DetectorCalibrationModel(pyrscore.PyRsCore())
        self.ctrl = DetectorCalibrationCrtl(self.model)
        self.view = DetectorCalibrationViewer(self.model, self.ctrl)
        self.view.show()


if __name__ == "__main__":
    app = App(sys.argv)
    sys.exit(app.exec_())
