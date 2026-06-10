import sys

from qtpy.QtWidgets import QApplication  # type:ignore

from pyrs.interface.manual_reduction.manual_reduction_viewer import ManualReductionViewer
from pyrs.interface.manual_reduction.manual_reduction_model import ManualReductionModel
from pyrs.interface.manual_reduction.manual_reduction_crtl import ManualReductionCrtl


class App(QApplication):
    def __init__(self, sys_argv):
        super().__init__(sys_argv)
        self.model = ManualReductionModel()
        self.ctrl = ManualReductionCrtl(self.model)
        self.view = ManualReductionViewer(self.model, self.ctrl)
        self.view.show()


if __name__ == "__main__":
    app = App(sys.argv)
    sys.exit(app.exec_())
