try:
    from PyQt5.QtWidgets import QMainWindow, QFileDialog
except ImportError:
    from PyQt4.QtGui import QMainWindow, QFileDialog
import ui.ui_sscalvizwindow
from pyrs.utilities import checkdatatypes
import pyrs.core.pyrscore
import os
import gui_helper
import numpy
import platform
import ui.ui_calibrationwindow
import dialogs
import datetime

# setup of constants
SLICE_VIEW_RESOLUTION = 0.0001


class InstrumentCalibrationWindow(QMainWindow):
    """
    GUI window to calculate strain and stress with simple visualization
    """
    def __init__(self, parent, pyrs_core):
        """
        initialization
        :param parent:
        :param pyrs_core:
        """
        super(InstrumentCalibrationWindow, self).__init__(parent)

        # check
        assert isinstance(pyrs_core, pyrs.core.pyrscore.PyRsCore), 'PyRS core {0} of type {1} must be a PyRsCore ' \
                                                                   'instance.'.format(pyrs_core, type(pyrs_core))

        self._core = pyrs_core

        # class variables for calculation (not GUI)
        self._default_dir = None

        # set up UI
        self.ui = ui.ui_calibrationwindow.Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.pushButton_loadRawData.clicked.connect(self.do_load_raw)
        self.ui.pushButton_loadMask.clicked.connect(self.do_load_mask)
        self.ui.pushButton_reduce.clicked.connect(self.do_reduce_data)
       
        self.ui.pushButton_calibrateGeometry.clicked.connect(self.do_calibrate_geometry)
        # TODO - NIGHT - Define link to methods
        self.ui.pushButton_decreaseCenterX.clicked.connect(self.do_decrease_value_center_x)
        self.ui.pushButton_decreaseCenterY.clicked.connect(self.do_decrease_value_center_y)
        self.ui.pushButton_decreaseCenterZ.clicked.connect(self.do_decrease_value_center_z)

        self.ui.pushButton_increaseCenterX.clicked.connect(self.do_increase_value_center_x)
        self.ui.pushButton_increaseCenterY.clicked.connect(self.do_increase_value_center_y)
        self.ui.pushButton_increaseCenterZ.clicked.connect(self.do_increase_value_center_z)

        self.ui.pushButton_decreaseRotationX.clicked.connect(self.do_decrease_value_rotation_x)
        self.ui.pushButton_decreaseRotationY.clicked.connect(self.do_decrease_value_rotation_y)
        self.ui.pushButton_decreaseRotationZ.clicked.connect(self.do_decrease_value_rotation_z)

        self.ui.pushButton_increaseRotationX.clicked.connect(self.do_increase_value_rotation_x)
        self.ui.pushButton_increaseRotationY.clicked.connect(self.do_increase_value_rotation_y)
        self.ui.pushButton_increaseRotationZ.clicked.connect(self.do_increase_value_rotation_z)

        self.ui.pushButton_decreaseWavelength.clicked.connect(self.do_increase_value_wavelength)
        self.ui.pushButton_increaseWavelength.clicked.connect(self.do_decrease_value_wavelength)


        # UI widgets

    def do_load_raw(self):

        ipts_number = gui_helper.parse_line_edit()
        run_number = gui_helper.parse_line_edit()

        if ipts_number is None or run_number is None:
            # load data file directory
            raw_file_name = gui_helper.get_open()
        else:
            raw_file_name = self._controller.archive_manager.get_nexus(ipts_number, run_number)

        # load
        self._controller.reduction_manager.load_data(raw_file_name)

        return

    def do_load_mask(self):
        """
        """
        curr_mask_file = gui_helper.parse_line_edit(self.ui.lineEdit_maskFile, str, False, 'Masking file')

        if curr_mask_file is None:
            default_dir = self._controller.working_dir
        else:
            default_dir = os.path.basename(curr_mask_file)

        curr_mask_file = gui_helper.get_open()


        #
        mask_id_names = self._controller.mask_manager.load_calibration_mask(curr_mask_file)

        self.ui.comboBox_masks.clear()
        for mask_name in mask_id_names:
            self.ui.comboBox_masks.addItem(mask_name)

        return

    def do_reduce_data)

        lineEdit_rotationX
        lineEdit_centerX





        pushButton_loadGeomCalFile







    def decrease_value_center_x(self):
        """
        decrease center X value according to current one and resolution
        :return:
        """
        curr_value = gui_helper.parse_line_edit(self.ui.lineEdit_centerX, float, throw_if_blank=False)
        if curr_value is None:
            curr_value = 0.
            self.ui.lineEdit_centerX.setText('{}'.format(curr_value))

        resolution = gui_helper.parse_line_edit(self.ui.lineEdit_resolutionCenterX, float,
                                                throw_if_blank=False,
                                                edit_name='Resolution center X')
        if resolution is None:
            curr_value = 0.001
            self.ui.lineEdit_resolutionCenterX.setText('{}'.format(curr_value))

        next_value = curr_value + resolution

        self.ui.lineEdit_centerX.setText('{}'.format(next_value))

        return

    def refine_instrument_geometry(self):
        """
        Refine instrument geometry
        :return:
        """
        def set_refine_entry(line_edit, check_box, name):
            init_value = gui_helper.parse_line_edit(line_edit, float, True, name)
            is_to_refine = check_box.isChecked()
            return init_value, is_to_refine

        param_refine_flags = dict()

        param_refine_flags['center_x'] = set_refine_entry(self.ui.lineEdit_centerX,
                                                          self.ui.checkBox_refineCenterX,
                                                          'center x')
        param_refine_flags['center_y'] = set_refine_entry(self.ui.lineEdit_centerY,
                                                          self.ui.checkBox_refineCenterY,
                                                          'center Y')
        param_refine_flags['center_z'] = set_refine_entry(self.ui.lineEdit_centerY,
                                                          self.ui.checkBox_refineCenterZ,
                                                          'center Z')
        param_refine_flags['rotation_x'] = set_refine_entry(self.ui.lineEdit_rotationX,
                                                            self.ui.checkBox_refineRotationX,
                                                            'rotation X')
        param_refine_flags['rotation_y'] = set_refine_entry(self.ui.lineEdit_rotationX,
                                                            self.ui.checkBox_refineRotationX,
                                                            'rotation Y')
        param_refine_flags['rotation_z'] = set_refine_entry(self.ui.lineEdit_rotationX,
                                                            self.ui.checkBox_refineRotationX,
                                                            'rotation Z')
        param_refine_flags['wavelength'] = set_refine_entry(self.ui.lineEdit_wavelength,
                                                            self.ui.checkBox_refineWavelength,
                                                            'Wave length')

        # refine
        self._controller.calibration_engine.calibrate_instrument(param_refine_flags)

        return

    def _promote_widgets(self):

        # frame_detector2DView
        # frame_multiplePlotsView
        # frame_reducedDataView, comboBox_unit

        # TODO - NIGHT - Implement UI to promote widgets - NIGHT

        # TODO - NIGHT - In UI, better name shall be given to widgets - NIGHT


        return

    def decrease_value(self):

        sender = self.sender()

        print ('[DB...BAT] Sender:  {}'.format(self.sender()))
        print ('[DB...BAT] Methods: \n'.format(dir(sender)))


