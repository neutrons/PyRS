import pyrs.interface.pyrs_main
try:
    from PyQt5.QtWidgets import QMainWindow, QFileDialog, QVBoxLayout
except ImportError:
    from PyQt4.QtGui import QMainWindow, QFileDialog, QVBoxLayout
from pyrs.utilities import checkdatatypes
import pyrs.core.pyrscore
import os
import gui_helper
import ui.ui_calibrationwindow
from ui import diffdataviews

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
        self._curr_data_id = None

        self._number_rois = 0

        # set up UI
        self.ui = ui.ui_calibrationwindow.Ui_MainWindow()
        self.ui.setupUi(self)
        self._promote_widgets()

        self.ui.pushButton_loadRawData.clicked.connect(self.do_load_raw)
        # TODO - TONIGHT2 - self.ui.pushButton_loadGeomCalFile : shall parse value and set up lineEdits

        self.ui.pushButton_loadMask.clicked.connect(self.do_load_mask)
        self.ui.pushButton_reduce.clicked.connect(self.do_reduce_data)
       
        self.ui.pushButton_calibrateGeometry.clicked.connect(self.do_calibrate_geometry)

        # decrease button associated line edits dictionary
        self._decrease_button_dict = dict()
        self._decrease_button_dict[self.ui.pushButton_decreaseCenterX] = (self.ui.lineEdit_centerX,
                                                                          self.ui.lineEdit_resolutionCenterX)
        self._decrease_button_dict[self.ui.pushButton_decreaseCenterY] = (self.ui.lineEdit_centerY,
                                                                          self.ui.lineEdit_resolutionCenterY)
        self._decrease_button_dict[self.ui.pushButton_decreaseCenterZ] = (self.ui.lineEdit_centerZ,
                                                                          self.ui.lineEdit_resolutionCenterZ)
        self._decrease_button_dict[self.ui.pushButton_decreaseRotationX] = (self.ui.lineEdit_rotationX,
                                                                            self.ui.lineEdit_resolutionRotationX)
        self._decrease_button_dict[self.ui.pushButton_decreaseRotationY] = (self.ui.lineEdit_rotationY,
                                                                            self.ui.lineEdit_resolutionRotationY)
        self._decrease_button_dict[self.ui.pushButton_decreaseRotationZ] = (self.ui.lineEdit_rotationZ,
                                                                            self.ui.lineEdit_resolutionRotationZ)
        self._decrease_button_dict[self.ui.pushButton_decreaseWavelength] = (self.ui.lineEdit_wavelength,
                                                                             self.ui.lineEdit_resolutionWavelength)

        # increase button associated line edits dictionary
        self._increase_button_dict = dict()
        self._increase_button_dict[self.ui.pushButton_increaseCenterX] = (self.ui.lineEdit_centerX,
                                                                          self.ui.lineEdit_resolutionCenterX)
        self._increase_button_dict[self.ui.pushButton_increaseCenterY] = (self.ui.lineEdit_centerY,
                                                                          self.ui.lineEdit_resolutionCenterY)
        self._increase_button_dict[self.ui.pushButton_increaseCenterZ] = (self.ui.lineEdit_centerZ,
                                                                          self.ui.lineEdit_resolutionCenterZ)
        self._increase_button_dict[self.ui.pushButton_increaseRotationX] = (self.ui.lineEdit_rotationX,
                                                                            self.ui.lineEdit_resolutionRotationX)
        self._increase_button_dict[self.ui.pushButton_increaseRotationY] = (self.ui.lineEdit_rotationY,
                                                                            self.ui.lineEdit_resolutionRotationY)
        self._increase_button_dict[self.ui.pushButton_increaseRotationZ] = (self.ui.lineEdit_rotationZ,
                                                                            self.ui.lineEdit_resolutionRotationZ)
        self._increase_button_dict[self.ui.pushButton_increaseWavelength] = (self.ui.lineEdit_wavelength,
                                                                             self.ui.lineEdit_resolutionWavelength)

        # define event handing methods
        self.ui.pushButton_decreaseCenterX.clicked.connect(self.do_decrease_value)
        self.ui.pushButton_decreaseCenterY.clicked.connect(self.do_decrease_value)
        self.ui.pushButton_decreaseCenterZ.clicked.connect(self.do_decrease_value)

        self.ui.pushButton_increaseCenterX.clicked.connect(self.do_increase_value)
        self.ui.pushButton_increaseCenterY.clicked.connect(self.do_increase_value)
        self.ui.pushButton_increaseCenterZ.clicked.connect(self.do_increase_value)

        self.ui.pushButton_decreaseRotationX.clicked.connect(self.do_decrease_value)
        self.ui.pushButton_decreaseRotationY.clicked.connect(self.do_decrease_value)
        self.ui.pushButton_decreaseRotationZ.clicked.connect(self.do_decrease_value)

        self.ui.pushButton_increaseRotationX.clicked.connect(self.do_increase_value)
        self.ui.pushButton_increaseRotationY.clicked.connect(self.do_increase_value)
        self.ui.pushButton_increaseRotationZ.clicked.connect(self.do_increase_value)

        self.ui.pushButton_decreaseWavelength.clicked.connect(self.do_decrease_value)
        self.ui.pushButton_increaseWavelength.clicked.connect(self.do_increase_value)

        # Figure's subplot vs mask
        self._subplot_mask_dict = dict()
        self._mask_subplot_dict = dict()

        return

    def _promote_widgets(self):
        """ Define promoted widgets on the pre-defined QFrame instance
        :return:
        """
        # detector view
        temp_layout = QVBoxLayout()
        self.ui.frame_detector2DView.setLayout(temp_layout)
        self.ui.graphicsView_detectorView = diffdataviews.DetectorView(self)
        temp_layout.addWidget(self.ui.graphicsView_detectorView)

        # calibration view
        temp_layout = QVBoxLayout()
        self.ui.frame_multiplePlotsView.setLayout(temp_layout)
        self.ui.graphicsView_calibration = diffdataviews.GeomCalibrationView(self)
        temp_layout.addWidget(self.ui.graphicsView_calibration)

        # reduced view
        temp_layout = QVBoxLayout()
        self.ui.frame_reducedDataView.setLayout(temp_layout)
        self.ui.graphicsView_reducedDataView = diffdataviews.GeneralDiffDataView(self)
        temp_layout.addWidget(self.ui.graphicsView_reducedDataView)

        return

    def do_decrease_value(self):
        """
        Decrease the value in the associated QLineEdit from QPushButton event taking account of the
        associated resolution value in QLineEdit
        :return:
        """
        # get sender of the event
        sender = self.sender()

        if sender not in self._decrease_button_dict:
            raise RuntimeError('Sender of decrease value message (registered as {}) is not in _decrease_button_dict'
                               ''.format(sender))
        else:
            value_edit, resolution_edit = self._decrease_button_dict[sender]

        # get current value
        try:
            curr_value = gui_helper.parse_line_edit(value_edit, float, throw_if_blank=False)
        except RuntimeError as run_err:
            gui_helper.pop_message(self, '{}'.format(run_err))
            curr_value = None
        # default
        if curr_value is None:
            curr_value = 0.
            value_edit.setText('{}'.format(curr_value))

        # resolution
        try:
            resolution = gui_helper.parse_line_edit(resolution_edit, float,
                                                    throw_if_blank=False,
                                                    edit_name='Resolution')
        except RuntimeError as run_err:
            gui_helper.pop_message(self, '{}'.format(run_err))
            resolution = None

        if resolution is None:
            resolution = 0.001
            resolution_edit.setText('{}'.format(resolution))

        # get next value
        next_value = curr_value - resolution
        value_edit.setText('{}'.format(next_value))

        # reduction?
        if self.ui.checkBox_reducedRealtime.isChecked():
            self.do_reduce_data()

        return

    def do_increase_value(self):
        """
        Decrease the value in the associated QLineEdit from QPushButton event taking account of the
        associated resolution value in QLineEdit
        :return:
        """
        # get sender of the event
        sender = self.sender()

        if sender not in self._increase_button_dict:
            raise RuntimeError('Sender of decrease value message (registered as {}) is not in _decrease_button_dict'
                               ''.format(sender))
        else:
            value_edit, resolution_edit = self._increase_button_dict[sender]

        # get current value
        try:
            curr_value = gui_helper.parse_line_edit(value_edit, float, throw_if_blank=False)
        except RuntimeError as run_err:
            gui_helper.pop_message(self, '{}'.format(run_err))
            curr_value = None
        # default
        if curr_value is None:
            curr_value = 0.
            value_edit.setText('{}'.format(curr_value))
        # resolution
        try:
            resolution = gui_helper.parse_line_edit(resolution_edit, float,
                                                    throw_if_blank=False,
                                                    edit_name='Resolution')
        except RuntimeError as run_err:
            gui_helper.pop_message(self, '{}'.format(run_err))
            resolution = None

        if resolution is None:
            resolution = 0.001
            resolution_edit.setText('{}'.format(resolution))

        # get next value
        next_value = curr_value + resolution
        value_edit.setText('{}'.format(next_value))

        # reduction?
        if self.ui.checkBox_reducedRealtime.isChecked():
            self.do_reduce_data()

        return

    def do_calibrate_geometry(self):
        # TODO - TONIGHT 2 - Implement

        return

    def do_load_raw(self):
        """
        Load raw data (TIFF, NeXus HDF5, SPICE .bin)
        :return:
        """
        # try: IPTS and run (regular way)
        ipts_number = gui_helper.parse_line_edit(self.ui.lineEdit_iptsNumber, int, False, 'IPTS  number', None)
        run_number = gui_helper.parse_line_edit(self.ui.lineEdit_runNumber, int, False, 'Run number', None)

        if ipts_number is None or run_number is None:
            # load data file directory
            raw_file_name = QFileDialog.getOpenFileName(self, 'Get experiment data', os.getcwd())
        else:
            # from archive
            raw_file_name = self.core.archive_manager.get_nexus(ipts_number, run_number)

        # load
        self.load_data_file(raw_file_name)

        return

    def load_data_file(self, file_name):
        """
        Load data fle
        :param file_name:
        :return:
        """
        # load
        try:
            self._curr_data_id = self._core.reduction_engine.load_data(file_name)
        except RuntimeError as run_err:
            gui_helper.pop_message(self, message_type='error', message='Unable to load data {}'.format(file_name),
                                   detailed_message='{}'.format(run_err))

        return

    def do_load_mask(self):
        """ Load mask file
        """
        # get the current mask file
        curr_mask_file = gui_helper.parse_line_edit(self.ui.lineEdit_maskFile, str, False, 'Masking file')

        # whether it has been loaded
        if curr_mask_file in self._core.reduction_engine.get_loaded_mask_files():
            gui_helper.pop_message(self, message='Mask {} has been loaded', message_type='info',
                                   detailed_message='If need to load a new mask, clear the file name in editor')
            return

        # get mask
        if not os.path.exists(curr_mask_file):
            # need to load
            if curr_mask_file == '':
                default_dir = os.getcwd()
            else:
                default_dir = os.path.basename(curr_mask_file)

            curr_mask_file = QFileDialog.getOpenFileName(self, 'Maks file name', default_dir, 'All Files(*.*)')
            if isinstance(curr_mask_file, tuple):
                raise NotImplementedError('Case of tuple of getOpenFileName')
            if curr_mask_file == '':
                return
            else:
                # set file names
                self.ui.lineEdit_maskFile.setText('{}'.format(curr_mask_file))
                self.ui.plainTextEdit_maskList.appendPlainText('{}\n'.format(curr_mask_file))
        # END-IF

        # load mask
        try:
            two_theta, note, mask_id = self._core.reduction_engine.load_mask_file(curr_mask_file)
        except RuntimeError as run_err:
            gui_helper.pop_message(self, message='Unable to load {}'.format(curr_mask_file),
                                   message_type='error',
                                   detailed_message='{}'.format(run_err))
            return

        # update UI
        self._number_rois += 1
        self.ui.graphicsView_calibration.set_number_rois(self._number_rois)
        self._subplot_mask_dict[self._number_rois - 1] = mask_id
        self._mask_subplot_dict[mask_id] = self._number_rois - 1

        return

    def do_reduce_data(self):
        """ reduce data
        :return:
        """
        try:
            cal_shift_x = gui_helper.parse_line_edit(self.ui.lineEdit_centerX, float, False, 'Center X', default=0.)
            cal_shift_y = gui_helper.parse_line_edit(self.ui.lineEdit_centerY, float, False, 'Center Y', default=0.)
            cal_shift_z = gui_helper.parse_line_edit(self.ui.lineEdit_centerZ, float, False, 'Center Z', default=0.)
            cal_rot_x = gui_helper.parse_line_edit(self.ui.lineEdit_rotationX, float, False, 'Rotation X', default=0.)
            cal_rot_y = gui_helper.parse_line_edit(self.ui.lineEdit_rotationY, float, False, 'Rotation Y', default=0.)
            cal_rot_z = gui_helper.parse_line_edit(self.ui.lineEdit_rotationZ, float, False, 'Rotation Z', default=0.)
            cal_wave_length = gui_helper.parse_line_edit(self.ui.lineEdit_wavelength, float, False, 'Rotation Z',
                                                         default=0.)
        except RuntimeError as run_err:
            gui_helper.pop_message(self, 'Unable to parse calibration value', str(run_err), 'error')
            return

        # get data file
        try:
            two_theta = gui_helper.parse_line_edit(self.ui.lineEdit_2theta, float, True, 'Two theta', default=None)
        except RuntimeError as run_err:
            gui_helper.pop_message(self, '2-theta error', str(run_err), 'error')
            return

        # load instrument
        self._core.reduction_manager.load_instrument(two_theta, cal_shift_x, cal_shift_y, cal_shift_z,
                                                     cal_rot_x, cal_rot_y, cal_rot_z,
                                                     cal_wave_length)

        # reduce masks
        for mask_id in self._core.reduction_engine.get_mask_ids():
            mask_vec = self._core.reduction_engine.get_mask_vector(mask_id)
            self._core.reduction_engine.reduce_to_2theta(data_id=self._curr_data_id,
                                                         output_name=None,
                                                         use_mantid_engine=False,
                                                         mask_vec=mask_vec,
                                                         two_theta=two_theta)
            vec_x, vec_y = self._core.reduction_engine.get_reduced_data()
            self.ui.graphicsView_calibration.plot_data(vec_x, vec_y, self._mask_subplot_dict[mask_id])

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
