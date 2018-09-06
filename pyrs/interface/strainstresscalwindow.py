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
import ui.ui_sscalvizwindow
import dialogs


class StrainStressCalculationWindow(QMainWindow):
    """
    GUI window to calculate strain and stress with simple visualization
    """
    def __init__(self, parent):
        """
        initialization
        :param parent:
        """
        super(StrainStressCalculationWindow, self).__init__(parent)

        # class variables
        self._core = None

        # child dialogs and windows
        self._d0_grid_dialog = None
        self._strain_stress_table_view = None
        self._grid_alignment_table_view = None
        self._new_session_dialog = None

        # set up UI
        self.ui = ui.ui_sscalvizwindow.Ui_MainWindow()
        self.ui.setupUi(self)
        self._init_widgets()

        # set up event handling
        self.ui.pushButton_browseReducedFile.clicked.connect(self.do_browse_reduced_file)
        self.ui.pushButton_browse_e11ScanFile.clicked.connect(self.do_browse_e11_file)
        self.ui.pushButton_browse_e22ScanFile.clicked.connect(self.do_browse_e22_file)
        self.ui.pushButton_browse_e33ScanFile.clicked.connect(self.do_browse_e33_file)

        self.ui.pushButton_loadFile.clicked.connect(self.do_load_strain_files)
        self.ui.pushButton_preAlignSampleLogXYZ.clicked.connect(self.do_get_grid_alignment_info)
        self.ui.pushButton_calUnconstrainedStress.clicked.connect(self.do_calculate_strain_stress)
        self.ui.pushButton_launchSSTable.clicked.connect(self.do_show_strain_stress_table)

        self.ui.pushButton_setd0Grid.clicked.connect(self.do_set_d0_by_grid)
        self.ui.pushButton_showAlignGridTable.clicked.connect(self.do_show_aligned_grid)
        self.ui.pushButton_alignGrids.clicked.connect(self.do_align_grids)

        # strain/stress save and export
        self.ui.pushButton_saveStressStrain.clicked.connect(self.save_stress_strain)
        self.ui.pushButton_exportSpecialType.clicked.connect(self.do_export_stress_strain)

        # radio buttons changed case
        self.ui.radioButton_loadRaw.toggled.connect(self.evt_load_file_type)
        self.ui.radioButton_loadReduced.toggled.connect(self.evt_load_file_type)

        self.ui.radioButton_uniformD0.toggled.connect(self.evt_change_d0_type)
        self.ui.radioButton_d0Grid.toggled.connect(self.evt_change_d0_type)

        # combo boxes handling
        self.ui.comboBox_plotParameterName.currentIndexChanged.connect(self.do_plot_sliced_3d)

        # menu
        self.ui.actionNew_Session.triggered.connect(self.do_new_session)
        self.ui.actionQuit.triggered.connect(self.do_quit)
        # TODO - 20180814 - actionPlot_Grids_3D

        # self.lineEdit_tdScanFile..connect(self.)
        # self.lineEdit_ndScanFile..connect(self.)
        # self.lineEdit_rdScanFile..connect(self.)
        #
        # self.lineEdit_reducedFile..connect(self.)
        #
        # self.lineEdit_outputFileName..connect(self.)
        # self.lineEdit_exportFileName..connect(self.)
        # self.plainTextEdit_info..connect(self.)
        self.ui.graphicsView_sliceView
        # self.lineEdit_sliceStartValue..connect(self.)
        # self.lineEdit_sliceEndValue..connect(self.)
        self.ui.horizontalSlider_slicer3D.valueChanged.connect(self.do_slice_3d_data)

        # current data/states
        self._core = None
        self._curr_data_key = None
        self._session_name = None

        # 3D data to slice
        self._slice_view_grid_vec = None
        self._slice_view_param_vec = None

        # mutex
        self._load_file_radio_mutex = False
        self._d0_type_mutex = False

        return

    def _init_widgets(self):
        """
        initialize widgets
        :return:
        """
        self.ui.radioButton_loadRaw.setChecked(True)
        self.ui.groupBox_importRawFiles.setEnabled(True)
        self.ui.radioButton_loadReduced.setChecked(False)
        self.ui.groupBox_importReducedFile.setEnabled(False)

        # set up label with Greek
        self.ui.label_poisson.setText(u'\u03BD (Poisson\' Ratio)')

        # combo boxes
        self.ui.comboBox_alignmentCriteria.clear()
        self.ui.comboBox_alignmentCriteria.addItem('E11')
        self.ui.comboBox_alignmentCriteria.addItem('E22')
        self.ui.comboBox_alignmentCriteria.addItem('E33')
        self.ui.comboBox_alignmentCriteria.addItem('User specified Grid')
        self.ui.comboBox_alignmentCriteria.addItem('Finest Grid (Auto)')

        return

    @property
    def core(self):
        """
        provide the reference to controller/core
        :return:
        """
        return self._core

    def do_get_grid_alignment_info(self):
        """
        align the loaded data for XYZ
        :return:
        """
        # get user specified sample log names
        pos_x_log_name = str(self.ui.comboBox_sampleLogNameX.currentText())
        pos_y_log_name = str(self.ui.comboBox_sampleLogNameY.currentText())
        pos_z_log_name = str(self.ui.comboBox_sampleLogNameZ.currentText())

        try:
            self._core.strain_stress_calculator.check_grids_alignment(pos_x=pos_x_log_name,
                                                                      pos_y=pos_y_log_name,
                                                                      pos_z=pos_z_log_name)
        except RuntimeError as run_err:
            print ('Measuring points are not aligned: {}'.format(run_err))
            self._core.strain_stress_calculator.align_matched_grids(resolution=0.001)

        self.ui.pushButton_alignGrids.setEnabled(True)
        # self.ui.groupBox_calculator.setEnabled(True)

        return

    def do_align_grids(self):
        """
        align grids according to user input
        :return:
        """
        # get the alignment method
        alignment_method = str(self.ui.comboBox_alignmentCriteria.currentText()).lower()

        direction = None
        user_define = False

        if alignment_method.count('auto'):
            direction = self._core.strain_stress_calculator.get_finest_direction()
        elif alignment_method.count('user'):
            user_define = True
        else:
            direction = alignment_method

        stat_dict = self._core.strain_stress_calculator.get_grids_information()

        # pop out dialog and get data
        ret_value = dialogs.get_strain_stress_grid_setup(self, user_define_grid=user_define,
                                                         grid_stat_dict=stat_dict)

        # return if user cancels the operation
        if ret_value is None:
            return

        self.align_user_grids(direction, user_define_flag=user_define,
                              grids_setup_dict=ret_value, show_aligned_grid=True)

        return

    def align_user_grids(self, direction, user_define_flag, grids_setup_dict, show_aligned_grid=True):
        """ align the current grid to
        :param direction:
        :param user_define_flag:
        :param grids_setup_dict:
        :param show_aligned_grid:
        :return:
        """
        # create user specified grids and align the experiment grids in e11/e22/e33 to user-specified grid
        grid_array, mapping_array = \
            self._core.strain_stress_calculator.align_grids(direction=direction, user_defined=user_define_flag,
                                                            grids_dimension_dict=grids_setup_dict)

        # convert (map or interpolate) peak positions from experiments to output
        center_d_vector = \
            self._core.strain_stress_calculator.align_peak_parameter_on_grids(grids_vector=grid_array,
                                                                              parameter='center_d',
                                                                              scan_log_map_vector=mapping_array)

        # show the align grids report???
        if show_aligned_grid:
            self.do_show_aligned_grid()
            self._grid_alignment_table_view.set_aligned_grids_info(grid_array, mapping_array)
            self._grid_alignment_table_view.set_peak_parameter_names(
                self._core.strain_stress_calculator.get_peak_parameter_names())
            self._grid_alignment_table_view.set_aligned_parameter_value('center_d', center_d_vector)
        # END-IF (show aligned grid)

        # allow to calculate strain and stress
        self.ui.pushButton_calUnconstrainedStress.setEnabled(True)

        return self._grid_alignment_table_view

    def do_browse_e11_file(self):
        """ browse LD raw file
        :return:
        """
        ld_file_name = gui_helper.browse_file(self, caption='Load LD (raw) File',
                                              default_dir=self._core.working_dir,
                                              file_filter='Data File (*.dat)',
                                              file_list=False,
                                              save_file=False)
        if ld_file_name is not None:
            self.ui.lineEdit_e11ScanFile.setText(ld_file_name)

        return

    def do_browse_e22_file(self):
        """ browse ND raw file
        :return:
        """
        nd_file_name = gui_helper.browse_file(self, caption='Load ND (raw) File',
                                              default_dir=self._core.working_dir,
                                              file_filter='Data File (*.dat)',
                                              file_list=False,
                                              save_file=False)
        if nd_file_name is not None:
            self.ui.lineEdit_e22ScanFile.setText(nd_file_name)

        return

    def do_browse_e33_file(self):
        """ browse TD raw file
        :return:
        """
        td_file_name = gui_helper.browse_file(self, caption='Load TD (raw) File',
                                              default_dir=self._core.working_dir,
                                              file_filter='Data File (*.dat)',
                                              file_list=False,
                                              save_file=False)
        if td_file_name is not None:
            self.ui.lineEdit_e33ScanFile.setText(td_file_name)

        return

    def do_browse_reduced_file(self):
        """ browse the previous calculated and saved strain/stress file
        :return:
        """
        reduced_file_name = gui_helper.browse_file(self, caption='Previously saved stress/strain File',
                                                   default_dir=self._core.default_dir,
                                                   file_filter='Data File (*.dat);;HDF File (*.hdf)',
                                                   file_list=False,
                                                   save_file=False)

        if reduced_file_name is not None:
            self.ui.lineEdit_reducedFile.setText(reduced_file_name)

        return

    def do_calculate_strain_stress(self):
        """
        calculate strain from loaded file
        :return:
        """
        # get values
        params = self.get_strain_parameters()
        if isinstance(params, str):
            err_msg = params
            gui_helper.pop_message(self, err_msg, message_type='error')
            return
        else:
            e_young, nu_poisson = params

        # set E, nu and d0 to calculator
        self._core.strain_stress_calculator.set_youngs_modulus(young_e=e_young)
        self._core.strain_stress_calculator.set_poisson_ratio(poisson_ratio=nu_poisson)

        # get d0
        if self.ui.radioButton_uniformD0.isChecked():
            # uniformed d0
            d0 = float(self.ui.lineEdit_d0.text())
            self._core.strain_stress_calculator.set_d0(d0=d0)
        else:
            # d0 varies across sample grids
            # TODO - 2018 - Need Jeff's data to develop the UI and etc
            grids_d0 = self._grid_d0_dialog.get_d0_grids()
            self._core.strain_stress_calculator.set_grids_d0(grids_d0)
        # END-IF

        # call the core to calculate strain
        self._core.strain_stress_calculator.execute()

        return

    def do_load_strain_files(self):
        """
        load strain/stress file from either raw files or previously saved file
        :return:
        """
        # current session is not canceled: ask user whether it is OK to delete and start a new one
        if self._curr_data_key is not None:
            continue_load = gui_helper.get_user_permit(caption='Current session shall be closed '
                                                               'before new session is started.', )
            if continue_load is False:
                return
        # END-IF (need to save)

        if self.ui.radioButton_loadRaw.isChecked():
            # load raw files
            e11_file_name = str(self.ui.lineEdit_e11ScanFile.text())
            self.load_raw_file(e11_file_name, 'e11')
            sample_logs_e11 = self._core.strain_stress_calculator.get_sample_logs_names('e11', to_set=True)

            e22_file_name = str(self.ui.lineEdit_e22ScanFile.text())
            self.load_raw_file(e22_file_name, 'e22')
            sample_logs_e22 = self._core.strain_stress_calculator.get_sample_logs_names('e22', to_set=True)

            common_sample_logs = sample_logs_e11 & sample_logs_e22

            if self._core.strain_stress_calculator.is_unconstrained_strain_stress:
                e33_file_name = str(self.ui.lineEdit_e33ScanFile.text())
                self.load_raw_file(e33_file_name, 'e33')
                sample_logs_e33 = self._core.strain_stress_calculator.get_sample_logs_names('e33', to_set=True)

                common_sample_logs = sample_logs_e33 & common_sample_logs
        else:
            # load saved files
            # TODO - 2018 - Next - Need an example for such file!
            reduced_file_name = str(self.ui.lineEdit_reducedFile.text())
            data_key, message = self._core.load_strain_stress_file(file_name=reduced_file_name)
            raise RuntimeError('Not Implemented')
        # END-IF

        # disable calculation until the alignment is finished
        self.ui.pushButton_alignGrids.setEnabled(False)
        self.ui.pushButton_calUnconstrainedStress.setEnabled(False)

        # set up the combo box for 3 directions
        self.ui.comboBox_sampleLogNameX.clear()
        self.ui.comboBox_sampleLogNameY.clear()
        self.ui.comboBox_sampleLogNameZ.clear()

        common_sample_logs = list(common_sample_logs)
        common_sample_logs.sort()
        for log_name in common_sample_logs:
            self.ui.comboBox_sampleLogNameX.addItem(log_name)
            self.ui.comboBox_sampleLogNameY.addItem(log_name)
            self.ui.comboBox_sampleLogNameZ.addItem(log_name)
        # END-FOR

        # convert the peak centers to dspacing
        self._core.strain_stress_calculator.convert_peaks_positions()

        return

    def do_load_peak_info_files(self):
        """
        load peak information files
        :return:
        """


        return

    def do_slice_3d_data(self):
        """
        slice the already loaded 3D data
        :return:
        """
        vec_x, vec_y, vec_z = self._slice_value_on_grid(self._slider_min, self._slider_max)

        self.ui.graphicsView_sliceView.plot_contour(vec_x, vec_y, vec_z)
        self.ui.graphicsView_sliceView.plot_scatter(vec_x, vec_y)

        return

    def evt_change_d0_type(self):
        """
        in case the d0 type is toggled to uniform d0 or d0 in grid
        :return:
        """
        if self.ui.radioButton_uniformD0.isChecked():
            self.ui.lineEdit_d0.setEnabled(True)
            self.ui.pushButton_setd0Grid.setEnabled(False)
        elif self.ui.radioButton_d0Grid.isChecked():
            self.ui.lineEdit_d0.setEnabled(False)
            self.ui.pushButton_setd0Grid.setEnabled(True)

        return

    def get_strain_parameters(self):
        """
        parse Young's modulus and Poisson's ratio
        :return:
        """
        try:
            young_modulus = float(self.ui.lineEdit_youngModulus.text())
            poisson_ratio = float(self.ui.lineEdit_poissonRatio.text())
        except ValueError:
            err_msg = 'Unable to parse Young\'s modulus E {} or Poisson\'s ratio {} to float' \
                      ''.format(self.ui.lineEdit_youngModulus.text(), self.ui.lineEdit_poissonRatio.text())
            return err_msg

        return young_modulus, poisson_ratio

    @staticmethod
    def load_column_file(file_name):
        """ load a column file (most for test)
        :param file_name:
        :return: an numpy.ndarray
        """
        checkdatatypes.check_file_name(file_name, check_exist=True, is_dir=False)

        # open file
        col_file = open(file_name, 'r')
        lines = col_file.readline()
        col_file.close()

        # parse
        data_set_list = list()
        for line in lines:
            line = line.strip()
            if len(line) == 0 or line.startswith('#'):
                # empty line or comment line
                continue
            terms = line.split()

            line_values = numpy.ndarray(shape=(len(terms),), dtype='float')
            for iterm in range(len(terms)):
                line_values[iterm] = float(terms[iterm])

            data_set_list.append(line_values)
        # END-FOR

        data_set = numpy.array(data_set_list)

        return data_set

    def load_raw_file(self, file_name, direction):
        """
        load stress/strain raw file with peak fit information
        :param file_name:
        :param direction:
        :return:
        """
        try:
            self._core.strain_stress_calculator.load_reduced_file(file_name=file_name, direction=direction)
        except RuntimeError as run_error:
            gui_helper.pop_message(self, message='Unable to load reduced HB2B diffraction file {} '
                                                 'due to {}'.format(file_name, run_error))

        return

    def new_strain_stress_session(self, session_name, is_plane_strain, is_plane_stress):
        """
        create a new session
        :param session_name:
        :param is_plane_strain:
        :return:
        """
        checkdatatypes.check_string_variable('Strain/stress session name', session_name)
        checkdatatypes.check_bool_variable('Flag for being plane strain', is_plane_strain)
        checkdatatypes.check_bool_variable('Flag for being plane stress', is_plane_stress)

        self._core.new_strain_stress_session(session_name, is_plane_stress=is_plane_stress,
                                             is_plane_strain=is_plane_strain)
        # set the class variable
        self._session_name = session_name

        # disable calculation group before align the measuring data points
        self.ui.groupBox_calculator.setEnabled(False)
        # disable e33 if it is plane strain/stress
        if is_plane_strain or is_plane_stress:
            self.ui.lineEdit_e33ScanFile.setEnabled(False)
            self.ui.pushButton_browse_e33ScanFile.setEnabled(False)
        else:
            self.ui.lineEdit_e33ScanFile.setEnabled(True)
            self.ui.pushButton_browse_e33ScanFile.setEnabled(True)
        return

    # TODO - NEW - 20180828 - Develop UI for this! - TODO
    def do_plot_slice_view(self):
        """
        plot a slice view for selected parameter selected
        :return:
        """
        if self.ui.radioButton_plotPeakParam.isChecked():
            peak_param_name = str(self.ui.comboBox_plotParameterName.currentText())
            slice_dir = str(self.ui.comboBox_sliceDirection.currentText())
            slice_pos = str(self.ui.lineEdit_slicePosition.text())
            self.plot_peak_param_slice(peak_param_name, slice_dir, slice_pos)
        elif self.ui.radioButton_plotStrainStress.isChecked():
            param_index = self.ui.comboBox_plotStrainStress.currentIndex()
            if param_index == 0:
                param = 'epsilon'
            else:
                param = 'nu'
            slice_index = gui_helper.parse_tuples(self.ui.lineEdit_ssElementIndex, 2)
            self.plot_strain_stress_slice()

        return

    # TESTME - 20180905 - NEXT BIG STEP
    def plot_peak_param_slice(self, param_name, ss_direction, is_raw_grid):
        """ plot specified peak parameter's value on 3D grid in slice view
        :param param_name:
        :param ss_direction:
        :param is_raw_grid:
        :return:
        """
        # check inputs
        checkdatatypes.check_string_variable('(Peak) parameter name', param_name)
        checkdatatypes.check_string_variable('Strain/stress direction (e11/e22/e33)', ss_direction,
                                             self._core.strain_stress_calculator.get_strain_stress_direction)
        checkdatatypes.check_bool_variable('Flag for raw experiment grid', is_raw_grid)

        if is_raw_grid:
            # parameter value on raw grids
            # get value
            grid_param_dict = self._core.strain_stress_calculator.get_raw_grid_param_values(ss_direction, param_name)
            # convert to a 2D array (grid position vector) and a 1D array (parameter value)
            self._slice_view_grid_vec, self._slice_view_param_vec = self._convert_grid_dict_to_vectors(grid_param_dict)
        else:
            # mapped/interpolated parameter value on output strain/stress grid
            self._slice_view_grid_vec = self._core.strain_stress_calculator.get_strain_stress_grid()
            self._slice_view_param_vec = self._core.strain_stress_calculator.get_user_grid_param_values(ss_direction,
                                                                                                        param_name)
        # END-IF-ELSE

        # set up slier slider
        self.ui.comboBox_sliceDirection.setCurrentIndex(0)  # as X
        min_value = numpy.min(self._slice_view_grid_vec[:, 0])
        max_value = numpy.max(self._slice_view_grid_vec[:, 0])

        self.ui.lineEdit_sliceStartValue.setText('{}'.format(min_value))
        self.ui.lineEdit_sliceEndValue.setText('{}'.format(max_value))
        self.ui.lineEdit_sliceStartValue.setEnabled(False)
        self.ui.lineEdit_sliceEndValue.setEnabled(False)

        self.ui.horizontalSlider_slicer3D.setRange(0, 99)
        self.ui.horizontalSlider_slicer3D.setTickInterval(1)
        self.ui.horizontalSlider_slicer3D.setTickPosition(0)

        # slice the data
        vec_x, vec_y, vec_z = self._slice_value_on_grid(0, 99)

        # 5. plot
        self.ui.graphicsView_sliceView.plot_contour(vec_x, vec_y, vec_z, contour_resolution=1, flush=True)
        self.ui.graphicsView_sliceView.plot_scatter(vec_x, vec_y, flush=True)

        # 6. store the 3D data

        # 7. self.ui.horizontalSlider_slicer3D: event handling


        return

    def _slice_value_on_grid(self, slider_min, slider_max):
        """ slice current parameter value on grid along a selected direction (x, y or z)
        :return:
        """
        # select the direction
        grid_dir = self.ui.comboBox_sliceDirection.currentIndex()

        # select the value
        slider_value = self.ui.horizontalSlider_slicer3D.sliderPosition()
        min_value = float(self.ui.lineEdit_sliceStartValue.text())
        max_value = float(self.ui.lineEdit_sliceEndValue.text())
        slice_pos = (int(slider_value) - slider_min)/(slider_max - slider_min) * (max_value - min_value) + min_value

        # slice
        RESOLUTION = 1.
        slice_min = slice_pos - 0.5 * RESOLUTION
        slice_max = slice_pos + 0.5 * RESOLUTION

        range_index_larger = self._slice_view_grid_vec[:, grid_dir] >= slice_min
        sub_grid_vec = self._slice_view_grid_vec[range_index_larger]
        sub_value_vec = self._slice_view_param_vec[range_index_larger]
        range_index_smaller = sub_grid_vec[:, grid_dir] < slice_max

        sliced_grid_vec = sub_grid_vec[range_index_smaller]
        sliced_value_vec = sub_value_vec[range_index_smaller]

        # TODO FIXME - 20180905 -

        # remove the column sliced on
        sliced_grid_vec = numpy.delete(sliced_grid_vec, grid_dir, 1)  # a column for axis=1
        vec_x = sliced_grid_vec[:, 0]
        vec_y = sliced_grid_vec[:, 1]

        return vec_x, vec_y, sliced_value_vec

    def plot_strain_stress_slice(self, param='epsilon', index=[0, 0], dir=1, position=0.0):
        """
        plot a strain and stress slice
        :param param:
        :param index:
        :param dir:
        :param position:
        :return:
        """


        return

    def save_stress_strain(self, file_type=None):
        """
        save the calculated strain/stress file
        :return:
        """
        if file_type is None:
            file_type = str(self.ui.comboBox_saveFileType.currentText())

        raise NotImplementedError('TO BE CONTINUED')

    def create_new_session(self, session_name, is_plane_strain, is_plane_stress):
        """ create a new strain/stress calculation session
        :param session_name:
        :param is_plane_strain:
        :param is_plane_stress:
        :return:
        """
        # check input
        checkdatatypes.check_string_variable('Strain/stress calculating session name', session_name)
        checkdatatypes.check_bool_variable('Flag to be plane strain', is_plane_strain)
        checkdatatypes.check_bool_variable('Flag to be plane stress', is_plane_stress)

        self._core.new_strain_stress_session(session_name,
                                             is_plane_strain=is_plane_strain,
                                             is_plane_stress=is_plane_stress)

        self.setWindowTitle(session_name)

        return

    def do_export_stress_strain(self):
        """
        export the stress/strain to some other format for future analysis
        :return:
        """
        # TODO - 20180813 - Next Step - Implement after discussing with beamline scientist
        raise NotImplementedError('ASAP')

    def do_quit(self):
        """
        quit without saving
        :return:
        """
        self.close()

        return

    def do_set_d0_by_grid(self):
        """
        set up non-uniform d0 by given a grid
        :return:
        """
        # TODO - 20180813 - SOON - Need data from beamline scientist
        if self._d0_grid_dialog is None:
            self._d0_grid_dialog = dialogs.GridD0SetupDialog(self)
        else:
            self._d0_grid_dialog.reset()

        self._d0_grid_dialog.show()

        return

    def do_show_aligned_grid(self):
        """
        launch table to show how grids are aligned (match or not match)
        --->  show an empty table???
        :param self:
        :return:
        """
        if self._grid_alignment_table_view is None:
            self._grid_alignment_table_view = dialogs.GridAlignmentCheckTablesView(self)

        # show table
        self._grid_alignment_table_view.show()

        # try to gather some grid alignment information from loaded data
        # TODO - 20180824 - method to set up the table

        return

    def do_show_strain_stress_table(self):
        """
        show the calculated strain and stress values
        :return:
        """
        # TODO - 20180813 - Implement the table view
        if self._strain_stress_table_view is None:
            self._strain_stress_table_view = dialogs.StrainStressTableView(self)
        else:
            self._strain_stress_table_view.reset_table()

        # get value and setup
        self._strain_stress_table_view.set_strain_stress_values(self._core.get_strain_stress_values())

        self._strain_stress_table_view.show()

        return

    def evt_load_file_type(self):
        """
        triggered when the radio buttons selection for file type to load is changed.
        enable and disable file loading group accordingly
        :return:
        """
        # if mutex is on, leave the method
        if self._load_file_radio_mutex:
            return

        # set the mutex
        self._load_file_radio_mutex = True

        # enable and disable
        if self.ui.radioButton_loadRaw.isChecked():
            self.ui.groupBox_importRawFiles.setEnabled(True)
            self.ui.groupBox_importReducedFile.setEnabled(False)
        else:
            self.ui.groupBox_importReducedFile.setEnabled(True)
            self.ui.groupBox_importRawFiles.setEnabled(False)
        # END-IF-ELSE

        # release the mutex
        self._load_file_radio_mutex = False

        return

    def do_new_session(self):
        """
        create a new session
        :return:
        """
        if self._new_session_dialog is None:
            self._new_session_dialog = dialogs.CreateNewSessionDialog(self)
        else:
            self._new_session_dialog.reset_dialog()

        self._new_session_dialog.show()

        return

    def do_plot_sliced_3d(self):
        """
        slice loaded 3D stress/strain and plot
        :return:
        """
        slice_direction = str(self.ui.comboBox_sliceDirection.currentText()).lower()
        plot_term = str(self.ui.comboBox_plotParameterName.currentText())

        # TODO - 20180813 - To be continued

    def set_items_to_plot(self):
        """

        :return:
        """
        try:
            items = self._core.strain_calculator.get_plot_items(self._curr_data_key)
        except RuntimeError as run_err:
            return False, run_err

        self.ui.comboBox_plotParameterName.clear()
        for item in items:
            self.ui.comboBox_plotParameterName.addItem(item)

        return

    def setup_window(self, pyrs_core):
        """ set up the texture analysis window
        :param pyrs_core:
        :return:
        """
        # check
        assert isinstance(pyrs_core, pyrs.core.pyrscore.PyRsCore), 'PyRS core {0} of type {1} must be a PyRsCore ' \
                                                                   'instance.'.format(pyrs_core, type(pyrs_core))

        self._core = pyrs_core

        return



