from qtpy.QtWidgets import QMainWindow, QFileDialog
from pyrs.utilities import load_ui

# try:
#     from PyQt5.QtWidgets import QMainWindow, QFileDialog
#     from PyQt5.uic import loadUi as load_ui
# except ImportError:
#     from PyQt4.QtGui import QMainWindow, QFileDialog
#     from PyQt4.uic import loadUi as load_ui
from pyrs.interface.ui import qt_util
from pyrs.interface.ui.diffdataviews import SampleSliceView
from pyrs.utilities import checkdatatypes
import pyrs.core.pyrscore
import os
import gui_helper
import numpy
import platform
import dialogs
import datetime

# setup of constants
SLICE_VIEW_RESOLUTION = 0.0001


class StrainStressCalculationWindow(QMainWindow):
    """
    GUI window to calculate strain and stress with simple visualization
    """
    def __init__(self, parent, pyrs_core):
        """
        initialization
        :param parent:
        :param pyrs_core:
        """
        super(StrainStressCalculationWindow, self).__init__(parent)

        # check
        assert isinstance(pyrs_core, pyrs.core.pyrscore.PyRsCore), 'PyRS core {0} of type {1} must be a PyRsCore ' \
                                                                   'instance.'.format(pyrs_core, type(pyrs_core))

        self._core = pyrs_core

        # class variables for calculation (not GUI)
        self._default_dir = None

        # child dialogs and windows
        self._d0_grid_dialog = None
        self._strain_stress_table_view = None
        self._grid_alignment_table_view = dialogs.GridAlignmentCheckTablesView(self)
        self._new_session_dialog = None

        # set up UI
        ui_path = os.path.join(os.path.dirname(__file__), os.path.join('ui', 'sscalvizwindow.ui'))
        self.ui = load_ui(ui_path, baseinstance=self)

        # promote widget
        self.ui.graphicsView_sliceView = qt_util.promote_widget(self, self.ui.graphicsView_sliceView_frame,
                                                                SampleSliceView)

        # set up event handling
        self.ui.pushButton_newSession.clicked.connect(self.do_new_session)
        self.ui.pushButton_browseReducedFile.clicked.connect(self.do_browse_reduced_file)
        self.ui.pushButton_browse_e11ScanFile.clicked.connect(self.do_browse_e11_file)
        self.ui.pushButton_browse_e22ScanFile.clicked.connect(self.do_browse_e22_file)
        self.ui.pushButton_browse_e33ScanFile.clicked.connect(self.do_browse_e33_file)

        self.ui.pushButton_loadFile.clicked.connect(self.do_load_strain_files)
        self.ui.pushButton_preAlignSampleLogXYZ.clicked.connect(self.do_check_exp_grids_alignment)
        self.ui.pushButton_calUnconstrainedStress.clicked.connect(self.do_calculate_strain_stress)
        self.ui.pushButton_launchSSTable.clicked.connect(self.do_show_strain_stress_table)

        self.ui.pushButton_setd0Grid.clicked.connect(self.do_set_d0_by_grid)
        self.ui.pushButton_showAlignGridTable.clicked.connect(self.do_show_aligned_grid)
        self.ui.pushButton_alignGrids.clicked.connect(self.do_generate_slice_view_grids)

        # visualization
        self.ui.pushButton_prevSlice.clicked.connect(self.do_show_prev_slice)
        self.ui.pushButton_nextSlice.clicked.connect(self.do_show_next_slice)

        # strain/stress save and export
        self.ui.pushButton_saveStressStrain.clicked.connect(self.save_stress_strain)
        self.ui.pushButton_exportSpecialType.clicked.connect(self.do_export_stress_strain)

        # radio buttons changed case
        self.ui.radioButton_loadRaw.toggled.connect(self.evt_load_file_type)
        self.ui.radioButton_loadReduced.toggled.connect(self.evt_load_file_type)

        self.ui.radioButton_uniformD0.toggled.connect(self.evt_change_d0_type)
        self.ui.radioButton_d0Grid.toggled.connect(self.evt_change_d0_type)

        # combo boxes handling
        self.ui.comboBox_typeStrainStress.currentIndexChanged.connect(self.do_reset_strain_stress)
        self.ui.comboBox_type.currentIndexChanged.connect(self.do_change_plot_type)
        self.ui.comboBox_plotParameterName.currentIndexChanged.connect(self.do_setup_plot_data)
        self.ui.comboBox_paramDirection.currentIndexChanged.connect(self.do_setup_plot_data)
        self.ui.comboBox_sliceDirection.currentIndexChanged.connect(self.evt_change_slice_direction)
        self.ui.comboBox_sampleLogNameX_E11.currentIndexChanged.connect(self.evt_sync_grid_x_name)
        self.ui.comboBox_sampleLogNameY_E11.currentIndexChanged.connect(self.evt_sync_grid_y_name)
        self.ui.comboBox_sampleLogNameZ_E11.currentIndexChanged.connect(self.evt_sync_grid_z_name)

        # menu
        self.ui.actionNew_Session.triggered.connect(self.do_new_session)
        self.ui.actionQuit.triggered.connect(self.do_quit)

        self.ui.horizontalSlider_slicer3D.valueChanged.connect(self.do_slice_3d_plot_contour)

        # current data/states
        self._curr_data_key = None
        self._session_name = None

        # 3D data to slice and grids information
        self._slice_view_exp_grid_vec = None
        self._slice_view_exp_value_vec = None
        self._grid_linear_arrays = [None] * 3   # [0/1/2] = np.arange(min, max, step)  0 for X, 1 for Y, 2 for Z
        self._grid_viz_setup_dict = None

        # mutex
        self._load_file_radio_mutex = False
        self._d0_type_mutex = False
        self._auto_plot_mutex = False  # Flag to control the event handler to load and plot peak parameter/strain/stress
        self._sync_grid_xyz_name_mutex = False  # Flag to control the event ...
        self._strain_stress_type_mutex = False

        # create a default new session
        session_name = str(self.ui.lineEdit_sessionName.text())
        self.create_new_session(session_name=session_name, is_plane_strain=False,
                                is_plane_stress=False)
        # set up everything correct
        self.ui.comboBox_type.setCurrentIndex(0)
        self.ui.comboBox_type.setCurrentIndex(2)

        return

    def evt_sync_grid_x_name(self):
        """ Synchronize the combo boxes for sample log X-direction for all E11/E22/E33
        :return:
        """
        if self._sync_grid_xyz_name_mutex:
            return

        # Note: using this algorithm because it is easier for coding
        sample_log_index = self.ui.comboBox_sampleLogNameX_E11.currentIndex()
        for box in [self.ui.comboBox_sampleLogNameX_E22, self.ui.comboBox_sampleLogNameX_E33]:
            box.setCurrentIndex(sample_log_index)

        return

    def evt_sync_grid_y_name(self):
        """ Synchronize the combo boxes for sample log Y-direction for all E11/E22/E33
        :return:
        """
        if self._sync_grid_xyz_name_mutex:
            return

        # Note: using this algorithm because it is easier for coding
        sample_log_index = self.ui.comboBox_sampleLogNameY_E11.currentIndex()
        for box in [self.ui.comboBox_sampleLogNameY_E22, self.ui.comboBox_sampleLogNameY_E33]:
            box.setCurrentIndex(sample_log_index)

        return

    def evt_sync_grid_z_name(self):
        """ Synchronize the combo boxes for sample log Z-direction for all E11/E22/E33
        :return:
        """
        if self._sync_grid_xyz_name_mutex:
            return

        # Note: using this algorithm because it is easier for coding
        sample_log_index = self.ui.comboBox_sampleLogNameZ_E11.currentIndex()
        for box in [self.ui.comboBox_sampleLogNameZ_E22, self.ui.comboBox_sampleLogNameZ_E33]:
            box.setCurrentIndex(sample_log_index)

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
        self.ui.comboBox_typeStrainStress.clear()
        self.ui.comboBox_typeStrainStress.addItem('Unconstrained Strain/Stress')
        self.ui.comboBox_typeStrainStress.addItem('Plane Strain')
        self.ui.comboBox_typeStrainStress.addItem('Plane Stress')

        # radio buttons
        self.ui.radioButton_uniformD0.setChecked(True)

        # push button
        self.ui.pushButton_loadFile.setEnabled(False)

        # line edits
        self.ui.lineEdit_sessionName.setText('new strain/stress session')

        # Groups of widgets to be enabled or disabled
        self.ui.groupBox_importData.setEnabled(False)
        self.ui.groupBox_alignGrids.setEnabled(False)
        self.ui.groupBox_calculator.setEnabled(False)

        return

    @property
    def core(self):
        """
        provide the reference to controller/core
        :return:
        """
        return self._core

    def do_change_plot_type(self, do_plot=True):
        """
        set the parameters to change according to plot type
        :return:
        """
        # set up the no-plot flag
        self._auto_plot_mutex = True

        # check whether this is necessary to clean the other boxes
        if self.ui.comboBox_paramDirection.count() == 0:
            # no peak parameters to plot: strain/stress
            is_strain_stress = True
        else:
            # peak parameters mode
            is_strain_stress = False

        # what is the new plot type?
        curr_plot_type = str(self.ui.comboBox_type.currentText())
        if curr_plot_type.startswith('St') and is_strain_stress is False:
            # change type from peak parameters to strain/stress
            self.ui.comboBox_plotParameterName.clear()
            self.ui.comboBox_paramDirection.clear()
            # strain/stress matrix elements
            for item_name in ['', '(1, 1)', '(2, 2)', '(3, 3)']:
                self.ui.comboBox_plotParameterName.addItem(item_name)
            self.ui.comboBox_plotParameterName.setCurrentIndex(0)

        elif curr_plot_type.startswith('P') and is_strain_stress:
            # change type from strain/stress to (peak) parameter
            self.ui.comboBox_plotParameterName.clear()
            self.ui.comboBox_plotParameterName.addItem('')   # add an empty string to plot nothing
            self.ui.comboBox_plotParameterName.addItem('center_d')
            self.ui.comboBox_paramDirection.addItem('')
            for dir_i in self._core.strain_stress_calculator.get_strain_stress_direction():
                self.ui.comboBox_paramDirection.addItem(dir_i)
        # END-IF-ELSE

        # release the flag/lock
        self._auto_plot_mutex = False

        return

    def do_check_exp_grids_alignment(self):
        """
        align the grids among all the loaded data for different direction
        :return:
        """
        pos_x_sample_name_dict = dict()
        pos_y_sample_name_dict = dict()
        pos_z_sample_name_dict = dict()

        # get user specified sample log names
        # X:
        pos_x_sample_name_dict['e11'] = str(self.ui.comboBox_sampleLogNameX_E11.currentText())
        pos_x_sample_name_dict['e22'] = str(self.ui.comboBox_sampleLogNameX_E22.currentText())
        if 'e33' in self._core.strain_stress_calculator.get_strain_stress_direction():
            pos_x_sample_name_dict['e33'] = str(self.ui.comboBox_sampleLogNameX_E33.currentText())

        # Y:
        pos_y_sample_name_dict['e11'] = str(self.ui.comboBox_sampleLogNameY_E11.currentText())
        pos_y_sample_name_dict['e22'] = str(self.ui.comboBox_sampleLogNameY_E22.currentText())
        if 'e33' in self._core.strain_stress_calculator.get_strain_stress_direction():
            pos_y_sample_name_dict['e33'] = str(self.ui.comboBox_sampleLogNameY_E33.currentText())

        # Z:
        pos_z_sample_name_dict['e11'] = str(self.ui.comboBox_sampleLogNameZ_E11.currentText())
        pos_z_sample_name_dict['e22'] = str(self.ui.comboBox_sampleLogNameZ_E22.currentText())
        if 'e33' in self._core.strain_stress_calculator.get_strain_stress_direction():
            pos_z_sample_name_dict['e33'] = str(self.ui.comboBox_sampleLogNameZ_E33.currentText())

        # Check the grid position sample log from each input scan log in each direction
        self._core.strain_stress_calculator.set_grid_log_names(pos_x_sample_names=pos_x_sample_name_dict,
                                                               pos_y_sample_names=pos_y_sample_name_dict,
                                                               pos_z_sample_names=pos_z_sample_name_dict)

        # check sample grids alignment and find out the matched grids among all directions
        try:
            status, error_message = self._core.strain_stress_calculator.check_grids_alignment()
        except RuntimeError as run_err:
            gui_helper.pop_message(self, message=str(run_err), message_type='error')
            return
        self.ui.plainTextEdit_info.insertPlainText(error_message)

        # locate aligned/matched grids
        self._core.strain_stress_calculator.located_matched_grids(resolution=0.001)

        self.ui.pushButton_alignGrids.setEnabled(True)

        stat_dict = self._core.strain_stress_calculator.get_grids_information()
        self._grid_alignment_table_view.set_grid_statistics_table(stat_dict)

        # end of process: enable strain/stress calculation button and alignment result button
        self.ui.pushButton_showAlignGridTable.setEnabled(True)
        # allow to calculate strain and stress
        self.ui.groupBox_calculator.setEnabled(True)
        self.ui.pushButton_launchSSTable.setEnabled(False)
        self.ui.pushButton_calUnconstrainedStress.setEnabled(True)
        self.ui.pushButton_calUnconstrainedStress.setStyleSheet("background-color: white")

        self.show_message(str(self._core.strain_stress_calculator))

        return

    def do_generate_slice_view_grids(self):
        """ Generate output grids for slice view on strain/stress/peak parameters
        align grids according to user input for slicing view of the 3D strain/stress in sample
        :return:
        """
        # get the grids information in a dictionary
        stat_dict = self._core.strain_stress_calculator.get_grids_information()

        # pop out dialog and get data
        grid_dim_setup = dialogs.get_strain_stress_grid_setup(self, user_define_grid=True,
                                                              grid_stat_dict=stat_dict,
                                                              grid_setup_dict=self._grid_viz_setup_dict)

        # return if user cancels the operation
        if grid_dim_setup is None:
            return

        # generate a series of vectors on X, Y and Z direction.  The slider will be set accordingly
        self._grid_viz_setup_dict = grid_dim_setup

        self._grid_linear_arrays = self._core.strain_stress_calculator.generate_grids(
            grids_dimension_dict=self._grid_viz_setup_dict)

        # do not plot anything
        self.ui.comboBox_plotParameterName.setCurrentIndex(0)

        # set up the sliders
        self._setup_slice_slider()

        return

    def _setup_slice_slider(self):
        """
        set up the range of the slice slider
        :return:
        """
        if self._grid_linear_arrays is None:
            print ('[WARNING] Slice view grids has not been set up yet')
            return False

        slice_direction_index = self.ui.comboBox_sliceDirection.currentIndex()

        self.ui.label_sliderMin.setText('{:.3f}'.format(self._grid_linear_arrays[slice_direction_index][0]))
        self.ui.label_sliderMax.setText('{:.3f}'.format(self._grid_linear_arrays[slice_direction_index][-1]))

        self.ui.horizontalSlider_slicer3D.setMinimum(0)
        self.ui.horizontalSlider_slicer3D.setMaximum(self._grid_linear_arrays[slice_direction_index].shape[0]-1)
        # self.ui.horizontalSlider_slicer3D.setRange(0, self._steps_number)
        self.ui.horizontalSlider_slicer3D.setTickInterval(1)
        self.ui.horizontalSlider_slicer3D.setTickPosition(0)  # set to the left end

        return True

    def _get_slider_value(self):
        """ get the value from the slider considering the arbitrary min and max
        :return:
        """
        slice_direction_index = self.ui.comboBox_sliceDirection.currentIndex()

        slider_pos = self.ui.horizontalSlider_slicer3D.value()
        try:
            slider_value = self._grid_linear_arrays[slice_direction_index][slider_pos]
        except TypeError as run_err:
            raise RuntimeError('Unable to get slider value {} from Grid array {} due to {}'
                               '.'.format(slider_pos, self._grid_linear_arrays, run_err))
        self.ui.label_sliderValue.setText('{:.3f}'.format(slider_value))

        return slider_value

    def plot_peak_parameter(self):
        """

        :return:
        """
        # TODO - 20181010 - Refactor!
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
            self._grid_alignment_table_view.set_aligned_parameter_value(grid_array, center_d_vector)
            self._grid_alignment_table_view.show_tab(1)
        # END-IF (show aligned grid)

        return self._grid_alignment_table_view

    def do_browse_e11_file(self):
        """ browse LD raw file
        :return:
        """
        if self._default_dir is None:
            self._default_dir = self._core.working_dir

        ld_file_name = gui_helper.browse_file(self, caption='Load LD (raw) File',
                                              default_dir=self._default_dir,
                                              file_filter='HDF File (*.hdf5);;Data File (*.dat)',
                                              file_list=False,
                                              save_file=False)
        if ld_file_name is not None:
            self.ui.lineEdit_e11ScanFile.setText(ld_file_name)
            self._default_dir = os.path.dirname(str(ld_file_name))

        return

    def do_browse_e22_file(self):
        """ browse ND raw file
        :return:
        """
        if self._default_dir is None:
            self._default_dir = self._core.working_dir

        nd_file_name = gui_helper.browse_file(self, caption='Load ND (raw) File',
                                              default_dir=self._default_dir,
                                              file_filter='HDF File (*.hdf5);;Data File (*.dat)',
                                              file_list=False,
                                              save_file=False)
        if nd_file_name is not None:
            self.ui.lineEdit_e22ScanFile.setText(nd_file_name)
            self._default_dir = str(os.path.dirname(nd_file_name))

        return

    def do_browse_e33_file(self):
        """ browse TD raw file
        :return:
        """
        if self._default_dir is None:
            self._default_dir = self._core.working_dir

        td_file_name = gui_helper.browse_file(self, caption='Load TD (raw) File',
                                              default_dir=self._default_dir,
                                              file_filter='HDF File (*.hdf5);;Data File (*.dat)',
                                              file_list=False,
                                              save_file=False)
        if td_file_name is not None:
            self.ui.lineEdit_e33ScanFile.setText(td_file_name)
            self._default_dir = str(os.path.dirname(td_file_name))

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
        try:
            self._core.strain_stress_calculator.execute()
        except RuntimeError as run_err:
            gui_helper.pop_message(self, 'Unable to calculate strain/stress',
                                   detailed_message='{}'.format(run_err),
                                   message_type='error')
            return

        # set message
        self.ui.plainTextEdit_info.insertPlainText('[{}] Strain/stress calculation is finished\n'
                                                   .format(datetime.datetime.now()))

        self.ui.pushButton_calUnconstrainedStress.setStyleSheet("background-color: green")

        # enable the next
        self.ui.pushButton_launchSSTable.setEnabled(True)

        return

    def do_load_strain_files(self):
        """
        load strain/stress file from either raw files or previously saved file
        :return:
        """
        print ('[DB...BAT] RAW files are to be loaded.')

        # current session is not canceled: ask user whether it is OK to delete and start a new one
        if self._curr_data_key is not None:
            continue_load = gui_helper.get_user_permit(caption='Current session shall be closed '
                                                               'before new session is started.', )
            if continue_load is False:
                return
        # END-IF (need to save)

        load_success = True
        if self.ui.radioButton_loadRaw.isChecked():
            # load raw files
            # check raw file name
            e11_file_name = str(self.ui.lineEdit_e11ScanFile.text()).strip()
            e22_file_name = str(self.ui.lineEdit_e22ScanFile.text()).strip()
            if self._core.strain_stress_calculator.is_unconstrained_strain_stress:
                e33_file_name = str(self.ui.lineEdit_e33ScanFile.text()).strip()
            else:
                e33_file_name = None
            # check whether they are different
            for dir_ss, file_name in [('E11', e11_file_name), ('E22', e22_file_name),
                                      ('E33', e33_file_name)]:
                if file_name == '':
                    gui_helper.pop_message(self, '{} file is not given'.format(dir_ss), message_type='error')
                    return
            # END-FOR
            if e11_file_name == e22_file_name or e22_file_name == e33_file_name or e11_file_name == e33_file_name:
                gui_helper.pop_message(self, 'E11/E22/E33 have duplicates.',
                                       detailed_message='E11/E22/E33 file names: {}, {} and {}'
                                                        ''.format(e11_file_name, e22_file_name, e33_file_name),
                                       message_type='error')
                return

            # load raw file
            load_success = self.load_raw_file(e11_file_name, 'e11') and load_success
            load_success = self.load_raw_file(e22_file_name, 'e22') and load_success
            if e33_file_name is not None:
                load_success = self.load_raw_file(e33_file_name, 'e33') and load_success

        else:
            # load saved files
            # TODO - 2018 - Next - Need an example for such file!
            reduced_file_name = str(self.ui.lineEdit_reducedFile.text())
            data_key, message = self._core.load_strain_stress_file(file_name=reduced_file_name)
            raise RuntimeError('Not Implemented')
        # END-IF

        if not load_success:
            gui_helper.pop_message(self, message='Not all files are loaded correctly', message_type='info')
            return

        # convert the peak centers to d-spacing
        self._core.strain_stress_calculator.convert_peaks_positions()

        # set up the peak parameter list in the right-panel
        # TODO - 20180906 - Need to support other peak parameters
        self._auto_plot_mutex = True
        self.ui.comboBox_plotParameterName.addItem('center_d')
        self.ui.comboBox_paramDirection.clear()
        for dir_i in self._core.strain_stress_calculator.get_strain_stress_direction():
            self.ui.comboBox_paramDirection.addItem(dir_i)
        self._auto_plot_mutex = False

        # enable alignment group
        self.ui.groupBox_alignGrids.setEnabled(True)
        self.ui.groupBox_calculator.setEnabled(False)
        # enable alignment options: force to check grid alignment first
        self.ui.pushButton_preAlignSampleLogXYZ.setEnabled(True)
        self.ui.pushButton_showAlignGridTable.setEnabled(False)
        # disable load
        self.ui.groupBox_importData.setEnabled(False)

        # show information
        info_str = '[{}]Reduced files for strain/stress calculation session {} are loaded:\n' \
                   ''.format(datetime.datetime.now(), self._session_name)
        for text_edit in [self.ui.lineEdit_e11ScanFile, self.ui.lineEdit_e22ScanFile,
                          self.ui.lineEdit_e33ScanFile]:
            info_str += '{}\n'.format(text_edit.text())
            # set text color
            text_edit.setStyleSheet("color: rgb(0, 255, 0);")
        info_str = '{}\n{}'.format(str(self._core.strain_stress_calculator), info_str)

        self.show_message(info_str)

        return

    def do_load_peak_info_files(self):
        """
        load peak information files
        :return:
        """
        raise RuntimeError('It is not decided about the peak information file format')

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

    def do_slice_3d_plot_contour(self):
        """
        slice the already loaded 3D data
        :return:
        """
        try:
            slider_value = self._get_slider_value()
        except RuntimeError as run_err:
            gui_helper.pop_message(self, message='Check Grid Setup', detailed_message=str(run_err),
                                   message_type='error')
            return
        slider_direction_index = int(str(self.ui.comboBox_sliceDirection.currentIndex()))  # 0, 1, 2
        self.ui.label_sliderValue.setText('{:.3f}'.format(slider_value))
        vec_x, vec_y, matrix_z, info, dir_tuple = self._slice_values_on_grid(slider_value, slider_direction_index)

        print ('INFO: {}'.format(info))
        self.ui.graphicsView_sliceView.plot_contour_raw(vec_x, vec_y, matrix_z, info, dir_tuple)

        return

    def create_new_session(self, session_name, is_plane_strain, is_plane_stress):
        """
        create a new session
        :param session_name:
        :param is_plane_strain:
        :param is_plane_stress:
        :return:
        """
        # get new session's name
        if isinstance(session_name, unicode) or session_name.__class__.__name__.count('QString') == 1:
            session_name = str(session_name).strip()
        else:
            checkdatatypes.check_string_variable('Strain/stress session name', session_name)

        print ('[DB...BAT] New Session {}... Plane Strain {}... Plane Stress {}'
               ''.format(session_name, is_plane_strain, is_plane_stress))

        # check with current session
        if self._session_name is not None and session_name == self._session_name:
            gui_helper.pop_message(self, 'Invalid name for a new session',
                                   detailed_message='User specified new session name is same as previous '
                                                    'session name {}'.format(session_name),
                                   message_type='error')
            return
        elif self._session_name is not None:
            # TODO - FIXME - Auto/Semi-auto/Manual save current session
            print ('[WARNING] Previous session {} is NOT saved'.format(self._session_name))

        # check types
        checkdatatypes.check_bool_variable('Flag for being plane strain', is_plane_strain)
        checkdatatypes.check_bool_variable('Flag for being plane stress', is_plane_stress)

        self._core.new_strain_stress_session(session_name, is_plane_stress=is_plane_stress,
                                             is_plane_strain=is_plane_strain)

        # set the class variable and so is the line edits
        self._session_name = session_name
        self.ui.lineEdit_sessionName.setText(self._session_name)

        # set information
        if is_plane_stress:
            index = 2
        elif is_plane_strain:
            index = 1
        else:
            index = 0
        if self.ui.comboBox_typeStrainStress.currentIndex() != index:
            self._strain_stress_type_mutex = True
            self.ui.comboBox_typeStrainStress.setCurrentIndex(index)
            self._strain_stress_type_mutex = False

        # disable calculation group before align the measuring data points
        self.ui.groupBox_importData.setEnabled(True)
        # disable e33 if it is plane strain/stress
        if is_plane_strain or is_plane_stress:
            self.ui.lineEdit_e33ScanFile.setEnabled(False)
            self.ui.pushButton_browse_e33ScanFile.setEnabled(False)
        else:
            self.ui.lineEdit_e33ScanFile.setEnabled(True)
            self.ui.pushButton_browse_e33ScanFile.setEnabled(True)
        self.ui.groupBox_calculator.setEnabled(False)

        # reset the line edits
        for edit in [self.ui.lineEdit_e11ScanFile, self.ui.lineEdit_e22ScanFile, self.ui.lineEdit_e33ScanFile]:
            edit.setStyleSheet("color: black;")
            edit.setText('')

        # enable load button
        self.ui.groupBox_importRawFiles.setEnabled(True)
        self.ui.pushButton_loadFile.setEnabled(True)

        # disable calculation until the alignment is finished
        self.ui.pushButton_alignGrids.setEnabled(False)
        self.ui.groupBox_calculator.setEnabled(False)
        self.ui.pushButton_calUnconstrainedStress.setEnabled(False)
        self.ui.pushButton_showAlignGridTable.setEnabled(False)

        # disable loading if not
        self.show_message(str(self._core.strain_stress_calculator))

        # set title
        self.setWindowTitle(session_name)

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

    def evt_change_slice_direction(self):
        """ As slicing direction is changed, something need to be changed
        :return:
        """
        can_set = self._setup_slice_slider()

        if can_set:
            self.do_slice_3d_plot_contour()

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
        :return: boolean to indicate whether loading is successful or not
        """
        try:
            self._core.strain_stress_calculator.load_reduced_file(file_name=file_name, direction=direction)
        except RuntimeError as run_error:
            gui_helper.pop_message(self, message='Unable to load reduced HB2B diffraction file {} '
                                                 'due to {}'.format(file_name, run_error))
            return False

        # set up the combo box for 3 directions
        sample_logs_list = self._core.strain_stress_calculator.get_sample_logs_names(direction, to_set=False)

        self._setup_sample_logs_combo_box(sample_logs_list, direction)

        return True

    def do_setup_plot_data(self):
        """ Set the data for slice view/plot
        :return:
        """
        # return if it is required not to plot
        if self._auto_plot_mutex:
            return

        # check whether there is anything to plot
        if str(self.ui.comboBox_plotParameterName.currentText()).strip() == '':
            return

        plot_type_str = str(self.ui.comboBox_type.currentText())
        # grid_coordinate_dir_index = self.ui.comboBox_sliceDirection.currentIndex()   # X, Y, Z
        if plot_type_str.startswith('St'):
            # strain or stress
            if True:  # a cheat but fast implementation
                strain_stress_index_str = str(self.ui.comboBox_plotParameterName.currentText()).strip()
                if strain_stress_index_str == 0:
                    return
                strain_stress_index = gui_helper.parse_tuples(strain_stress_index_str, int, 2)

            else:
                matrix_index_str = str(self.ui.comboBox_plotParameterName.currentText())
                strain_stress_index = gui_helper.parse_tuples(matrix_index_str, int, 2)
            if plot_type_str == 'Strain':
                is_strain = True
            else:
                is_strain = False  # then it is stress
            # plot
            self._set_strain_stress_to_plot(is_strain, strain_stress_index)

        else:
            # peak parameters
            # TODO - 20181011 - Implement (from cleaning old codes)
            # TODO - ASAP - 20181011 - Continue from here!
            peak_param_name = str(self.ui.comboBox_plotParameterName.currentText())
            ss_dir = str(self.ui.comboBox_paramDirection.currentText())
            self._set_peak_parameter_to_plot(param_name=peak_param_name, ss_direction=ss_dir,
                                             is_raw_grid=False)
        # END-IF-ELSE

        self._auto_plot_mutex = False

        # plot
        self.do_slice_3d_plot_contour()

        return

    @staticmethod
    def _convert_grid_dict_to_vectors(ss_direction, grid_param_value_dict):
        """ Convert the grid parameters to vectors
        :param grid_param_value_dict:
        :return:
        """
        grid_list = list()
        value_list = list()
        for grid_pos in grid_param_value_dict:
            grid_list.append(grid_pos)
            value_list.append(grid_param_value_dict[grid_pos]['value'])
            assert grid_param_value_dict[grid_pos]['dir'] == ss_direction, 'Direction not matching'

        grid_vec = numpy.array(grid_list)
        print grid_vec.shape
        value_vec = numpy.array(value_list)
        print value_vec.shape

        return grid_vec, value_vec

    def _setup_sample_logs_combo_box(self, sample_logs_list, measurement_direction):
        """ set up the sample log combo box
        :param sample_logs_list:
        :param measurement_direction:
        :return:
        """
        def add_values(box_list, sample_log_list, func_is_allowed):
            """ add values to the boxes
            :param box_list:
            :param sample_log_list:
            :param func_is_allowed:
            :return:
            """
            for box in box_list:
                box.clear()
            for log_name_i in sorted(sample_log_list):
                if func_is_allowed(log_name_i) is False:
                    continue
                for box in box_list:
                    box.addItem(log_name_i)

            return
        # END-DEF-add_values()s

        print ('[DB...BAT] Direction {}. Sample log list: {}'.format(measurement_direction, sample_logs_list))

        # check inputs
        checkdatatypes.check_list('Sample log names', sample_logs_list)
        checkdatatypes.check_string_variable('Measurement direction', measurement_direction,
                                             allowed_values=['e11', 'e22', 'e33'])

        # set up box list
        if measurement_direction == 'e11':
            sample_box_list = [self.ui.comboBox_sampleLogNameX_E11,
                               self.ui.comboBox_sampleLogNameY_E11,
                               self.ui.comboBox_sampleLogNameZ_E11]
        elif measurement_direction == 'e22':
            sample_box_list = [self.ui.comboBox_sampleLogNameX_E22,
                               self.ui.comboBox_sampleLogNameY_E22,
                               self.ui.comboBox_sampleLogNameZ_E22]
        elif measurement_direction == 'e33':
            sample_box_list = [self.ui.comboBox_sampleLogNameX_E33,
                               self.ui.comboBox_sampleLogNameY_E33,
                               self.ui.comboBox_sampleLogNameZ_E33]
        else:
            raise RuntimeError('Not possible')

        # set up the box
        self._sync_grid_xyz_name_mutex = True
        add_values(sample_box_list, sample_logs_list,
                   self._core.strain_stress_calculator.is_allowed_grid_position_sample_log)
        self._sync_grid_xyz_name_mutex = False

        return

    def _slice_values_on_grid(self, slice_value, slice_direction):
        """ Slice current parameter value on grid along a selected direction (x, y or z)
        Algorithm: slice 3D and then do 3D-interpolation
        :param param_vector: vector of the parameter value to project on the sliced 3D grids
        :return:
        """
        checkdatatypes.check_int_variable('Slice direction index', slice_direction, (0, 3))

        info = ('[DB....BAT] Generate grid at the plane normal to Axis = {} @ value = {}'
                ''.format(slice_direction, slice_value))
        print (info)

        # construct 3D values
        dir_index_set = sorted(list(set(range(3)) - {slice_direction}))
        vec_row_tick = self._grid_linear_arrays[dir_index_set[0]]
        vec_col_tick = self._grid_linear_arrays[dir_index_set[1]]
        matrix = numpy.ndarray(shape=(vec_row_tick.shape[0], vec_col_tick.shape[0], 3), dtype='float')

        dir_x = dir_index_set[0]
        dir_y = dir_index_set[1]

        for i_col in range(vec_row_tick.shape[0]):
            matrix[i_col, :, dir_x] = vec_row_tick[i_col]
            matrix[i_col, :, slice_direction] = slice_value
            matrix[i_col, :, dir_y] = vec_col_tick

        value_matrix_2d = numpy.ndarray(shape=(vec_row_tick.shape[0], vec_col_tick.shape[0]), dtype='float')

        grid_pos_vec = numpy.ndarray(shape=(vec_row_tick.shape[0] * vec_col_tick.shape[0], 3), dtype='float')
        index = 0
        for i_col in range(vec_row_tick.shape[0]):
            for i_row in range(vec_col_tick.shape[0]):
                grid_pos = matrix[i_col][i_row]
                grid_pos_vec[index] = grid_pos
                index += 1
                # self._core.strain_stress_calculator.interpolate3d(self._slice_view_exp_grid_vec,
                #                                                   self._slice_view_exp_value_vec,
                #                                                   grid_pos)
        # END-FOR

        value_vec = self._core.strain_stress_calculator.interpolate3d(self._slice_view_exp_grid_vec,
                                                          self._slice_view_exp_value_vec,
                                                          grid_pos_vec)
        # print (value_vec)
        # print (value_vec.shape)

        index = 0
        for i_col in range(vec_row_tick.shape[0]):
            for i_row in range(vec_col_tick.shape[0]):
                value_matrix_2d[i_col, i_row] = value_vec[index]
                index += 1
        # END-FOR

        dir_str_dict = {0: 'X', 1: 'Y', 2: 'Z'}
        dir_str = dir_str_dict[slice_direction]
        info = self._slice_view_label + ': @ {} = {}'.format(dir_str, slice_value)
        dir_tuple = dir_str_dict[dir_x], dir_str_dict[dir_y]

        return vec_row_tick, vec_col_tick, value_matrix_2d, info, dir_tuple

    def _set_peak_parameter_to_plot(self, param_name, ss_direction, is_raw_grid):
        """ plot specified peak parameter's value on 3D grid in slice view
        :param param_name:
        :param ss_direction:
        :param is_raw_grid:
        :return:
        """
        # check inputs
        checkdatatypes.check_string_variable('(Peak) parameter name', param_name)
        try:
            checkdatatypes.check_string_variable('Strain/stress direction (e11/e22/e33)', ss_direction,
                                                  self._core.strain_stress_calculator.get_strain_stress_direction())
        except ValueError as value_error:
            gui_helper.pop_message(self, message='Unable to plot @ {}'.format(ss_direction),
                                   detailed_message='{}'.format(value_error),
                                   message_type='error')
            return
        checkdatatypes.check_bool_variable('Flag for raw experiment grid', is_raw_grid)

        print ('[DB...ASAP] Try to plot {} @ direction {}'.format(param_name, ss_direction))

        # return if nothing chosen
        if param_name == '' or ss_direction == '':
            return False

        grid_param_dict = self._core.strain_stress_calculator.get_raw_grid_param_values(ss_direction, param_name)
        # convert to a 2D array (grid position vector) and a 1D array (parameter value)
        vec_tuple = self._convert_grid_dict_to_vectors(ss_direction,  grid_param_dict)
        self._slice_view_exp_grid_vec, self._slice_view_exp_value_vec = vec_tuple
        self._slice_view_label = '{} @ {}'.format(param_name, ss_direction)

        return


        # # TODO - 20181010 - Consider how to integrate method plot_peak_parameter()
        # self.plot_peak_parameter()
        #
        # # TODO - FIXME - It is not correct anymore for
        # raise RuntimeError('Time to refactor/rewrite!')
        # if is_raw_grid:
        #     # parameter value on raw grids
        #     # get value
        #     grid_param_dict = self._core.strain_stress_calculator.get_raw_grid_param_values(ss_direction, param_name)
        #     # convert to a 2D array (grid position vector) and a 1D array (parameter value)
        #     self._slice_view_exp_grid_vec, self._slice_view_exp_value_vec = self._convert_grid_dict_to_vectors(ss_direction,
        #                                                                                                        grid_param_dict)
        # else:
        #     # mapped/interpolated parameter value on output strain/stress grid
        #     self._slice_view_exp_grid_vec = self._core.strain_stress_calculator.get_strain_stress_grid()
        #     self._slice_view_exp_value_vec = self._core.strain_stress_calculator.get_user_grid_param_values(ss_direction,
        #                                                                                                     param_name)
        # # END-IF-ELSE
        #
        # # set up slier slider
        # slice_dir = self.ui.comboBox_sliceDirection.currentIndex()  # as X
        # min_value = numpy.min(self._slice_view_exp_grid_vec[:, slice_dir])
        # max_value = numpy.max(self._slice_view_exp_grid_vec[:, slice_dir])
        #
        # self.ui.lineEdit_sliceStartValue.setText('{}'.format(min_value))
        # self.ui.lineEdit_sliceEndValue.setText('{}'.format(max_value))
        # self.ui.lineEdit_sliceStartValue.setEnabled(False)
        # self.ui.lineEdit_sliceEndValue.setEnabled(False)
        #
        # self.ui.horizontalSlider_slicer3D.setRange(self._slider_min, self._slider_max)
        # self.ui.horizontalSlider_slicer3D.setTickInterval(1)
        # self.ui.horizontalSlider_slicer3D.setTickPosition(0)
        #
        # # slice the data
        # vec_x, vec_y, vec_z, info = self._slice_values_on_grid(0, 99)
        #
        # # plot
        # RESOLUTION = 1  # TODO - 20180905 - Need to be configurable
        # self.ui.graphicsView_sliceView.plot_contour_interpolate(vec_x, vec_y, vec_z,
        #                                                         contour_resolution=(vec_x.max() - vec_x.min())/RESOLUTION,
        #                                                         contour_resolution_y=(vec_y.max() - vec_y.min())/RESOLUTION,
        #                                                         flush=True)
        # self.ui.graphicsView_sliceView.plot_scatter(vec_x, vec_y, flush=True)
        #
        # # 6. store the 3D data
        #
        # # 7. self.ui.horizontalSlider_slicer3D: event handling
        #
        #
        # return

    def _set_strain_stress_to_plot(self, is_strain, matrix_index):
        """ set the strain or stress to 3D data to plot
        :param is_strain:
        :param matrix_index:
        :return:
        """
        checkdatatypes.check_bool_variable('Flag showing it a strain', is_strain)
        # checkdatatypes.check_tuple('Matrix index (i, j)', matrix_index, 2)

        # print ('[DB...BAT] Matrix index = {} of type {}'.format(matrix_index, type(matrix_index)))

        # get data and set to class variable
        self._slice_view_exp_grid_vec, strain_vec, stress_vec = \
            self._core.strain_stress_calculator.get_strain_stress_values()

        if is_strain:
            slice_view_param_vec = strain_vec
            label = 'Strain '
        else:
            slice_view_param_vec = stress_vec
            label = 'Stress '

        # get the matrix element out: need to convert to (1, 2, 3) to (0, 1, 2)
        self._slice_view_exp_value_vec = slice_view_param_vec[:, matrix_index[0]-1, matrix_index[1]-1]
        label += '({}, {}) '.format(matrix_index[0], matrix_index[1])
        self._slice_view_label = label

        return

    def save_stress_strain(self):
        """
        save the calculated strain/stress file
        :return:
        """
        file_name = str(self.ui.lineEdit_outputFileName.text()).strip()
        if file_name == '':
            # file name is not defined
            file_name = gui_helper.browse_file(self, 'Save Strain and Stress', self._core.working_dir,
                                               file_list=False, save_file=True, file_filter='CSV (*.csv)')
            if file_name == '' or file_name is None:
                return
            else:
                self.ui.lineEdit_outputFileName.setText(file_name)
        # END-IF

        try:
            self._core.strain_stress_calculator.save_strain_stress(file_name)
        except RuntimeError as run_err:
            gui_helper.pop_message(self, message_type='error', message='Unable to save strain and stress',
                                   detailed_message=str(run_err))

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

    def do_reset_strain_stress(self):
        """
        reset the strain and stress type
        :return:
        """
        # check mutex
        if self._strain_stress_type_mutex:
            return

        type_index = self.ui.comboBox_typeStrainStress.currentIndex()
        is_plane_strain = False
        is_plane_stress = False
        if type_index == 1:
            is_plane_strain = True
        elif type_index == 2:
            is_plane_stress = True

        try:
            self._core.reset_strain_stress(is_plane_strain, is_plane_stress)
        except RuntimeError as run_err:
            gui_helper.pop_message(self, 'Failed to reset strain/stress type',
                                   detailed_message=str(run_err), message_type='error')
            return

        # set the UI
        if is_plane_stress or is_plane_strain:
            self.ui.lineEdit_e33ScanFile.setEnabled(False)
            self.ui.groupBox_calculator.setEnabled(False)
            self.ui.pushButton_browse_e33ScanFile.setEnabled(False)
            self.ui.comboBox_sampleLogNameX_E33.setEnabled(False)
            self.ui.comboBox_sampleLogNameY_E33.setEnabled(False)
            self.ui.comboBox_sampleLogNameZ_E33.setEnabled(False)
        else:
            self.ui.lineEdit_e33ScanFile.setEnabled(True)
            self.ui.groupBox_importData.setEnabled(True)
            self.ui.groupBox_alignGrids.setEnabled(False)
            self.ui.groupBox_calculator.setEnabled(False)
            self.ui.pushButton_browse_e33ScanFile.setEnabled(True)
            self.ui.comboBox_sampleLogNameX_E33.setEnabled(True)
            self.ui.comboBox_sampleLogNameY_E33.setEnabled(True)
            self.ui.comboBox_sampleLogNameZ_E33.setEnabled(True)

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
        # show table
        self._grid_alignment_table_view.show()

        # set to tab-2
        self._grid_alignment_table_view.show_tab(4)

        # try to gather some grid alignment information from loaded data
        exp_aligned_grids_list = self._core.strain_stress_calculator.get_experiment_aligned_grids()
        if exp_aligned_grids_list is not None:
            self._grid_alignment_table_view.set_matched_girds_info(exp_aligned_grids_list)

        return

    def do_show_prev_slice(self):
        """
        show the previous slice
        :return:
        """
        slider_value = self.ui.horizontalSlider_slicer3D.value()
        if slider_value > 0:
            slider_value -= 1
        self.ui.horizontalSlider_slicer3D.setValue(slider_value)

        print ('[DB...BAT] New slider value: {}'.format(slider_value))

        return

    def do_show_next_slice(self):
        """
        show the next slice
        :return:
        """
        slider_value = self.ui.horizontalSlider_slicer3D.value()
        print ('[DB...BAT] Slider maximum: {}'.format(self.ui.horizontalSlider_slicer3D.maximum()))
        if slider_value < self.ui.horizontalSlider_slicer3D.maximum():
            slider_value += 1
        self.ui.horizontalSlider_slicer3D.setValue(slider_value)

        print ('[DB...BAT] New slider value: {}'.format(slider_value))

        return

    def do_show_strain_stress_table(self):
        """
        show the calculated strain and stress values
        :return:
        """
        if self._strain_stress_table_view is None:
            self._strain_stress_table_view = dialogs.StrainStressTableView(self)
        else:
            self._strain_stress_table_view.reset_main_table()

        # get value and setup
        grid_array, strain_vec, stress_vec = self._core.strain_stress_calculator.get_strain_stress_values()
        self._strain_stress_table_view.set_strain_stress_values(grid_array, strain_vec, stress_vec)

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

    def show_message(self, message):
        """
        show message in the plain text edit
        :param message:
        :return:
        """
        checkdatatypes.check_string_variable('Message to show', message)

        self.ui.plainTextEdit_info.clear()
        self.ui.plainTextEdit_info.setPlainText(message)

        return

    def set_to_slice_peaks(self):
        """
        for external to call
        :return:
        """
        self.ui.comboBox_type.setCurrentIndex(0)

        return
