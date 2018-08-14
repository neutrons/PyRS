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
        self._grid_alignment_table = None
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
        self.ui.pushButton_alignSampleLogXYZ.clicked.connect(self.do_align_xyz)
        self.ui.pushButton_calUnconstrainedStress.clicked.connect(self.do_cal_unconstrained_strain_stress)
        self.ui.pushButton_launchSSTable.clicked.connect(self.do_show_strain_stress_table)

        self.ui.pushButton_setd0Grid.clicked.connect(self.do_set_d0_by_grid)
        self.ui.pushButton_showAlignGridTable.clicked.connect(self.do_show_aligned_grid)

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
        # self.graphicsView_sliceView..connect(self.)
        # self.lineEdit_sliceStartValue..connect(self.)
        # self.lineEdit_sliceEndValue..connect(self.)
        # self.horizontalSlider_slicer..connect(self.)

        # current data/states
        self._core = None
        self._curr_data_key = None
        self._session_name = None

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
        self.ui.comboBox_alignmentCriteria.addItem('Finest Grid (Auto)')
        self.ui.comboBox_alignmentCriteria.addItem('E11')
        self.ui.comboBox_alignmentCriteria.addItem('E22')
        self.ui.comboBox_alignmentCriteria.addItem('E33')
        self.ui.comboBox_alignmentCriteria.addItem('User specified Grid')


        return

    def do_align_xyz(self):
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
            self._core.strain_stress_calculator.align_grids(resolution=0.001)
            print ('Intermittent 2')

        self.ui.groupBox_calculator.setEnabled(True)

        return

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

    def do_cal_unconstrained_strain_stress(self):
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

        # call the core to calculate strain
        self._core.calcualte_uncontrained_strain(self._session_name, e_young, nu_poisson)

        # TODO

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

        return

    def do_load_peak_info_files(self):
        """
        load peak information files
        :return:
        """


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
            self._core.strain_stress_calculator.load_raw_file(file_name=file_name, direction=direction)
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
        :param self:
        :return:
        """
        # TODO - 20180813 - Implement the table view
        if self._grid_alignment_table is None:
            self._grid_alignment_table = dialogs.GridAlignmentCheckTableView(self)
        else:
            self._grid_alignment_table.reset_table()

        # set up
        # TODO - 20180814 - self._grid_alignment_table.set_alignment_info(self._core.get_alignment_info())

        # show table
        self._grid_alignment_table.show()


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



