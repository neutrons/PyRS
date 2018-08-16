try:
    from PyQt5.QtWidgets import QDialog, QMainWindow
    from PyQt5.QtCore import pyqtSignal
except ImportError:
    from PyQt4.QtGui import QDialog, QMainWindow
    from PyQt4.QtCore import pyqtSignal

import gui_helper
from pyrs.utilities import checkdatatypes
from ui import ui_newsessiondialog
from ui import ui_strainstressgridsetup


class CreateNewSessionDialog(QDialog):
    """ Create a new strain/stress session dialog
    """
    NewSessionSignal = pyqtSignal(str, bool, bool, name='new session signal')

    def __init__(self, parent):
        """
        initialization
        :param parent:
        """
        super(CreateNewSessionDialog, self).__init__(parent)

        self.ui = ui_newsessiondialog.Ui_Dialog()
        self.ui.setupUi(self)

        self.ui.buttonBox.accepted.connect(self.do_quit)
        self.ui.buttonBox.rejected.connect(self.do_quit)

        # connect the signal
        self.NewSessionSignal.connect(parent.new_strain_stress_session)

        # init widgets
        self.ui.comboBox_strainStressType.clear()
        self.ui.comboBox_strainStressType.addItem('Unconstrained Strain/Stress')
        self.ui.comboBox_strainStressType.addItem('Plane Strain')
        self.ui.comboBox_strainStressType.addItem('Plane Stress')

        return

    def do_new_session(self):
        """
        new session
        :return:
        """
        session_name = str(self.ui.lineEdit_sessionName.text()).strip()
        if len(session_name) == 0:
            gui_helper.pop_message(self, 'Session name must be specified', 'error')
            return

        ss_type = self.ui.comboBox_strainStressType.currentIndex()
        is_plane_stress = False
        is_plane_strain = False

        if ss_type == 1:
            is_plane_strain = True
        elif ss_type == 2:
            is_plane_stress = True

        # send signal
        self.NewSessionSignal.emit(session_name, is_plane_strain, is_plane_stress)

        # quit eventually
        self.do_quit()

        return

    def do_quit(self):

        self.close()

    def reset_dialog(self):
        """
        reset the dialog input from last use
        :return:
        """
        self.ui.lineEdit_sessionName.setText('')

        return


class GridAlignmentCheckTableView(QMainWindow):
    """

    """

    def __init__(self, parent):
        """

        :param parent:
        """
        import ui.ui_gridsalignmentview

        super(GridAlignmentCheckTableView, self).__init__(parent)

        self.ui = ui.ui_gridsalignmentview.Ui_MainWindow()
        self.ui.setupUi(self)

        # TODO - 20180814 - Clean up
        # self.ui.actionQuit
        #

        return


# TODO - 20180818 - Clean up!
class StrainStressGridSetup(QDialog):
    """

    """
    def __init__(self, parent):
        """

        :param parent:
        """
        super(StrainStressGridSetup, self).__init__(parent)

        self.ui = ui_strainstressgridsetup.Ui_Dialog()
        self.ui.setupUi(self)

        self.ui.buttonBox.accepted.connect(self.do_accept_user_input)

        self._is_user_specified_grids = False

        # for return
        self._grid_setup_dict = dict()

        return

    def do_accept_user_input(self):
        """ pay attention for extra check up for user-specified grids
        :return:
        """
        try:
            self._parse_user_input(allow_empty=not self._is_user_specified_grids)
        except RuntimeError as run_err:
            gui_helper.pop_message(self,
                                   'User specified grids requiring all values to be input: {}'.format(run_err),
                                   'error')
            return
        except ValueError as value_err:
            gui_helper.pop_message(self, 'User specified value incorrect: {}'.format(value_err),
                                   'error')
            return

        super(StrainStressGridSetup, self).accept()

        return

    def _parse_user_input(self, allow_empty):
        """
        get the user inputs
        Example: lineEdit_gridMinX
        :return: as a dictionary
        """
        checkdatatypes.check_bool_variable('Flag to allow empty input', allow_empty)

        self._grid_setup_dict = dict()
        for param_name in ['Min', 'Max', 'Resolution']:
            for coord_i in ['X', 'Y', 'Z']:
                self._grid_setup_dict[param_name][coord_i] = None

                line_edit_name = 'lineEdit_grid{}{}'.format(param_name, coord_i)
                line_edit = getattr(self.ui, line_edit_name)
                value_str = str(line_edit.text()).strip()
                if value_str == '' and not allow_empty:
                    raise RuntimeError('{} {} is empty'.format(param_name, coord_i))
                elif value_str != '':
                    value_i = float(value_str)
                    self._grid_setup_dict[param_name][coord_i] = value_i
            # END-FOR (coord_i)
        # END-FOR (param_name)

        return

    def get_grid_setup(self):
        """
        get setup
        :return:
        """
        return self._grid_setup_dict

    def set_user_grids(self, state=True):
        """
        set the status to be user specified grids or not
        :param state: 
        :return: 
        """
        checkdatatypes.check_bool_variable('Flag for user specified grid', state)
        self._is_user_specified_grids = state

        return

    def set_experimental_data_statistics(self, stat_dict):
        """
        set the statistics data for the experiments
        [requirement] statistics dictionary
        level 1: type: min, max, num_indv_values
          level 2: direction: e11, e22(, e33)
            level 3: coordinate_dir: x, y, z
        :param stat_dict:
        :return:
        """
        checkdatatypes.check_dict('Grids statistics', stat_dict)

        # set up the minimum values, maximum values and number of individual values
        for dir_i in stat_dict['min'].keys():
            for coord_i in ['X', 'Y', 'Z']:
                line_edit_name = 'lineEdit_{}{}Min'.format(dir_i, coord_i)
                line_edit = getattr(self.ui, line_edit_name)
                line_edit.setText('{}'.format(stat_dict['min'][dir_i][coord_i]))
            # END-FOR
        # END-FOR

        # set up the maximum values
        for dir_i in stat_dict['max'].keys():
            for coord_i in ['X', 'Y', 'Z']:
                line_edit_name = 'lineEdit_{}{}Max'.format(dir_i, coord_i)
                line_edit = getattr(self.ui, line_edit_name)
                line_edit.setText('{}'.format(stat_dict['max'][dir_i][coord_i]))
            # END-FOR
        # END-FOR

        # set up the number of individual values
        for dir_i in stat_dict['num_indv_values'].keys():
            for coord_i in ['X', 'Y', 'Z']:
                line_edit_name = 'lineEdit_{}NumIndvPoints{}'.format(dir_i, coord_i)
                line_edit = getattr(self.ui, line_edit_name)
                line_edit.setText('{}'.format(stat_dict['num_indv_values'][dir_i][coord_i]))
            # END-FOR
        # END-FOR

        return


# END-DEFINE-CLASS


def get_strain_stress_grid_setup(parent, grid_stat_dict):
    """

    :return:
    """
    # set up dialog
    ss_dialog = StrainStressGridSetup(parent)
    ss_dialog.set_experimental_data_statistics(grid_stat_dict)

    # launch dialog and wait for result
    result = ss_dialog.exec_()

    # process result
    if not result:
        return None

    grid_setup_dict = ss_dialog.get_grid_setup()

    return grid_setup_dict
