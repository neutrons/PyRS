########################################################################
#
# General-purposed plotting window
#
########################################################################
from mantidipythonwidget import MantidIPythonWidget
import time
try:
    from PyQt5 import QtCore
    from PyQt5.QtWidgets import QWidget
    from PyQt5.QtWidgets import QVBoxLayout
    from PyQt5.uic import loadUi as load_ui
except ImportError:
    from PyQt4 import QtCore
    from PyQt4.QtGui import QWidget
    from PyQt4.QtGui import QVBoxLayout
    from PyQt4.uic import loadUi as load_ui

from mplgraphicsview import MplGraphicsView
import NTableWidget as baseTable
from pyrs.interface.ui.mantidipythonwidget import MantidIPythonWidget

from mantid.api import AnalysisDataService
import mantid.simpleapi
import os

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s


class WorkspaceViewWidget(QWidget):
    """ Class for general-purposed plot window
    """
    # reserved command
    Reserved_Command_List = ['plot', 'refresh', 'exit', 'vhelp', 'what']

    def __init__(self, parent=None):
        """ Init
        """
        # call base
        QWidget.__init__(self)

        # Parent & others
        self._myMainWindow = None
        self._myParent = parent

        # set up UI
        ui_path = os.path.join(os.path.dirname(__file__), "WorkspacesView.ui")
        self.ui = load_ui(ui_path, baseinstance=self)
        self._promote_widgets()

        self.ui.tableWidget_dataStructure.setup()
        self.ui.widget_ipython.set_main_application(self)

        # define event handling methods
        # TODO - FUTURE - self.ui.pushButton_plot.clicked.connect(self.do_plot_workspace)
        self.ui.pushButton_toIPython.clicked.connect(self.do_write_workspace_name)
        self.ui.pushButton_toIPythonMtd.clicked.connect(self.do_write_workspace_instance)
        self.ui.pushButton_toIPythonAssign.clicked.connect(self.do_assign_workspace)
        self.ui.pushButton_clear.clicked.connect(self.do_clear_canvas)
        self.ui.pushButton_fitCanvas.clicked.connect(self.do_fit_canvas)

        return

    def _promote_widgets(self):
        """ promote widgets
        :return:
        """
        tableWidget_dataStructure_layout = QVBoxLayout()
        self.ui.frame_tableWidget_dataStructure.setLayout(tableWidget_dataStructure_layout)
        self.ui.tableWidget_dataStructure = WorkspaceTableWidget(self)
        tableWidget_dataStructure_layout.addWidget(self.ui.tableWidget_dataStructure)

        graphicsView_general_layout = QVBoxLayout()
        self.ui.frame_graphicsView_general.setLayout(graphicsView_general_layout)
        self.ui.graphicsView_general = WorkspaceGraphicView(self)
        graphicsView_general_layout.addWidget(self.ui.graphicsView_general)

        widget_ipython_layout = QVBoxLayout()
        self.ui.frame_widget_ipython.setLayout(widget_ipython_layout)
        self.ui.widget_ipython = MantidIPythonWidget(self)
        widget_ipython_layout.addWidget(self.ui.widget_ipython)

        return

    def do_clear_canvas(self):
        """
        clear the plots on the canvas
        :return:
        """
        self.ui.graphicsView_general.reset_canvas()

        return

    def do_fit_canvas(self):
        """
        resize the canvas to make the plots fit (0 to 5% above max value)
        :return:
        """
        self.ui.graphicsView_general.resize_canvas(0, 1.05)

        return

    def do_plot_workspace(self):
        """
        plot selected workspace
        :return:
        """
        # get selected workspace name
        selected_workspace_name_list = self.ui.tableWidget_dataStructure.get_selected_workspaces()

        # get the data from main application
        # controller = self._myMainWindow.get_controller()

        for workspace_name in selected_workspace_name_list:
            # data_set = controller.get_data_from_workspace(workspace_name)
            self.ui.graphicsView_general.plot_workspace(workspace_name)

        return

    def plot_diff_workspace(self, ws_name, bank_id):
        """
        ?????
        :param ws_name:
        :param bank_id:
        :return:
        """
        # get the data from main application
        controller = self._myMainWindow.get_controller()

        # use bank but not workspace index
        status, ret_obj = controller.get_data_from_workspace(ws_name, bank_id, target_unit=None, starting_bank_id=1)
        if not status:
            err_msg = str(ret_obj)
            return False, err_msg

        # being good
        data_set_dict, curr_unit = ret_obj

        for bank_id in data_set_dict:
            data_set = data_set_dict[bank_id]
            self.ui.graphicsView_general.plot_diffraction_data(data_set, ws_name, bank_id, curr_unit)

        return

    def do_assign_workspace(self):
        """
        write the workspace name to IPython console with assign the workspace instance to a variable
        :return:
        """
        # get workspace name
        ws_name_list = self.ui.tableWidget_dataStructure.get_selected_workspaces()

        # output string
        ipython_str = ''
        for ws_name in ws_name_list:
            ipython_str += 'ws_ = mtd["{0}"] '.format(ws_name)

        # export the ipython
        self.ui.widget_ipython.write_command(ipython_str)

        return

    def do_write_workspace_name(self):
        """
        write the workspace name to IPython console
        :return:
        """
        # get workspace name
        ws_name_list = self.ui.tableWidget_dataStructure.get_selected_workspaces()

        # output string
        ipython_str = ''
        for ws_name in ws_name_list:
            ipython_str += '"{0}"    '.format(ws_name)

        # export the ipython
        self.ui.widget_ipython.append_string_in_console(ipython_str)

        return

    def do_write_workspace_instance(self):
        """
        write the workspace name to IPython console
        :return:
        """
        # get workspace name
        ws_name_list = self.ui.tableWidget_dataStructure.get_selected_workspaces()

        # output string
        ipython_str = ''
        for ws_name in ws_name_list:
            ipython_str += 'mtd["{0}"] '.format(ws_name)

        # export the ipython
        self.ui.widget_ipython.append_string_in_console(ipython_str)

        return

    def execute_reserved_command(self, script):
        """ Execute command!
        :param script:
        :return:
        """
        script = script.strip()
        # get command name
        command = script.split(',')[0]

        print '[INFO] Executing reserved command: {}'.format(script)

        if command.startswith('plot'):
            status, exec_message = self.exec_command_plot(script)

        elif command == 'refresh':
            status, exec_message = self.exec_command_refresh()

        elif command == 'exit':
            self._myParent.close()
            # self.close()
            exec_message = None
            status = True

        elif command == 'vhelp' or command == 'what':
            # output help
            exec_message = self.get_help_message()
            status = True

        else:
            # Reserved VDRIVE-IDL command
            status, cmd_msg = self._myMainWindow.execute_command(script)
            # assertion error is not to be caught as it reflects coding error

            if status:
                exec_message = 'VDRIVE command {} is executed successfully ({}).'.format(command, cmd_msg)
            else:
                exec_message = 'VDRIVE command {} is failed to execute due to {}.'.format(command, cmd_msg)

        # ENDIF

        # Write to both plain text edit
        self.write_message(exec_message, False, status)
        self.write_message(exec_message, is_history_view=True)

        return exec_message

    @staticmethod
    def get_command_help(command):
        """
        get a help line for a specific command
        :param command:
        :return:
        """
        if command == 'plot':
            help_str = 'Plot a workspace.  Example: plot <workspace name>'

        elif command == 'refresh':
            help_str = 'Refresh the graph above.'

        elif command == 'exit':
            help_str = 'Exist the application.'

        elif command == 'vhelp' or command == 'what':
            # output help
            help_str = 'Get help.'

        else:
            help_str = 'Reserved VDRIVE command.  Run> %s' % command

        return help_str

    def get_help_message(self):
        """

        :return:
        """
        message = 'LAVA Reserved commands:\n'\

        for command in sorted(self.Reserved_Command_List):
            message += '%-15s: %s\n' % (command, self.get_command_help(command))

        return message

    def is_reserved_command(self, script):
        """ check a command is reserved or not
        :param script:
        :return:
        """
        # command can be IDL-like or Python-like
        command = script.strip().split(',')[0].strip()
        if command.count('('):
            command = script.split('(')[0].strip()

        is_reserved = command in self.Reserved_Command_List
        # if is_reserved:
        #     print ('[DB...INFO] command: {} is reserved'.format(command))

        return is_reserved

    def exec_command_clear(self, script):
        """

        :param script:
        :return:
        """
        # TODO - NIGHT - Make it better looking
        if terms[1] == 'clear':
            # clear canvas
            # TODO - 20181213 - Create a new reserved command
            self.ui.graphicsView_general.clear_all_lines()
            return_message = ''

        return return_message

    def exec_command_plot(self, script):
        """ execute command plot
        Use cases
        1. plot(workspace=abcd, bank=1)
        2. plot()
        3. plot: help function
        :param script:
        :return:
        """
        # TODO-In-Progress - 20181215 - clean this section
        script = script.strip()

        if script == 'plot':
            # no given option, it provides help information (man page)
            return_message = 'Reserved command to plot workspace(s)\n'
            return_message += 'Example 1:  plot(workspace_name, bank_id)\n'
            return_message += 'Example 2:  plot(workspace_name) plot all banks from selected workspace  in the table\n'
            return_message += 'Example 3:  plot(bank_id): plot selected workspace with bank ID'
            status = True
        else:
            # do something else
            plot_args = script[4:]
            print ('[DB...BAT] arguments: {}'.format(plot_args))
            plot_args = plot_args.strip()
            if plot_args.startswith('('):
                plot_args = plot_args[1:]
            if plot_args.endswith(')'):
                plot_args = plot_args[:-1]
            print ('[DB...BAT] Processed arguments: {}'.format(plot_args))

            # split to terms
            arg_terms = plot_args.split(',')
            if len(arg_terms) == 2:
                # Example 1:  plot(workspace_name, bank_id)
                ws_name = arg_terms[0].replace('\'', '').strip()
                ws_name = ws_name.replace('"', '').strip()
                try:
                    bank_id = int(arg_terms[1].strip())
                except ValueError:
                    return False, 'Bank ID {} must be an integer'.format(arg_terms[1])
            elif len(arg_terms) == 1:
                # Example 2 or 3
                value = arg_terms[0].strip()
                if value.count('\'') or value.count('"'):
                    # with ' or ", must be workspace
                    ws_name = value.replace('\'', '').replace('"', '').strip()
                    bank_id = None
                else:
                    # without ', then must be a bank ID
                    try:
                        bank_id = int(arg_terms[0].strip())
                    except ValueError:
                        return False, 'Bank ID {} must be an integer'.format(arg_terms[0])
                    ws_names = self.ui.tableWidget_dataStructure.get_selected_workspaces()
                    if len(ws_names) != 1:
                        return False, 'With only bank ID specified, there must be 1 and only 1 workspace ' \
                                      'selected in table'
                    else:
                        ws_name = ws_names[0]
            else:
                # too many
                return False, 'More than 2 arguments is not supported for command plot'

            if not AnalysisDataService.doesExist(ws_name):
                return False, 'Workspace {} ({}) does not exist.'.format(ws_name, type(ws_name))

            try:
                self.plot_diff_workspace(ws_name, bank_id)
                status = True
                return_message = '[DB...BAT] Processed arguments: {}'.format(plot_args)
            except RuntimeError as run_err:
                status = False
                return_message = '[Error] {}'.format(run_err)
        # END-IF-ELSE

        # switch to plot tab
        self.ui.tabWidget_table_view.setCurrentIndex(1)

        return status, return_message

    def process_workspace_change(self, diff_set):
        """

        :param diff_set:
        :return:
        """
        print ('Workspace set differece: {}'.format(diff_set))

        # TODO/NOW/ISSUE/51 - 20181214 - Implement!

        return

    def exec_command_refresh(self):
        """ Refresh workspace in the memory
        :return:
        """
        workspace_names = AnalysisDataService.getObjectNames()

        self.ui.tableWidget_dataStructure.remove_all_rows()
        error_message = ''
        for ws_name in workspace_names:
            try:
                # get workspace and its type
                workspace = AnalysisDataService.retrieve(ws_name)
                ws_type = workspace.id()
                # find out workspace information
                ws_info = ''
                if ws_type == 'EventWorkspace':
                    num_events = workspace.getNumberEvents()
                    num_hist = workspace.getNumberHistograms()
                    num_bins = len(workspace.readY(0))
                    ws_info = '{}/{}/{}'.format(num_events, num_hist, num_bins)
                elif ws_type == 'Workspace2D':
                    num_hist = workspace.getNumberHistograms()
                    num_bins = len(workspace.readY(0))
                    ws_info = '{}/{}'.format(num_hist, num_bins)
                elif ws_type == 'TableWorkspace':
                    num_rows = workspace.rowCount()
                    num_cols = workspace.columnCount()
                    ws_info = '{}/{}'.format(num_rows, num_cols)
                self.ui.tableWidget_dataStructure.add_workspace(ws_name, ws_type, ws_info)
            except Exception as ex:
                error_message += 'Unable to add %s to table due to %s.\n' % (ws_name, str(ex))
        # END-FOR

        # switch to table tab
        self.ui.tabWidget_table_view.setCurrentIndex(0)

        if len(error_message) == 0:
            return True, ''

        return False, error_message

    def set_main_window(self, main_window):
        """
        Set up the main window which generates this window
        :param main_window:
        :return:
        """
        # check
        assert main_window is not None
        # set
        self._myMainWindow = main_window

        try:
            main_window.get_reserved_commands
        except AttributeError as att_err:
            # TODO - FUTURE - Expand VDRIVE-like command
            return
            # raise AttributeError('Parent window does not have required method get_reserved_command. FYI: {0}'
            #                      ''.format(att_err))

        reserved_command_list = main_window.get_reserved_commands()
        self.Reserved_Command_List.extend(reserved_command_list)

        return

    def write_message(self, message_body, is_history_view=False, is_cmd_success=None):
        """
        write a message to the plain text edit
        :param message_body:
        :param is_history_view:
        :return:
        """
        # TODO - NIGHT - clean!
        import datetime
        cur_time = time.time()

        text = '{}:\n{}\n'.format(datetime.datetime.now(), message_body)

        if is_history_view:
            self.ui.plainTextEdit_loggingHistory.appendPlainText(text)
        else:
            assert isinstance(is_cmd_success, bool)
            if is_cmd_success:
                self.ui.plainTextEdit_info.clear()
                self.ui.plainTextEdit_info.appendPlainText(text)
                self.ui.tabWidget_logging.setCurrentIndex(0)
            else:
                self.ui.plainTextEdit_error.clear()
                self.ui.plainTextEdit_error.appendPlainText(text)
                self.ui.tabWidget_logging.setCurrentIndex(1)
        # END-IF

        return


class WorkspaceGraphicView(MplGraphicsView):
    """

    """
    BankColorDict = {1: 'black', 2: 'red', 3: 'blue'}

    def __init__(self, parent):
        """
        :param parent:
        """
        MplGraphicsView.__init__(self, None)

        self._parent = parent

        # class variable
        self._rangeX = (0, 1.)
        self._rangeY = (0, 1.)

        return

    def plot_diffraction_data(self, data_set, ws_name, bank_id, curr_unit):
        """
        ????
        :param data_set:
        :param ws_name:
        :param bank_id:
        :param curr_unit:
        :return:
        """
        # set X-axis
        self.canvas().set_xy_label(side='x', text=curr_unit, font_size=16)

        # get data
        vec_x = data_set[0]
        vec_y = data_set[1]

        # plot
        num_bins = len(vec_y)
        data_label = '{}: bank {} {}'.format(ws_name, bank_id, num_bins)
        self.plot_1d_data(vec_x, vec_y, bank_id, data_label)
        self._update_data_range(vec_x, vec_y)

        return

    def plot_workspace(self, workspace_name, unit=None, bank_id=None):
        """ Plot a workspace
        :param workspace_name:
        :param unit:
        :param bank_id:
        :return:
        """
        # TODO - 20181214 - New requests: - ToTest
        # TODO           1. Better label including X-unit, Legend (bank, # bins) and title (workspace name)
        # TODO           2. Use auto color
        # TODO           3. Use over-plot to compare
        # TODO           4. Change tab: ui.tabWidget_table_view
        # FIXME   -      This is a dirty shortcut because it is not suppose to access AnalysisDataService at this level

        # form bank IDs
        if bank_id is None:
            bank_id_list = self._parent.controller.get_bank_ids(workspace_name)

        else:
            bank_id_list = [bank_id]

        # unit
        if unit is None:
            unit = self._parent.controller.get_workspace_unit(workspace_name)
        # set unit
        self.canvas().set_xy_label(side='x', text='unit', font_size=16)

        for bank_id in sorted(bank_id_list):
            # get data
            data_set = self._parent.controller.get_diff_data(workspace_name, bank_id, unit)
            vec_x = data_set[0]
            vec_y = data_set[1]
            # plot
            num_bins = len(vec_y)
            data_label = '{}: bank {} {}'.format(workspace_name, bank_id, num_bins)
            self.plot_1d_data(vec_x, vec_y, bank_id, data_label)
            self._update_data_range(vec_x, vec_y)
        # END-FOR

        # # ws = AnalysisDataService.retrieve(workspace_name)
        # # mantid.simpleapi.ConvertToPointData(InputWorkspace=ws, OutputWorkspace='temp_ws')
        # # point_ws = AnalysisDataService.retrieve('temp_ws')
        #
        # # get X and Y
        # vec_x = point_ws.readX(0)
        # vec_y = point_ws.readY(0)
        #
        #
        #

        return

    def plot_1d_data(self, vec_x, vec_y, bank_id, data_label):
        """

        :param vec_x:
        :param vec_y:
        :param bank_id:
        :param data_label:
        :return:
        """
        line_color = WorkspaceGraphicView.BankColorDict[bank_id]

        # TODO - 20181215 - Shall the reference to line be handled somewhere?
        self.add_plot_1d(vec_x, vec_y, color=line_color, label=data_label)

        return

    def _update_data_range(self, vec_x, vec_y):
        """
        udpate the min and max of the data that is plot on figure now
        :param vec_x:
        :param vec_y:
        :return:
        """
        # get X and Y's range
        min_x = min(self._rangeX[0], vec_x[0])
        max_x = max(self._rangeX[1], vec_x[-1])

        min_y = min(self._rangeY[0], min(vec_y))
        max_y = max(self._rangeY[1], max(vec_y))

        self._rangeX = (min_x, max_x)
        self._rangeY = (min_y, max_y)

        return

    def resize_canvas(self, y_min, y_max_ratio):
        """

        :param y_min:
        :param y_max_ratio:
        :return:
        """
        y_max = self._rangeY[1] * y_max_ratio

        self.setXYLimit(self._rangeX[0], self._rangeX[1], y_min, y_max)

        return

    def reset_canvas(self):
        """
        reset the canvas by removing all lines and registered values
        :return:
        """
        self.clear_all_lines()

        self._rangeX = (0., 1.)
        self._rangeY = (0., 1.)

        self.setXYLimit(0., 1., 0., 1.)

        return

    @staticmethod
    def setInteractive(status):
        """
        It is a native method of QtCanvas.  It is not used in MplGraphicView at all.
        But the auto-generated python file from .ui file have this method added anyhow.
        :param status:
        :return:
        """
        return


class WorkspaceTableWidget(baseTable.NTableWidget):
    """
    Table Widget for workspaces
    """
    SetupList = [('Workspace', 'str'),
                 ('Type', 'str'),
                 ('Size', 'str'),
                 ('', 'checkbox')]

    def __init__(self, parent):
        """
        Initialization
        :param parent:
        """
        baseTable.NTableWidget.__init__(self, None)

    def setup(self):
        self.init_setup(self.SetupList)
        self.setColumnWidth(0, 360)
        return

    def add_workspace(self, ws_name, ws_type='', size_info=''):
        """ add a workspace
        :param ws_name:
        :param ws_type:
        :param size_info:
        :return:
        """
        self.append_row([ws_name, ws_type, size_info, False])

        return

    def get_selected_workspaces(self):
        """
        get the names of workspace in the selected rows
        :return:
        """
        selected_rows = self.get_selected_rows(True)

        print '[DB...BAT] selected rows: ', selected_rows

        ws_name_list = list()
        for i_row in selected_rows:
            ws_name = self.get_cell_value(i_row, 0)
            ws_name_list.append(ws_name)

        return ws_name_list



