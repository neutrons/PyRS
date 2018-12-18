import threading
import types
import inspect
import sys
import os

home_dir = os.path.expanduser('~')
# # NOTE: This is the entry point to define the path to Mantid
# if home_dir.startswith('/SNS/'):
#     # analysis
#     # sys.path.insert(1, '/opt/mantid313/bin/')
#     sys.path.insert(1, '/opt/mantidnightly/bin/')
# elif home_dir.startswith('/home/wzz') is False:
#     # Mac debug build
#     sys.path.append('/Users/wzz/MantidBuild/debug-stable/bin')
#     # Analysis cluster build
#     # No need: auto set sys.path.append('/opt/mantidnightly/bin/')
#     sys.path.insert(1, '/SNS/users/wzz/Mantid_Project/builds/debug/bin')
#     # print ('system path: {0}'.format(sys.path))
#     # Personal VULCAN build
#     sys.path.append('/SNS/users/wzz/Mantid_Project/builds/build-vulcan/bin')
#     # sys.path.append('/SNS/users/wzz/Mantid_Project/builds/build-vulcan/bin')
# ....

# IPython monkey patches the  pygments.lexer.RegexLexer.get_tokens_unprocessed method
# and breaks Sphinx when running within MantidPlot.
# We store the original method definition here on the pygments module before importing IPython
from pygments.lexer import RegexLexer
# Monkeypatch!
RegexLexer.get_tokens_unprocessed_unpatched = RegexLexer.get_tokens_unprocessed

try:
    # This is PyQt5 compatible
    from qtconsole.rich_ipython_widget import RichIPythonWidget
    from qtconsole.inprocess import QtInProcessKernelManager
    print ('mantidipythonwidget: import PyQt5')
except ImportError as import_err:
    # This is PyQt4 compatible
    from IPython.qt.console.rich_ipython_widget import RichIPythonWidget
    from IPython.qt.inprocess import QtInProcessKernelManager
    print ('mantidipythonwidget: import PyQt4')
from mantid.api import AnalysisDataService as mtd

try:
    from PyQt5.QtWidgets import QApplication
except ImportError:
    from PyQt4.QtGui import QApplication


def our_run_code(self, code_obj, result=None):
    """ Method with which we replace the run_code method of IPython's InteractiveShell class.
        It calls the original method (renamed to ipython_run_code) on a separate thread
        so that we can avoid locking up the whole of MantidPlot while a command runs.
        Parameters
        ----------
        code_obj : code object
          A compiled code object, to be executed
        result : ExecutionResult, optional
          An object to store exceptions that occur during execution.
        Returns
        -------
        False : Always, as it doesn't seem to matter.
    """

    t = threading.Thread()
    #ipython 3.0 introduces a third argument named result
    nargs = len(inspect.getargspec(self.ipython_run_code).args)
    if (nargs == 3):
        t = threading.Thread(target=self.ipython_run_code, args=[code_obj,result])
    else:
        t = threading.Thread(target=self.ipython_run_code, args=[code_obj])
    t.start()
    while t.is_alive():
        QApplication.processEvents()
    # We don't capture the return value of the ipython_run_code method but as far as I can tell
    #   it doesn't make any difference what's returned
    return 0


class MantidIPythonWidget(RichIPythonWidget):
    """ Extends IPython's qt widget to include setting up and in-process kernel as well as the
        Mantid environment, plus our trick to avoid blocking the event loop while processing commands.
        This widget is set in the QDockWidget that houses the script interpreter within ApplicationWindow.
    """

    def __init__(self, *args, **kw):
        super(MantidIPythonWidget, self).__init__(*args, **kw)

        # Create an in-process kernel
        kernel_manager = QtInProcessKernelManager()
        kernel_manager.start_kernel()
        kernel = kernel_manager.kernel
        kernel.gui = 'qt4'

        # Figure out the full path to the mantidplotrc.py file and then %run it
        from os import path
        mantidplotpath = path.split(path.dirname(__file__))[0] # It's the directory above this one
        print '[....]  mantid plot path: ', mantidplotpath
        mantidplotrc = path.join(mantidplotpath, 'mantidplotrc.py')
        shell = kernel.shell
        shell.run_line_magic('run', mantidplotrc)
        print '[DB...BAUnderstand]: shell run: ', mantidplotrc

        # These 3 lines replace the run_code method of IPython's InteractiveShell class (of which the
        # shell variable is a derived instance) with our method defined above. The original method
        # is renamed so that we can call it from within the our_run_code method.
        f = shell.run_code
        shell.run_code = types.MethodType(our_run_code, shell)
        shell.ipython_run_code = f

        kernel_client = kernel_manager.client()
        kernel_client.start_channels()

        self.kernel_manager = kernel_manager
        self.kernel_client = kernel_client

        self._mainApplication = None

        return

    def execute(self, source=None, hidden=False, interactive=False):
        """
        Override super's execute() in order to emit customized signals to main application
        Parameters
        ----------
        source
        hidden
        interactive

        Returns
        -------

        """
        # record previous information: commened out for more test
        if self._mainApplication is not None:
            prev_workspace_names = set(mtd.getObjectNames())
        else:
            prev_workspace_names = None

        # interpret command: command is in self.input_buffer
        script = str(self.input_buffer).strip()

        # convert previous command "Run: vbin, ipts=18420, runs=139148, tag='C', output='\tmp'" to a property command
        if script.startswith('"Run: '):
            # strip "Run: and " away
            script = script.split('Run: ')[1]
            if script[-1] == '"':
                script = script[:-1]
        elif script.startswith('Run: '):
            # strip Run: away
            script = script.split('Run: ')[1]

        # main application is workspace viewer
        is_reserved = False
        if self._mainApplication.is_reserved_command(script):
            # reserved command: main application executes the command and return the message
            is_reserved = True
            # call main app/parent to execute the reserved command ***
            exec_message = self._mainApplication.execute_reserved_command(script)
            # create a fake command for IPython console (a do-nothing string)
            script_transformed = script[:]
            script_transformed = script_transformed.replace('"', "'")
            source = '\"Run: %s\"' % script_transformed
        else:
            exec_message = None

        # call base class to execute
        super(RichIPythonWidget, self).execute(source, hidden, interactive)

        # result message: append plain text to the console
        if is_reserved:
            #
            print ('[DB...BAT] Append Plain Text To Console: {}'.format(exec_message))
            self._append_plain_text('\n%s\n' % exec_message)

        # update workspaces for inline workspace operation
        if self._mainApplication is not None:
            post_workspace_names = set(mtd.getObjectNames())
            diff_set = post_workspace_names - prev_workspace_names
            self._mainApplication.process_workspace_change(diff_set)

        return

    def set_main_application(self, main_app):
        """
        Set the main application to the iPython widget to call
        Parameters
        ----------
        main_app :: main FastGR application

        Returns
        -------
        None
        """
        # check
        assert main_app is not None

        # set
        self._mainApplication = main_app

        return

    def append_string_in_console(self, input_str):
        """

        :param input_str:
        :return:
        """
        # TODO - NIGHT - Clean
        # check
        assert isinstance(input_str, str), 'blabla'

        # set
        curr_input = self.input_buffer
        self.input_buffer = '{} {}'.format(curr_input, input_str)

        return

    def write_command(self, command):
        """
        Write a command to the iPython console
        Args:
            command: string for a python command

        Returns:
            None
        """
        # TODO - NIGHT - Clean
        # check
        assert isinstance(command, str), 'blabla'

        # set
        self._store_edits()
        self.input_buffer = command

        return


