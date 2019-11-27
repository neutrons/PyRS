# a collection of helper methdos for GUI
from pyrs.utilities import checkdatatypes
from qtpy.QtWidgets import QLineEdit, QFileDialog, QMessageBox, QVBoxLayout


def browse_dir(parent, caption, default_dir):
    """ Browse a directory
    :param parent:
    :param caption:
    :param default_dir:
    :return: non-empty string for selected directory; empty string for canceled operation
    """
    # check inputs
    assert isinstance(parent, object), 'Parent {} must be of some object.'.format(parent)
    checkdatatypes.check_string_variable('File browsing title/caption', caption)
    checkdatatypes.check_file_name(default_dir, check_exist=False, is_dir=True)

    # get directory
    chosen_dir = QFileDialog.getExistingDirectory(parent, caption, default_dir)
    print('[DB...BAT] Chosen dir: {} of type {}'.format(chosen_dir, type(chosen_dir)))
    chosen_dir = str(chosen_dir).strip()

    return chosen_dir


def browse_file(parent, caption, default_dir, file_filter, file_list=False, save_file=False):
    """ browse a file or files
    :param parent:
    :param caption:
    :param default_dir:
    :param file_filter:
    :param file_list:
    :param save_file:
    :return: if file_list is False: return string (file name); otherwise, return a list;
             if user cancels the operation, then return None
    """
    # check inputs
    # assert isinstance(parent, object), 'Parent {} must be of some object.'.format(parent)
    # checkdatatypes.check_string_variable('File browsing title/caption', caption)
    # checkdatatypes.check_file_name(default_dir, check_exist=False, is_dir=True)
    # checkdatatypes.check_bool_variable('Flag for browse a list of files to load', file_list)
    # checkdatatypes.check_bool_variable('Flag to select loading or saving file', save_file)
    if file_filter is None:
        file_filter = 'All Files (*.*)'
    else:
        checkdatatypes.check_string_variable('File filter', file_filter)
        file_filter = '{};;All Files (*.*)'.format(file_filter)

    if save_file:
        # browse file name to save to
        save_set = QFileDialog.getSaveFileName(parent,
                                               caption=caption,
                                               directory=default_dir,
                                               filter=file_filter)
        if isinstance(save_set, tuple):
            # returned include both file name and filter
            file_name = str(save_set[0])
        else:
            file_name = str(save_set)

    elif file_list:
        # browse file names to load
        open_set = QFileDialog.getOpenFileNames(parent, caption, default_dir, file_filter)

        if isinstance(open_set, tuple):
            file_name_list = open_set[0]
        else:
            file_name_list = open_set

        if len(file_name_list) == 0: # user cancel browser
            return None
        else:
            return file_name_list

    else:
        # browse single file name
        open_set = QFileDialog.getOpenFileName(parent, caption, default_dir, file_filter)

        if isinstance(open_set, tuple):
            file_name = open_set[0]
        else:
            file_name = open_set

    # check result for single file whether user cancels operation
    if len(file_name) == 0:
        return None

    return str(file_name)


def get_save_file_name(parent, dir_name, caption, file_filter):
    # TODO - 20181204 - Replace getSaveFileName by a self-extended method to be fine with both Qt4 and Qt5 ASAP(1)
    file_info = QFileDialog.getSaveFileName(parent, directory=dir_name,
                                            caption=caption,
                                            filter=file_filter)

    if isinstance(file_info, tuple):
        file_name = str(file_info[0])
        file_filter = file_info[1]
        print('[DB...Save Pole Figure] File name: {0}, Filter = {1}'.format(file_name, file_filter))
    else:
        file_name = str(file_info)
        file_filter = None

    return file_name, file_filter


def parse_line_edit(line_edit, data_type, throw_if_blank=False, edit_name=None, default=None):
    """
    Parse a LineEdit
    :param line_edit:
    :param data_type:
    :param throw_if_blank:
    :param edit_name: name of LineEdit for better error message
    :param default: default value of the returned value
    :return:
    """
    assert isinstance(line_edit, QLineEdit), 'Method parse_line_edit expects 0-th input {} to be a ' \
                                             'QLineEdit instance but not a {}' \
                                             ''.format(line_edit, type(line_edit))
    assert isinstance(data_type, type), 'Method parse_line_edit expects 1-st input {} to be a type ' \
                                        'but not a {}'.format(data_type, type(data_type))
    checkdatatypes.check_bool_variable('Method parse_line_edit expects 2nd input "throw"', throw_if_blank)

    # parse
    input_str = str(line_edit.text()).strip()

    if input_str == '':
        # empty string
        if throw_if_blank:
            raise RuntimeError('Input line edit {} is empty'.format(edit_name))
        elif default is not None:
            return_value = default
        else:
            return_value = None

    else:
        try:
            return_value = data_type(input_str)
        except ValueError as value_error:
            raise RuntimeError('Unable to parse LineEdit {} (given value = {}) to '
                               '{} due to {}'.format(edit_name, input_str, data_type, value_error))
    # END-IF-ELSE

    return return_value


def parse_float(float_str):
    """
    parse flaots from a string or a LineEdit
    :param float_str:
    :return:
    """
    if isinstance(float_str, QLineEdit):
        # Input is QLineEdit
        float_str = str(float_str.text())
    else:
        # Input has to be string
        checkdatatypes.check_string_variable('Integer string', float_str)

    try:
        float_value = float(float_str)
    except ValueError as value_error:
        raise RuntimeError('Unable to parse {0} to integer due to {1}'.format(float_str, value_error))

    return float_value


def parse_integer(int_str):
    """
    parse integer from a string or a LineEdit
    :param int_str:
    :return:
    """
    if isinstance(int_str, QLineEdit):
        # QLineEdit: get the string out of it
        int_str = str(int_str.text())
    else:
        # Then it has to be a string
        checkdatatypes.check_string_variable('Integer string', int_str)

    try:
        int_value = int(int_str)
    except ValueError as value_error:
        raise RuntimeError('Unable to parse {0} to integer due to {1}'.format(int_str, value_error))

    return int_value


def parse_rigorous_int_string(int_str):
    """
    parse a string which must be an integer but  not anything can be converted to integer
    :param int_str:
    :return:
    """
    checkdatatypes.check_string_variable('Integer in string', int_str)

    # negative?
    if int_str.startswith('-'):
        sign = -1
        int_str = int_str.split('-')[1]
    else:
        sign = 1

    # must be an integer
    if int_str.isdigit() is False:
        raise ValueError('{} cannot be recognized as an integer rigorously'.format(int_str))

    # convert
    try:
        int_number = sign*int(int_str)
    except ValueError as val_err:
        raise ValueError('Unable to convert string {} to an integer: {}'.format(int_str, val_err))

    return int_number


def parse_integers(int_list_string):
    """ parse a list of integers.  Note that the start is inclusive and the end is exclusive
    example 1:4, 6:12, 8:12
    :param int_list_string:
    :return: list of int or range tuples
    """
    checkdatatypes.check_string_variable('Integer list (string)', int_list_string)

    # remove unnecessary spaces
    int_list_string = int_list_string.replace(' ', '')

    # split by ,
    int_range_list = int_list_string.split(',')

    # parse to integers
    int_list = list()
    try:
        for int_range in int_range_list:
            column_counts = int_range.count(':')

            if column_counts == 0:
                # single value
                int_list.append(parse_rigorous_int_string(int_range))

            elif column_counts == 1:
                # given a range
                int_str_list = int_range.split(':')
                start_int = parse_rigorous_int_string(int_str_list[0])
                end_int = parse_rigorous_int_string(int_str_list[1])
                int_list.extend(range(start_int, end_int))

            else:
                # bad inputs
                raise ValueError('{0} has too many : to recognize'.format(int_range))
    except ValueError as val_err:
        raise RuntimeError('Unable to parse integer list "{}" due to {}'.format(int_list_string, val_err))

    # remove additional integers
    int_list = list(set(int_list))

    int_list.sort()

    return int_list


def promote_widget(frame_widget, promoted_widget):
    """
    Add a promoted widget to a QFrame in the main window
    :param frame_widget:
    :param promoted_widget:
    :return:
    """
    curr_layout = QVBoxLayout()
    frame_widget.setLayout(curr_layout)
    curr_layout.addWidget(promoted_widget)

    return


def get_boolean_from_dialog(window_title, message):
    """
    pop out a dialog showing a message to user.  User will choose OK or Cancel
    :param window_title
    :param message:
    :return:
    """
    def msgbtn(i):
        # debugging output
        print "Button pressed is:", i.text()

    message_box = QMessageBox()
    message_box.setIcon(QMessageBox.Information)
    message_box.setText(message)
    if window_title is not None:
        message_box.setWindowTitle(window_title)
    message_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    message_box.buttonClicked.connect(msgbtn)

    # get the message executed
    return_value = message_box.exec_()

    # identify output
    if return_value == 4194304:
        return_value = True
    elif return_value == 1024:
        return_value = False
    else:
        raise RuntimeError('Return value {} of type {} is not recognized.'
                           ''.format(return_value, type(return_value)))

    return return_value


def parse_tuples(tuple_str, data_type, size=None):
    """

    :param tuple_str:
    :param data_type:
    :param size:
    :return:
    """
    # check type ...

    # doc...

    # strip parenthesis
    tuple_str = tuple_str.replace('(', '').replace(')', '')

    # TODO - 20180906 - Refine!
    items = tuple_str.strip().split(',')

    if size is not None:
        assert len(items) == size, '{} vs {}'.format(items, size)

    ret_list = list()
    for item in items:
        item = item.strip()
        item = data_type(item)
        ret_list.append(item)

    return ret_list


def pop_message(parent, message, detailed_message=None, message_type='error'):
    """ pop up a message with specified message type such as error, warning, info...
    :param parent:
    :param message:
    :param detailed_message: detailed message optionally shown to user
    :param message_type: str as ['error', 'warning', 'info'] but NOT case sensitive
    :return:
    """
    message_type = message_type.lower()
    if message_type not in ['error', 'warning', 'info']:
        raise TypeError('Message type {0} is not supported.'.format(message_type))

    # check types
    checkdatatypes.check_string_variable('(Main) message to show', message)
    if detailed_message is not None:
        checkdatatypes.check_string_variable('(Detailed) message to show', detailed_message)

    # create a QMessageBox
    msg_box = QMessageBox()

    # set information type
    if message_type == 'info':
        msg_box.setIcon(QMessageBox.Information)
    elif message_type == 'error':
        msg_box.setIcon(QMessageBox.Critical)
    elif message_type == 'warning':
        msg_box.setIcon(QMessageBox.Warning)

    # set text
    msg_box.setText(message)
    if detailed_message is not None:
        msg_box.setDetailedText(detailed_message)  # another button
    msg_box.setWindowTitle('PyRS Message')

    # box
    msg_box.setStandardButtons(QMessageBox.Ok)

    ret_val = msg_box.exec_()
    print('Message box return value: {}'.format(ret_val))

    return
