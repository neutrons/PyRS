# a collection of helper methdos for GUI
import os
from pyrs.core import rshelper
try:
    from PyQt5.QtWidgets import QDialog, QLineEdit
    is_qt4 = False
except ImportError:
    from PyQt4.QtGui import QDialog, QLineEdit
    from PyQt4 import QtCore
    is_qt4 = True


def parse_integer(int_str):
    """
    parse integer from a string or a LineEdit
    :param int_str:
    :return:
    """
    if isinstance(int_str, QLineEdit):
        int_str = str(int_str.text())
    elif is_qt4 and isinstance(int_str, QtCore.QString):
        # There is no QString in PyQt5
        int_str = str(int_str)
    else:
        rshelper.check_string_variable('Integer string', int_str)

    try:
        int_value = int(int_str)
    except ValueError as value_error:
        raise RuntimeError('Unable to parse {0} to integer due to {1}'.format(int_str, value_error))

    return int_value


def parse_integers(int_list_string):
    """ parse a list of integers.  Note that the start is inclusive and the end is exclusive
    example 1:4, 6:12, 8:12
    :param int_list_string:
    :return: list of int or range tuples
    """
    rshelper.check_string_variable('Integer list (string)', int_list_string)

    # remove unnecessary spaces
    int_list_string = int_list_string.replace(' ', '')

    # split by ,
    int_range_list = int_list_string.split(',')

    # parse to integers
    int_list = list()
    try:
        for int_range in int_range_list:
            if int_range.isdigit():
                # is an integer
                int_list.append(int(int_range))
            elif int_range.count(':') == 1:
                # integer range:
                int_str_list = int_range.split(':')
                start_int = int(int_str_list[0])
                end_int = int(int_str_list[1])
                int_list.extend(range(start_int, end_int))
            else:
                raise ValueError('{0} is not recognized'.format(int_range))
    except ValueError as val_err:
        raise RuntimeError('Unable to parse integer list {0} due to {1}'.format(int_list_string, val_err))

    # remove additional integers
    int_list = list(set(int_list))

    int_list.sort()

    return int_list


def pop_message(parent, message, message_type='error'):
    """

    :param parent:
    :param message:
    :param message_type: str as ['error', 'warning', 'info'] but NOT case sensitive
    :return:
    """
    message_type = message_type.lower()
    if message_type not in ['error', 'warning', 'info']:
        raise TypeError('Message type {0} is not supported.'.format(message_type))

    # TODO finish it!

    return


