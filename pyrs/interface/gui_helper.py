# a collection of helper methdos for GUI
import os
try:
    from PyQt5.QtWidgets import QDialog
except ImportError:
    from PyQt4.QtGui import QDialog


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


