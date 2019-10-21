#!/usr/bin/python
# Test utilities methods without GUI
from pyrs.interface import gui_helper
import pytest


def test_parse_integers():
    """
    test main
    :return:
    """
    print(gui_helper.parse_integers('3, 4, 5'))
    print(gui_helper.parse_integers('3:5, 4:10, 19'))

    try:
        int_list = gui_helper.parse_integers('3.2, 4')
    except RuntimeError as run_err:
        print run_err
    else:
        raise AssertionError('Shall be failed but get {0}'.format(int_list))


if __name__ == '__main__':
    pytest.main()
