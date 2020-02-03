#!/usr/bin/python
# Test utilities methods without GUI
from __future__ import (absolute_import, division, print_function)  # python3 compatibility
import os
from pyrs.interface import gui_helper
from pyrs.utilities.file_util import get_ipts_dir, get_default_output_dir
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
        print(run_err)
    else:
        raise AssertionError('Shall be failed but get {0}'.format(int_list))


@pytest.mark.skipif(not os.path.exists('/HFIR/HB2B/shared/'), reason='HFIR data archive is not mounted')
def test_get_ipts_dir():
    """Test to get IPTS directory from run number

    Returns
    -------

    """
    # Test good
    assert get_ipts_dir(1060) == '/HFIR/HB2B/IPTS-22731/', 'IPTS directory is not correct for run 1060'

    # Test no such run
    with pytest.raises(RuntimeError):
        get_ipts_dir(112123260)

    # Test exception
    with pytest.raises(TypeError):
        get_ipts_dir(1.2)
    with pytest.raises(ValueError):
        get_ipts_dir('1.2')
    with pytest.raises(ValueError):
        get_ipts_dir('abc')


@pytest.mark.skipif(not os.path.exists('/HFIR/HB2B/shared/'), reason='HFIR data archive is not mounted')
def test_get_default_output_dir():
    assert get_default_output_dir(1060) == '/HFIR/HB2B/IPTS-22731/shared/manualreduce', 'Output directory is not '\
                                                                                        'correct for run 1060'

    # Test no such run
    with pytest.raises(RuntimeError):
        get_default_output_dir(112123260)


if __name__ == '__main__':
    pytest.main()
