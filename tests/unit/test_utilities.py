#!/usr/bin/python
# Test utilities methods without GUI
from __future__ import (absolute_import, division, print_function)  # python3 compatibility
import os
from pyrs.interface import gui_helper
from pyrs.utilities import get_default_output_dir, get_input_project_file, get_ipts_dir, get_nexus_file
import pytest


MISSING_RUNNUMBER = 112123260


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
        get_ipts_dir(MISSING_RUNNUMBER)

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
        get_default_output_dir(MISSING_RUNNUMBER)


@pytest.mark.skipif(not os.path.exists('/HFIR/HB2B/shared/'), reason='HFIR data archive is not mounted')
def test_get_input_project_file():
    assert get_input_project_file(1060) == '/HFIR/HB2B/IPTS-22731/shared/manualreduce', 'Output directory is not '\
                                                                                        'correct for run 1060'

    # Test no such run
    with pytest.raises(RuntimeError):
        get_input_project_file(MISSING_RUNNUMBER)

    # Test bad preferred type
    with pytest.raises(ValueError):
        get_input_project_file(1060, preferredType='nonsense')


@pytest.mark.skipif(not os.path.exists('/HFIR/HB2B/shared/'), reason='HFIR data archive is not mounted')
def test_get_nexus_file():
    assert get_nexus_file(1060) == '/HFIR/HB2B/IPTS-22731/nexus/HB2B_1060.nxs.h5'
    assert get_nexus_file(1017) == '/HFIR/HB2B/IPTS-22731/nexus/HB2B_1017.ORIG.nxs.h5'

    # Test no such run
    with pytest.raises(RuntimeError):
        get_nexus_file(MISSING_RUNNUMBER)


if __name__ == '__main__':
    pytest.main()
