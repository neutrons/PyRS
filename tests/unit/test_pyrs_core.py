#!/usr/bin/python
from pyrs.core import pyrscore
import os
import pytest


def test_main():
    """
    test main
    :return:
    """
    rs_core = pyrscore.PyRsCore()

    # pre-requisite is that the data file exists
    test_data = 'data/Hydra_16-1_cor_log.hdf5'
    assert os.path.exists(test_data), 'File {} does not exist'.format(test_data)

    # load data
    test_hd_ws = rs_core.load_hidra_project(test_data, 'test core', False, True)
    assert test_hd_ws

    # Get sub runs
    sub_runs = rs_core.reduction_service.get_sub_runs('test core')
    assert sub_runs

    # Get sample logs
    log_names = rs_core.reduction_service.get_sample_logs_names('test core', None)
    assert isinstance(log_names, list)

    return


if __name__ == '__main__':
    pytest.main()
