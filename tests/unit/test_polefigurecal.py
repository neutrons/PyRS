#!/usr/bin/python
from pyrs.core import pyrscore
import pytest


def broken_test_pole_figure_calculation():
    """
    main testing body to test the workflow to calculate pole figure
    :return:
    """
    # initialize core
    rs_core = pyrscore.PyRsCore()
    assert rs_core

    # import data file: detector ID and file name
    test_data_set = [(1, 'tests/testdata/HB2B_exp129_Long_Al_222[1]_single.hdf5'),
                     (2, 'tests/testdata/HB2B_exp129_Long_Al_222[2]_single.hdf5'),
                     (3, 'tests/testdata/HB2B_exp129_Long_Al_222[3]_single.hdf5'),
                     (4, 'tests/testdata/HB2B_exp129_Long_Al_222[4]_single.hdf5'),
                     (5, 'tests/testdata/HB2B_exp129_Long_Al_222[5]_single.hdf5'),
                     (6, 'tests/testdata/HB2B_exp129_Long_Al_222[6]_single.hdf5'),
                     (7, 'tests/testdata/HB2B_exp129_Long_Al_222[7]_single.hdf5')]
    assert test_data_set

    # data_key, message = rs_core.load_rs_raw_set(test_data_set)
    #
    # # peak fitting for detector - ALL
    # detector_id_list = rs_core.get_detector_ids(data_key)
    #
    # for det_id in detector_id_list:
    #     scan_range = rs_core.data_center.get_scan_range(data_key, det_id)
    #     rs_core.fit_peaks((data_key, det_id), scan_index=scan_range, peak_type='Gaussian',
    #                       background_type='Linear', fit_range=(80, 85))
    # # END-FOR

    # calculate pole figure
    # rs_core.calculate_pole_figure(data_key, range(1, 8))
    #
    # # export
    # rs_core.save_pole_figure(data_key, None, '/tmp/polefiguretest.mtex', 'mtex')
    # assert os.path.exists('/tmp/polefiguretest.mtex')
    # os.unlink('/tmp/polefiguretest.mtex')


if __name__ == '__main__':
    pytest.main()
