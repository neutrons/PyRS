"""
Integration test for workflow used in manual reduction UI
"""
import numpy as np
import os
import pytest
from pyrs.interface.manual_reduction.pyrs_api import ReductionController, reduce_hidra_workflow
from pyrs.core.powder_pattern import ReductionApp
import h5py


def test_get_ipts():
    """Test to get IPTS directory from run number

    Returns
    -------

    """
    # Test good
    test_ipts_dir = ReductionController.get_ipts_from_run(1060)
    assert test_ipts_dir == '/HFIR/HB2B/IPTS-22731/', 'IPTS dir {} is not correct for run {}' \
                                                      ''.format(test_ipts_dir, 1060)

    # Test no such run
    test_ipts_dir = ReductionController.get_ipts_from_run(112123260)
    assert test_ipts_dir == '/HFIR/HB2B/IPTS-22731/', 'IPTS dir {} is not correct for run {}' \
                                                      ''.format(test_ipts_dir, 112123260)

    # Test exception
    ReductionController.get_ipts_from_run(1.2)
    ReductionController.get_ipts_from_run('1.2')
    ReductionController.get_ipts_from_run('abc')


def test_find_run():
    """Test to find NeXus file

    Returns
    -------

    """
    test_nexus = ReductionController.get_nexus_file_by_run(1017)
    assert test_nexus == '/HFIR/HB2B/IPTS-22731/nexus/HB2B_1017.nxs.h5'

    # case for invalid run number
    test_nexus = ReductionController.get_nexus_file_by_run(1)
    assert test_nexus is None


def test_default_calibration_file():
    """Test to find current/latest calibration file

    Returns
    -------

    """
    default_calib_file = ReductionController.get_default_calibration_dir()

    if os.path.exists('/HFIR/HB2B/shared'):
        assert os.path.exists(default_calib_file)
        assert default_calib_file.lower().endswith('.json'), 'Must be a JSON file'
    else:
        pytest.skip('Unable to access HB2B archive')


@pytest.mark.parametrize('nexus_file, calibration_file, mask_file, gold_file',
                         [('/HFIR/HB2B/nexus/IPTS-22732/nexus/HB2B_1017.nxs.h5', None, None,
                           'data/gold/1017_NoMask.h5'),
                          ('/HFIR/HB2B/nexus/IPTS-22732/nexus/HB2B_1017.nxs.h5', None,
                           'data/HB2B_Mask_12-18-19.xml', 'data/gold/1017_Mask.h5'),
                          ('/HFIR/HB2B/nexus/IPTS-22732/nexus/HB2B_1017.nxs.h5', 'data/HB2B_CAL_Si333.json',
                           'data/HB2B_Mask_12-18-19.xml', 'data/gold/1017_Mask.h5')],
                         ids=('HB2B_1017_Masked', 'HB2B_1017_NoMask'))
def test_manual_reduction(nexus_file, calibration_file, mask_file, gold_file):
    """Test the workflow to do manual reduction.

    From splitting sub runs and sample logs, converting to powder pattern and then save
    including 3 cases:
    (1) No calibration, No masking
    (2) No calibration, Masking
    (3) Calibration, Masking

    Parameters
    ----------
    nexus_file
    calibration_file
    mask_file
    gold_file

    Returns
    -------

    """
    if os.path.exists(nexus_file) is False:
        pytest.skip('Testing file {} cannot be accessed'.format(nexus_file))

    # Get output directory and reduce
    output_dir = os.getcwd()
    target_file_path = 'HB2B_1017_{}_{}.h5'.format(calibration_file is not None, mask_file is not None)
    # remove previously generated
    if os.path.exists(target_file_path):
        os.remove(target_file_path)

    reduce_hidra_workflow(nexus_file, output_dir, progressbar=None,
                          calibration=calibration_file, mask=mask_file,
                          project_file_name=target_file_path)

    # Check whether the target file generated
    assert os.path.exists(target_file_path), 'Hidra project file {} is not generated'.format(target_file_path)

    # using gold file to compare the result
    parse_gold_file(gold_file)

    # delete
    # os.remove(target_file_path)

    return


def test_load_split():
    """Test method to load, split, convert to powder pattern and save

    Returns
    -------

    """
    pytest.skip('Manual reduction UI classes has not been refactored yet.')

    # Init load/split service instance

    nexus_file = '/HFIR/HB2B/nexus/IPTS-22732/nexus/HB2B_1017.nxs.h5'

    # Get list of sub runs
    # Get output directory and reduce
    output_dir = os.getcwd()
    target_file_path = 'HB2B_1017_test.h5'
    # remove previously generated
    if os.path.exists(target_file_path):
        os.remove(target_file_path)

    # Reduce (full)
    controller = ReductionController()
    controller.reduce_hidra_workflow(nexus_file, output_dir, progressbar=None,
                                     calibration=None, mask=None,
                                     project_file_name=target_file_path)

    # Get counts
    sub_run_1_counts = controller.get_detector_counts(1, True)
    assert sub_run_1_counts.shape == (1024, 1024)
    assert np.sum(sub_run_1_counts) > 1024**2, 'Sub run 1 counts = {} is too small'.format(sub_run_1_counts.sum())

    # Get diffraction pattern
    vec_2theta, vec_intensity = controller.get_powder_pattern(2)
    assert 78 < vec_2theta.mean() < 82, '2theta range ({}, {}) shall be centered around 80 for sub run 2.' \
                                        ''.format(vec_2theta[0], vec_2theta[-1])
    assert vec_intensity.max() > 5

    # Sample logs
    assert controller.get_sample_log_value('2theta', 1) == 69.5
    assert controller.get_sample_log_value('2theta', 2) == 80.0
    assert controller.get_sample_log_value('2theta', 3) == 90.0


@pytest.mark.parametrize('project_file, calibration_file, mask_file, gold_file',
                         [('data/HB2B_1017.h5', None, None, 'data/gold/1017_NoMask.h5'),
                          ('data/HB2B_1017.h5', None, 'data/HB2B_Mask_12-18-19.xml', 'data/gold/1017_Mask.h5')],
                         ids=('HB2B_1017_Masked', 'HB2B_1017_NoMask'))
def test_diffraction_pattern(project_file, calibration_file, mask_file, gold_file):
    """

    Parameters
    ----------
    project_file
    calibration_file
    mask_file : str or None
        mask file name
    gold_file

    Returns
    -------

    """
    pytest.skip('Manual reduction UI classes has not been refactored yet.')

    controller = ReductionController()

    # Load gold Hidra project file for diffraction pattern (run 1017)
    test_workspace = controller.load_project_file(project_file, True, False)

    # Convert to diffraction pattern
    powder_red_service = ReductionApp(use_mantid_engine=False)

    # Set workspace
    powder_red_service.load_hidra_workspace(test_workspace)

    # Reduction
    powder_red_service.reduce_data(sub_runs=None,
                                   instrument_file=None,
                                   calibration_file=calibration_file,
                                   mask=mask_file,
                                   mask_id=None,
                                   van_file=None,
                                   num_bins=1000)

    # Load gold data
    gold_data_set = parse_gold_file(gold_file)
    gold_sub_runs = np.array(gold_data_set.keys())
    gold_sub_runs.sort()

    #  Test number of sub runs
    np.testing.assert_allclose(gold_sub_runs, powder_red_service.get_sub_runs())

    # Get diffraction pattern
    for sub_run_i in gold_sub_runs:
        np.testing.assert_allclose(gold_data_set[sub_run_i], powder_red_service.get_diffraction_data(sub_run_i),
                                   rtol=1e-8)


def test_diffraction_pattern_geometry_shift():
    """

    Returns
    -------

    """
    pytest.skip('Manual reduction UI classes has not been refactored yet.')

    return


def parse_gold_file(file_name):
    """

    Parameters
    ----------
    file_name

    Returns
    -------
    ~dict or ~numpy.ndarray
        gold data in array or dictionary of arrays

    """
    # Init output
    data_dict = dict()

    # Parse file
    gold_file = h5py.File(file_name, 'r')
    data_set_names = list(gold_file.keys())
    for name in data_set_names:
        if isinstance(gold_file[name], h5py.Dataset):
            data_dict[name] = gold_file[name].value
        else:
            data_dict[name] = gold_file[name]['x'].value, gold_file[name]['y'].value
    # END-FOR

    if len(data_dict) == 1 and data_dict.keys()[0] == 'data':
        # only 1 array with default name
        return data_dict['data']

    return data_dict
