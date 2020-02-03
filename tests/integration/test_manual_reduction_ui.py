"""
Integration test for workflow used in manual reduction UI
"""
import numpy as np
import os
import pytest
from pyrs.interface.manual_reduction.pyrs_api import ReductionController, reduce_hidra_workflow
import h5py


def test_default_calibration_file():
    """Test to find current/latest calibration file

    Returns
    -------

    """
    default_calib_file = os.path.join(ReductionController.get_default_calibration_dir(),
                                      'HB2B_Latest.json')

    if os.path.exists('/HFIR/HB2B/shared'):
        assert os.path.exists(default_calib_file)
        print(default_calib_file)
        print(default_calib_file.lower())
        assert default_calib_file.lower().endswith('.json'), 'Must be a JSON file'
    else:
        pytest.skip('Unable to access HB2B archive')


@pytest.mark.parametrize('nexus_file, calibration_file, mask_file, gold_file',
                         [('/HFIR/HB2B/IPTS-22731/nexus/HB2B_1017.nxs.h5', None, None,
                           'data/HB2B_1017_NoMask_Gold.h5'),
                          ('/HFIR/HB2B/IPTS-22731/nexus/HB2B_1017.nxs.h5', None,
                           'data/HB2B_Mask_12-18-19.xml', 'data/HB2B_1017_NoMask_Gold.h5'),
                          ('/HFIR/HB2B/IPTS-22731/nexus/HB2B_1017.nxs.h5', 'data/HB2B_CAL_Si333.json',
                           'data/HB2B_Mask_12-18-19.xml', 'data/HB2B_1017_NoMask_Gold.h5')],
                         ids=('HB2B_1017_NoCal_NoMask', 'HB2B_1017_NoCal_Mask', 'HB2B_1017_Cal_Mask'))
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

    # reduce data
    test_ws = reduce_hidra_workflow(nexus_file, output_dir, progressbar=None,
                                    calibration=calibration_file, mask=mask_file,
                                    project_file_name=target_file_path)

    # get sub run 2
    sub_run_2_pattern = test_ws.get_reduced_diffraction_data(2, mask_id=None)
    write_gold_file('Gold_{}_Cal{}.h5'.format(mask_file is not None,
                                              calibration_file is not None), {'sub run 2': sub_run_2_pattern})

    # Check whether the target file generated
    assert os.path.exists(target_file_path), 'Hidra project file {} is not generated'.format(target_file_path)

    # using gold file to compare the result
    parse_gold_file(gold_file)

    # delete
    os.remove(target_file_path)

    return


def test_load_split():
    """Test method to load, split, convert to powder pattern and save

    Returns
    -------

    """
    # Init load/split service instance
    nexus_file = '/HFIR/HB2B/IPTS-22731/nexus/HB2B_1017.nxs.h5'
    if os.path.exists(nexus_file) is False:
        pytest.skip('Unable to access {}'.format(nexus_file))

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
    assert np.sum(sub_run_1_counts) > 500000, 'Sub run 1 counts = {} is too small'.format(sub_run_1_counts.sum())

    # Get diffraction pattern
    vec_2theta, vec_intensity = controller.get_powder_pattern(2)
    assert 78 < vec_2theta.mean() < 82, '2theta range ({}, {}) shall be centered around 80 for sub run 2.' \
                                        ''.format(vec_2theta[0], vec_2theta[-1])

    # from matplotlib import pyplot as plt
    # plt.plot(vec_2theta, vec_intensity)
    # plt.show()

    assert vec_intensity[~np.isnan(vec_intensity)].max() > 2,\
        'Max intensity {} must larger than 2'.format(vec_intensity[~np.isnan(vec_intensity)].max())

    # Sample logs
    assert abs(controller.get_sample_log_value('2theta', 1) - 69.99525) < 1E-5
    assert controller.get_sample_log_value('2theta', 2) == 80.0
    assert abs(controller.get_sample_log_value('2theta', 3) - 97.50225) < 1E-5


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


def write_gold_file(file_name, data):
    """Write value to gold file (format)

    Parameters
    ----------
    file_name : str
        output file
    data : ~tuple or ~dict
        numpy array data or dictionary of data
    Returns
    -------

    """
    gold_file = h5py.File(file_name, 'w')

    if isinstance(data, np.ndarray):
        dataset = {'data': data}
    else:
        dataset = data

    for data_name in dataset:
        if isinstance(dataset[data_name], tuple):
            # write (x, y)
            group = gold_file.create_group(data_name)
            group.create_dataset('x', data=dataset[data_name][0])
            group.create_dataset('y', data=dataset[data_name][1])
        else:
            # write value directly
            gold_file.create_dataset(data_name, data=dataset[data_name])

    gold_file.close()
