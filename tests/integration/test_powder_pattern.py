"""
Integration test for PyRS to calculate powder pattern from detector counts
"""
from pyrs.projectfile import HidraProjectFile
from pyrs.core.nexus_conversion import NeXusConvertingApp
from pyrs.utilities import calibration_file_io
from pyrs.core import workspaces
import numpy as np
from pyrs.split_sub_runs.load_split_sub_runs import NexusProcessor
from pyrs.core.powder_pattern import ReductionApp
import os
import h5py
from pyrs.core.workspaces import HidraWorkspace
from pyrs.core.reduce_hb2b_pyrs import PyHB2BReduction, ResidualStressInstrument
from pyrs.core.instrument_geometry import AnglerCameraDetectorGeometry
import pytest


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
    data : ~numpy.ndarray or dict
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
        gold_file.create_dataset(data_name, data=dataset[data_name])

    gold_file.close()


def test_2theta_calculation():
    """Test the calculation of 2theta position

    Returns
    -------

    """
    # Create geometry setup
    test_setup = AnglerCameraDetectorGeometry(1024, 1024, 0.0003, 0.0003, 0.985, False)
    # Create instrument
    instrument = ResidualStressInstrument(test_setup)

    # Set 2theta
    instrument.build_instrument(two_theta=85., l2=None, instrument_calibration=None)

    # Calculate pixel positions
    pixel_positions = instrument.get_pixel_array()

    # Calculate 2theta values
    two_theta_arrays = instrument.get_pixels_2theta(dimension=1)

    # compare with gold file
    gold_pixel_positions = parse_gold_file('???')
    np.testing.assert_allclose(pixel_positions, gold_pixel_positions, rtol=1E-8)

    gold_2theta_values = parse_gold_file('???')
    np.testing.assert_allclose(two_theta_arrays, gold_2theta_values, rtol=1E-8)


@pytest.mark.parametrize('project_file_name, mask_file_name',
                         [('data/HB2B_1017.h5', 'data/HB2B_Mask_12-18-19.xml', 'data/HB2B_1017_Masked_Gold.h5'),
                          ('data/HB2B_1017.h5', None, 'data/HB2B_1017_NoMask_Gold.h5')],
                         ids=('HB2B_1017_Masked', 'HB2B_1017_NoMask'))
def test_powder_pattern_service(project_file_name, mask_file_name, gold_file):
    """Test the powder pattern calculator (service) with HB2B-specific reduction routine

    Parameters
    ----------
    project_file_name
    mask_file_name

    Returns
    -------

    """
    # Parse input file
    test_ws = HidraWorkspace('test_powder_pattern')
    test_ws.load_hidra_project(project_file_name, load_raw_counts=True, load_reduced_diffraction=False)

    # Create geometry setup
    test_setup = AnglerCameraDetectorGeometry(1024, 1024, 0.0003, 0.0003, 0.985, False)
    # Create instrument
    instrument = ResidualStressInstrument(test_setup)
    pyrs_service = PyHB2BReduction(instrument, None)

    # Sub runs
    sub_runs = test_ws.get_sub_runs()

    # Import gold file
    gold_pattern = parse_gold_file(gold_file)

    for index, sub_run_i in enumerate(sub_runs):
        two_theta_i = test_ws.get_sample_log_value('2theta', sub_run_i)
        counts_i = test_ws.get_detector_counts(sub_run_i)
        pyrs_service.set_experimental_data(two_theta=two_theta_i, l2=None, raw_count_vec=counts_i)

        # Get gold data of pattern (i).  Note that vector 2theta are bin boundaries!
        gold_data_i = gold_pattern[str(sub_run_i)]

        # Reduce
        pattern = pyrs_service.reduce_to_2theta_histogram(two_theta_bins=gold_data_i[0],
                                                          mask_array=None,
                                                          is_point_data=False,
                                                          vanadium_counts_array=None)

        # Verify
        np.testing.assert_allclose(pattern[1], gold_data_i[1], rtol=1E-8)

    return
