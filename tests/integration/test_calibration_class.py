# Migrated from /HFIR/HB2B/shared/Quick_Calibration.py
# Original can be found at ./Quick_Calibration_v3.py
# Renamed from  ./prototypes/calibration/Quick_Calibration_Class.py
import pytest
import time
import os
import json
from pyrs.utilities import calibration_file_io
from pyrs.utilities.rs_project_file import HydraProjectFile, HidraProjectFileMode
try:
    from pyrs.calibration import peakfit_calibration
except ImportError as e:
    peakfit_calibration = str(e)  # import failed exception explains why


def are_equivalent_jsons(test_json_name, gold_json_name, atol):
    """Print out the difference of two JSON files

    Parameters
    ----------
    test_json_name
    gold_json_name
    atol : float
        absolute tolerance

    Returns
    -------
    boolean
        equivalent or not

    """
    # Load file 1
    with open(test_json_name, 'r') as json1:
        test_json_dict = json.load(json1)

    # Load file 2
    with open(gold_json_name, 'r') as json2:
        gold_json_dict = json.load(json2)

    # Compare
    diff = False
    for key in gold_json_dict.keys():
        # expect a file with same key
        if key not in test_json_dict:
            diff = True
            break

        # compare value
        gold_value = gold_json_dict[key]
        test_value = test_json_dict[key]
        if isinstance(gold_value, float):
            # float: check with tolerance
            diff = abs(gold_value - test_value) > atol
        else:
            # int or str: must be exactly same
            diff = gold_value != test_value

        if diff:
            # quit loop if not same
            break
    # END-FOR

    return not diff


def print_out_json_diff(json_file1_name, json_file2_name):
    """Print out the difference of two JSON files

    Parameters
    ----------
    json_file1_name
    json_file2_name

    Returns
    -------

    """
    # Load file 1
    with open(json_file1_name, 'r') as json1:
        json_dict1 = json.load(json1)

    # Load file 2
    with open(json_file2_name, 'r') as json2:
        json_dict2 = json.load(json2)

    # Output difference
    if set(json_dict1.keys()) != set(json_dict2.keys()):
        # Compare keys
        print('[JSON Keys are different]\n{}: {}\n{}: {}'
              ''.format(json_file1_name, sorted(json_dict1.keys()),
                        json_file2_name, sorted(json_dict2.keys())))

    else:
        # Compare values
        keys = sorted(json_dict1.keys())
        print('[JSON Value are different]\nField\t{}\t{}'.format(json_file1_name, json_file2_name))
        for k in keys:
            print('{}\t{}\t{}'.format(k, json_dict1[k], json_dict2[k]))

    return


# On analysis cluster, it will fail due to lmfit is not supported
@pytest.mark.skipif(isinstance(peakfit_calibration, str), reason=peakfit_calibration)
def test_main():
    """Main test for the script

    Returns
    -------

    """
    # Set up
    # reduction engine
    project_file_name = 'data/HB2B_000.h5'
    engine = HydraProjectFile(project_file_name, mode=HidraProjectFileMode.READONLY)

    # instrument geometry
    idf_name = 'data/XRay_Definition_1K.txt'

    t_start = time.time()

    # instrument
    hb2b = calibration_file_io.import_instrument_setup(idf_name)

    calibrator = peakfit_calibration.PeakFitCalibration(hb2b, engine)
    calibrator.calibrate_wave_length()
    calibrator._caliberr[6] = 1e-4
    calibrator._calibstatus = 3

    # write out
    if os.path.exists('HB2B_CAL_Test.json'):
        os.remove('HB2B_CAL_Test.json')
    file_name = os.path.join(os.getcwd(), 'HB2B_CAL_Test.json')
    calibrator.write_calibration(file_name)

    t_stop = time.time()
    print('Total Time: {}'.format(t_stop - t_start))

    # Compare output file with gold file for test
    if are_equivalent_jsons('data/HB2B_CAL_Si333.json', file_name, atol=5E-3):
        # Same: remove file generated in test
        os.remove(file_name)
    else:
        print_out_json_diff('data/HB2B_CAL_Si333.json', 'HB2B_CAL_Test.json')
        assert False, 'Test output {} is different from gold file {}'.format(file_name, 'data/HB2B_CAL_Si333.json')

    return


if __name__ == '__main__':
    pytest.main([__file__])
