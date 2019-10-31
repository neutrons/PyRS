# Migrated from /HFIR/HB2B/shared/Quick_Calibration.py
# Original can be found at ./Quick_Calibration_v3.py
# Renamed from  ./prototypes/calibration/Quick_Calibration_Class.py
# print ('Prototype Calibration: Quick_Calibration_v4')

import pytest
import time
import os
import filecmp
import json
try:
    from pyrs.calibration import peakfit_calibration
    from pyrs.utilities import calibration_file_io
    from pyrs.utilities import rs_project_file
except ImportError as e:
    peakfit_calibration = str(e)  # import failed exception explains why


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


@pytest.mark.skipif(isinstance(peakfit_calibration, str), reason=peakfit_calibration)
def test_main():
    """Main test for the script

    Returns
    -------

    """
    # Set up
    # reduction engine
    project_file_name = 'data/HB2B_000.h5'
    engine = rs_project_file.HydraProjectFile(project_file_name, mode=rs_project_file.HydraProjectFileMode.READONLY)

    # instrument geometry
    idf_name = 'data/XRay_Definition_1K.txt'

    t_start = time.time()

    # instrument
    hb2b = calibration_file_io.import_instrument_setup(idf_name)

    calibrator = peakfit_calibration.PeakFitCalibration(hb2b, engine)
    calibrator.calibrate_wave_length()

    # write out
    if os.path.exists('HB2B_CAL_Test.json'):
        os.remove('HB2B_CAL_Test.json')
    file_name = os.path.join(os.getcwd(), 'HB2B_CAL_Test.json')
    calibrator.write_calibration(file_name)

    t_stop = time.time()
    print('Total Time: {}'.format(t_stop - t_start))

    # Compare output file with gold file for test
    if filecmp.cmp('data/HB2B_CAL_Si333.json', file_name):
        os.remove(file_name)
    else:
        print_out_json_diff('data/HB2B_CAL_Si333.json', 'HB2B_CAL_Test.json')
        assert False, 'Test output {} is different from gold file {}'.format(file_name, 'data/HB2B_CAL_Si333.json')

    return


if __name__ == '__main__':
    pytest.main([__file__])
