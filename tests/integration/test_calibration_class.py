# Migrated from /HFIR/HB2B/shared/Quick_Calibration.py
# Original can be found at ./Quick_Calibration_v3.py
# Renamed from  ./prototypes/calibration/Quick_Calibration_Class.py
# print ('Prototype Calibration: Quick_Calibration_v4')

import pytest
import time
import os
import filecmp
from pyrs.calibration import peakfit_calibration
from pyrs.utilities import calibration_file_io
from pyrs.utilities import rs_project_file


def test_main():
    """Main test for the script

    Returns
    -------

    """
    # Set up
    # reduction engine
    project_file_name = 'tests/data/HB2B_000.hdf5'
    engine = rs_project_file.HydraProjectFile(project_file_name, mode=rs_project_file.HydraProjectFileMode.READONLY)
    
    # instrument geometry
    idf_name = 'tests/data/XRay_Definition_1K.txt'

    t_start = time.time()

    # instrument
    hb2b = calibration_file_io.import_instrument_setup(idf_name)

    calibrator = peakfit_calibration.PeakFitCalibration(hb2b, engine)
    calibrator.calibrate_wave_length()

    # write out
    file_name = os.path.join(os.getcwd(), 'HB2B_CAL_Test.json')
    calibrator.write_calibration()

    t_stop = time.time()
    print('Total Time: {}'.format(t_stop - t_start))

    # Compare output file with gold file for test
    if filecmp.cmp('data/HB2B_CAL_Si333.json', file_name):
        os.remove(file_name)
    else:
        assert False, 'Test output {} is different from gold file {}'.format(file_name, 'data/HB2B_CAL_Si333.json')

    return


if __name__ == '__main__':
    pytest.main()
