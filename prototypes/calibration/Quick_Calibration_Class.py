# Migrated from /HFIR/HB2B/shared/Quick_Calibration.py
# Original can be found at ./Quick_Calibration_v3.py
print ('Prototype Calibration: Quick_Calibration_v4')
import numpy as np
import time

from pyrs.calibration import peakfit_calibration
from pyrs.core import reduce_hb2b_pyrs
from pyrs.core import instrument_geometry
from pyrs.core import calibration_file_io
from pyrs.utilities import rs_project_file



def main():
    # Set up
    # reduction engine
    project_file_name = 'tests/testdata/HB2B_000000.hdf5' 
    engine = rs_project_file.HydraProjectFile(project_file_name,
                                                           mode=rs_project_file.HydraProjectFileMode.READONLY)
    
    # instrument geometry
    idf_name = 'tests/testdata/xray_data/XRay_Definition_1K.txt'      

    t_start = time.time()

    # instrument
    instrument  = calibration_file_io.import_instrument_setup(idf_name)

    HB2B        = instrument_geometry.AnglerCameraDetectorGeometry( instrument.detector_rows, instrument.detector_columns, instrument.pixel_size_x, instrument.pixel_size_y, instrument.arm_length, False )

    CalibrationObject = peakfit_calibration.PeakFitCalibration( HB2B, engine )
    CalibrationObject.FullCalibration() 
    CalibrationObject.write_calibration()

    t_stop = time.time()
    print ('Total Time: {}'.format(t_stop - t_start))


    return

main()
