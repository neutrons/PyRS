# Migrated from /HFIR/HB2B/shared/Quick_Calibration.py
# Original can be found at ./Quick_Calibration_v3.py
# Renamed from  ./prototypes/calibration/Quick_Calibration_Class.py
import os
import numpy
from pyrs.utilities import calibration_file_io
from pyrs.utilities.rs_project_file import HidraProjectFile, HidraProjectFileMode
from pyrs.calibration import peakfit_calibration

# DEFAULT VALUES FOR DATA PROCESSING
DEFAULT_CALIBRATION = None
DEFAULT_INSTRUMENT = None
DEFAULT_MASK = None

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Script for auto-reducing HB2B')
    parser.add_argument('run', help='name of nexus file')
    parser.add_argument('projectdir', nargs='?', help='Path to output directory')
    parser.add_argument('--instrument', nargs='?', default=DEFAULT_INSTRUMENT,
                        help='instrument configuration file overriding embedded (arm, pixel number'
                        ' and size) (default=%(default)s)')
    parser.add_argument('--calibration', nargs='?', default=DEFAULT_CALIBRATION,
                        help='instrument geometry calibration file overriding embedded (default=%(default)s)')
    parser.add_argument('--mask', nargs='?', default=DEFAULT_MASK,
                        help='masking file (PyRS hdf5 format) or mask name (default=%(default)s)')

    options = parser.parse_args()

    # generate project name if not already determined
    project_file_name = '{}/HB2B_{}.h5'.format(options.projectdir, options.run)
    engine = HidraProjectFile(project_file_name, mode=HidraProjectFileMode.READONLY)

    # instrument geometry
    if options.instrument == DEFAULT_INSTRUMENT:
        idf_name = 'data/XRay_Definition_1K.txt'
    else:
        idf_name = options.instrument

    hb2b = calibration_file_io.import_instrument_setup(idf_name)
    calibrator = peakfit_calibration.PeakFitCalibration(hb2b, engine, scheme=0)

    calibrator._calib[0] = 0.0025
    calibrator._calib[2] = -0.02281116

    if options.calibration in [DEFAULT_CALIBRATION, 'geometry']:
        SaveCalibError(calibrator, 'HB2B_{}_before.csv'.format(options.run))
        calibrator.CalibrateGeometry()
        print calibrator.get_calib()
        SaveCalibError(calibrator, 'HB2B_{}_calibres.csv'.format(options.run))

    if options.calibration in ['shift']:
        calibrator.singlepeak = False
        calibrator.CalibrateShift(ConstrainPosition=True)
        print calibrator.get_calib()

    if options.calibration in ['rotate']:
        calibrator.CalibrateRotation(ConstrainPosition=True)
        print calibrator.get_calib()

    if options.calibration in ['wavelength']:
        calibrator.singlepeak = False
        calibrator.calibrate_wave_length()
        print calibrator.get_calib()

    if options.calibration in ['full']:
        calibrator.singlepeak = False
        calibrator.FullCalibration(ConstrainPosition=True)
        print calibrator.get_calib()

    if options.calibration == 'distance':
        calibrator.calibrate_distance(ConstrainPosition=True, Brute=False)
        print calibrator.get_calib()

    if options.calibration == 'geoNew':
        print 'Calibrating Geometry in Steps'
        calibrator.calibrate_shiftx(ConstrainPosition=True)
        print calibrator.get_calib()
        calibrator.calibrate_distance(ConstrainPosition=False)
        print calibrator.get_calib()
        calibrator.calibrate_wave_length(ConstrainPosition=True)
        print calibrator.get_calib()
        calibrator.CalibrateRotation()
        print calibrator.get_calib()
        calibrator.FullCalibration()
        print calibrator.get_calib()

    if options.calibration in [DEFAULT_CALIBRATION, 'runAll']:
        calibrator = peakfit_calibration.PeakFitCalibration(hb2b, engine)
        calibrator.FullCalibration()
        FullCalib = calibrator.get_calib()

        calibrator = peakfit_calibration.PeakFitCalibration(hb2b, engine)
        calibrator.CalibrateGeometry()
        GeoCalib = calibrator.get_calib()

        calibrator = peakfit_calibration.PeakFitCalibration(hb2b, engine)
        calibrator.CalibrateRotation()
        RotateCalib = calibrator.get_calib()

        calibrator = peakfit_calibration.PeakFitCalibration(hb2b, engine)
        calibrator.CalibrateShift()
        ShiftCalib = calibrator.get_calib()

        calibrator = peakfit_calibration.PeakFitCalibration(hb2b, engine)
        calibrator.calibrate_wave_length()
        LambdaCalib = calibrator.get_calib()

        print FullCalib
        print GeoCalib
        print RotateCalib
        print ShiftCalib
        print LambdaCalib

    fName = 'HB2B_{}_LSQ_{}_Method_{}.json'.format(options.run, calibrator.Method, options.calibration)
    file_name = os.path.join(os.getcwd(), fName)
    calibrator.write_calibration(file_name)
