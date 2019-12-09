# Migrated from /HFIR/HB2B/shared/Quick_Calibration.py
# Original can be found at ./Quick_Calibration_v3.py
# Renamed from  ./prototypes/calibration/Quick_Calibration_Class.py
import os
import numpy
from pyrs.utilities import calibration_file_io
from pyrs.utilities.rs_project_file import HidraProjectFile, HidraProjectFileMode
try:
    from pyrs.calibration import peakfit_calibration
except ImportError as e:
    peakfit_calibration = str(e)  # import failed exception explains why

# DEFAULT VALUES FOR DATA PROCESSING
DEFAULT_CALIBRATION = None
DEFAULT_INSTRUMENT = None
DEFAULT_MASK = None


def SurveyLandscape(calibrator, fName, i_1, i_2, ll=[], ul=[], steps=20):
    import multiprocessing
    import time

    def GetError(i_1, i_2, val1, val2, return_dict, procnum):
        calibrator._calib[i_1] = val1
        calibrator._calib[i_2] = val2
        res = calibrator.singleEval(ReturnScalar=True)
        return_dict[procnum] = [res, val1, val2]
        return

    step1 = (ul[0]-ll[0])/(steps-1)
    step2 = (ul[1]-ll[1])/(steps-1)

    pipe_list = []
    jobs = []

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    index = 1

    for val1 in numpy.arange(ll[0], ul[0]+step1/10, step1):
        for val2 in numpy.arange(ll[1], ul[1]+step2/10, step2):
            recv_end, send_end = multiprocessing.Pipe(False)
            p = multiprocessing.Process(target=GetError, args=(i_1, i_2, val1, val2, return_dict, index))
            jobs.append(p)
            pipe_list.append(recv_end)
            p.start()
            index += 1
            time.sleep(3)

    for proc in jobs:
        proc.join()

    with open('ParamSurvey_{}_{}_{}steps.csv'.format(i_1, i_2, steps), 'w') as fOUT:
        fOUT.write('# key_{},key_{},res\n'.format(1, 2))
        keys = sorted(return_dict.keys())
        for i_key in keys:
            res, val1, val2 = return_dict[i_key]
            fOUT.write('{},{},{}\n'.format(val1, val2, res))

    return


def SaveCalibError(calibrator, fName):
    calibrator.singleEval(ConstrainPosition=True, start=1, stop=0)

    tths = sorted(list(calibrator.ReductionResults.keys()))

    Rois = list(calibrator.ReductionResults[tths[0]].keys())
    DataPoints = len(calibrator.ReductionResults[tths[0]][Rois[0]][0])

    DataOut = numpy.zeros((DataPoints, 3*len(tths)*len(Rois)))
    header = ''

    lcv = -1
    for i_tth in tths:
        for j in list(calibrator.ReductionResults[i_tth].keys()):
            tempdata = calibrator.ReductionResults[i_tth][j]

            lcv += 1
            DataOut[:, lcv*3+0] = tempdata[0]
            DataOut[:, lcv*3+1] = tempdata[1]
            DataOut[:, lcv*3+2] = tempdata[2]
            header += ',pos{}_roi{}_tth,pos{}_roi{}_obs,pos{}_roi{}_calc'.format(i_tth, j[0],
                                                                                 i_tth, j[0], i_tth, j[0])

    DataOut = DataOut[:, :lcv*3+3]

    numpy.savetxt(fName, DataOut, delimiter=',', header=header[1:])


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
    calibrator = peakfit_calibration.PeakFitCalibration(hb2b, engine)

    calibrator._calib[0] = 0.0025
    calibrator._calib[2] = -0.02281116
#    SaveCalibError(calibrator, 'HB2B_{}_before.txt'.format(options.run))

    if options.calibration in ['testshift']:
        calibrator = peakfit_calibration.PeakFitCalibration(hb2b, engine)
        calibrator._calib[2] = -0.02281116
        calibrator._calib[0] = 0.000
        SaveCalibError(calibrator, 'HB2B_{}_shift1.txt'.format(options.run))
        calibrator = peakfit_calibration.PeakFitCalibration(hb2b, engine)
        calibrator._calib[2] = -0.02281116
        calibrator._calib[0] = 0.00125
        SaveCalibError(calibrator, 'HB2B_{}_shift2.txt'.format(options.run))
        calibrator = peakfit_calibration.PeakFitCalibration(hb2b, engine)
        calibrator._calib[2] = -0.02281116
        calibrator._calib[0] = 0.0025
        SaveCalibError(calibrator, 'HB2B_{}_shift3.txt'.format(options.run))

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
        fName = 'HB2B_{}_LSQ_{}.json'.format(options.run, calibrator.Method)
        file_name = os.path.join(os.getcwd(), fName)
        calibrator.write_calibration(file_name)

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

    if options.calibration == 'TEST':

        # generate project name if not already determined
        project_file_name = '{}/HB2B_{}.h5'.format(options.projectdir, 1086)
        engine = HidraProjectFile(project_file_name, mode=HidraProjectFileMode.READONLY)

        # instrument geometry
        if options.instrument == DEFAULT_INSTRUMENT:
            idf_name = 'data/XRay_Definition_1K.txt'
        else:
            idf_name = options.instrument

        hb2b = calibration_file_io.import_instrument_setup(idf_name)
        calibrator = peakfit_calibration.PeakFitCalibration(hb2b, engine)

        i_tth = 8
        print calibrator._engine.read_log_value(calibrator.tth_ref).value
        print calibrator._engine.read_log_value('2theta').value

        vals = numpy.concatenate([calibrator._engine.read_log_value('2theta').value,
                                  calibrator._engine.read_log_value(calibrator.tth_ref).value])
        vals = vals.reshape((2, calibrator._engine.read_log_value('2theta').value.shape[0])).T

        numpy.savetxt('HB2B_1086_tths.dat', vals, delimiter=',')

        Data1 = calibrator._engine.read_raw_counts(i_tth+1).reshape((1024, 1024))

        # generate project name if not already determined
        project_file_name = '{}/HB2B_{}.h5'.format(options.projectdir, 1087)
        engine = HidraProjectFile(project_file_name, mode=HidraProjectFileMode.READONLY)

        # instrument geometry
        if options.instrument == DEFAULT_INSTRUMENT:
            idf_name = 'data/XRay_Definition_1K.txt'
        else:
            idf_name = options.instrument

        hb2b = calibration_file_io.import_instrument_setup(idf_name)
        calibrator = peakfit_calibration.PeakFitCalibration(hb2b, engine)

        i_tth = 16
        print calibrator._engine.read_log_value(calibrator.tth_ref).value[i_tth]
        print calibrator._engine.read_log_value('2theta').value[i_tth]

        Data2 = calibrator._engine.read_raw_counts(i_tth+1).reshape((1024, 1024))

        numpy.savetxt('HB2B_1086_DATA.dat', Data1, delimiter=',')
        numpy.savetxt('HB2B_1087_DATA.dat', Data2, delimiter=',')

    if options.calibration == 'survey':
        print 'Calibrating Geometry in Steps'
        calibrator.singlepeak = False

        SurveyLandscape(calibrator, '', 0, 6, ll=[-0.025, 1.53], ul=[0.025, 1.55], steps=21)

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
