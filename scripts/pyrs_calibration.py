#!/usr/bin/python
import time
import os
from pyrs.calibration import mantid_peakfit_calibration
from pyrs.utilities import get_nexus_file  # type: ignore


# Define Default Material
_Materials = {}
_Materials['ffe'] = 2.8663982
_Materials['mo'] = 3.14719963
_Materials['ni'] = 3.523799438
_Materials['ge'] = 5.657568976977
_Materials['si'] = 5.4307

# Define strings to help the user define the json input
_options = 'Required and optional entry for json input:\n'
_options += 'REQUIRED\n'
_options += 'must define "powder scan"\n'
_options += 'Optional\n'
_options += '"IPTS" for the proposal with powder or pin data (default is 22731)\n'
_options += '"powder scan": run number for measurment with powder line data\n'
_options += '"mask": default detector mask\n'
_options += '"old calibration": override default settings with an old calibration file\n'
_options += '"method": method that will be used to refine instrument geometry (default is shifts+rotations)\n'
_options += '"cycle": HFIR run cycle\n'

M_options = 'Options for Method inputs:\n'
M_options += '"full": refine x, y, z shifts+rotations and wavelength\n'
M_options += '"geometry": refine x, y, z shifts+rotations\n'
M_options += '"shifts": refine x, y, z shifts\n'
M_options += '"shift x": refine x shift\n'
M_options += '"shift y": refine y shift\n'
M_options += '"distance": refine z shift\n'
M_options += '"rotations": refine x, y, z rotations\n'
M_options += '"wavelength": refine only wavelength\n'
M_options += '"shifts+rotations+...": refine shifts then rotations the wavelength (not order or repeat restrictions)'

# DEFAULT VALUES FOR DATA PROCESSING
INSTRUMENT_CALIBRATION = None
DATA_MASK = None
VAN_RUN = None
POWDER_RUN = None
IPTS_ = 22731
REFINE_METHOD = 'full'
SAVE_CALIB = True
WRITE_LATEST = False
ETA_Slice = 3.
TTH_Bins = 512
nexus_file = None
KEEP_SUB_RUNS = None
CalibName = None
HFIR_CYCLE = None

# Allow a varriety of inputs to catch errors
powerinput = ['powder scan', 'powder', 'powder_scan', 'powderscan', 'powder scans', 'powder_scans', 'powderscans']
calinput = ['calibration', 'old calibration', 'old_calibration']
methodinput = ['method', 'methods']
maskinput = ['mask', 'default mask']
exportinput = ['save', 'export', 'save calibration', 'save_calibration']
savelatest = ['write latest', 'write_latest', 'latest']
output_input = ['output', 'calib file']
nexus_input = ['nexus', 'nexus file', 'nexus run', 'nexus_file']
keep_input = ['keep', 'keep list']

# Defualt check for method input
method_options = ["full", "geometry", "shifts", "shift x", "shift_x", "shift y", "shift_y",
                  "distance", "rotations", "wavelength"]

calib_folder_base = '/HFIR/HB2B/shared/CALIBRATION'


def _get_nexus_data(nexus_run):

    try:
        int(nexus_run)
        return get_nexus_file(nexus_run)
    except ValueError:
        return nexus_run


def _run_calibration(calibrator, calib_method):
    """
    """

    if calib_method == "full":
        calibrator.singlepeak = False
        calibrator.FullCalibration()
    elif calib_method == "geometry":
        calibrator.singlepeak = False
        calibrator.CalibrateGeometry()
    elif calib_method == "shifts":
        calibrator.singlepeak = False
        calibrator.CalibrateShift()
    elif calib_method in ["shift x", "shift_x"]:
        calibrator.singlepeak = True
        calibrator.calibrate_shiftx()
    elif calib_method in ["shift y", "shift_y"]:
        calibrator.singlepeak = False
        calibrator.calibrate_shifty()
    elif calib_method == "distance":
        calibrator.singlepeak = False
        calibrator.calibrate_distance()
    elif calib_method == "rotations":
        calibrator.singlepeak = False
        calibrator.CalibrateRotation()
    elif calib_method == "wavelength":
        calibrator.singlepeak = True
        calibrator.calibrate_wave_length()
    else:
        raise RuntimeError('{} is not a valid calibration method\n{}'.format(calib_method, M_options))

    calibrator.print_calibration(print_to_screen=False, refine_step=calib_method)

    return calibrator


def _write_template():
    with open('template.json', 'w') as out:
        out.write('{\n')
        out.write('\t"IPTS": 22731,\n')
        out.write('\t"Powder scan": 1090,\n')
        out.write('\t"Pin scan": 1086,\n')
        out.write('\t"Method": "full",\n')
        out.write('}\n')


def check_method_input(REFINE_METHOD, SPLITTER):
    for calib_method in REFINE_METHOD.split(SPLITTER):
        if calib_method not in method_options:
            raise RuntimeError('\n{} is not a valid calibration method\n\n{}'.format(calib_method, M_options))

    return


def main():
    import sys
    import json

    if len(sys.argv) == 1:
        print('Requires a json input.\n{}\n{}'.format(_options, M_options))
        _write_template()
        raise RuntimeError('template.json was created as an example input')

    with open(sys.argv[1], 'r') as json_input:
        try:
            calibration_inputs = json.load(json_input)
        except ValueError:
            print('Formating error in json input file.\n{}\n{}'.format(_options, M_options))
            _write_template()
            raise RuntimeError('template.json was created as an example input')

    for key in list(calibration_inputs.keys()):
        if key.lower() in powerinput:
            POWDER_RUN = calibration_inputs[key]
        elif key.lower() == 'ipts':
            IPTS_ = calibration_inputs[key]
        elif key.lower() == 'cycle':
            HFIR_CYCLE = calibration_inputs[key]
        elif key.lower() == 'method':
            REFINE_METHOD = calibration_inputs[key]
        elif key.lower() in calinput:
            INSTRUMENT_CALIBRATION = calibration_inputs[key]
        elif key.lower() in maskinput:
            DATA_MASK = calibration_inputs[key]
        elif key.lower() in exportinput:
            SAVE_CALIB = bool(str(calibration_inputs[key]).lower() == 'true')
        elif key.lower() in savelatest:
            WRITE_LATEST = bool(str(calibration_inputs[key]).lower() == 'true')
        elif key.lower() in output_input:
            CalibName = calibration_inputs[key]
        elif key.lower() in nexus_input:
            nexus_file = calibration_inputs[key]
        elif key.lower() in keep_input:
            KEEP_SUB_RUNS = calibration_inputs[key]

    if '+' in REFINE_METHOD:
        SPLITTER = '+'
    elif ',' in REFINE_METHOD:
        SPLITTER = ','
    else:
        SPLITTER = ' '

    check_method_input(REFINE_METHOD, SPLITTER)

    if nexus_file is None:
        nexus_file = _get_nexus_data(POWDER_RUN)

    calibrator = mantid_peakfit_calibration.FitCalibration(nexus_file=nexus_file, eta_slice=ETA_Slice, bins=TTH_Bins,
                                                           mask_file=DATA_MASK, vanadium=VAN_RUN)

    if KEEP_SUB_RUNS is not None:
        calibrator.set_keep_subrun_list(KEEP_SUB_RUNS)

    if INSTRUMENT_CALIBRATION is not None:
        calibrator.get_archived_calibration(INSTRUMENT_CALIBRATION)

    for calib_method in REFINE_METHOD.split(SPLITTER):
        calibrator = _run_calibration(calibrator, calib_method)

    if CalibName is not None:
        calibrator.write_calibration(CalibName)
    elif SAVE_CALIB:
        datatime = time.strftime('%Y-%m-%dT%H-%M', time.localtime())
        if HFIR_CYCLE is not None:
            FolderName = '{}/cycle{}'.format(calib_folder_base, HFIR_CYCLE)
            if not os.path.exists(FolderName):
                os.makedirs(FolderName)
            CalibName = '{}/cycle{}/HB2B_{}_{}.json'.format(calib_folder_base, HFIR_CYCLE,
                                                            calibrator.monosetting, datatime)
            calibrator.write_calibration(CalibName)

        CalibName = '{}/HB2B_{}_{}.json'.format(calib_folder_base, calibrator.monosetting, datatime)
        calibrator.write_calibration(CalibName)

        if WRITE_LATEST:
            CalibName = '{}/HB2B_Latest.json'.format(calib_folder_base)
            calibrator.write_calibration(CalibName)

        print(calibrator.refinement_summary)
    else:
        calibrator.print_calibration()
        print(calibrator.refinement_summary)

if __name__ == '__main__':
    main()
    