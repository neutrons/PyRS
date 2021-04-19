# type: ignore
import numpy as np
import time
import os
from pyrs.calibration import correlation_calibration as peakfit_calibration
from pyrs.core.nexus_conversion import NeXusConvertingApp
from pyrs.core import MonoSetting  # type: ignore

# Define Default Material
_Materials = {}
_Materials['fe'] = 2.8663982
_Materials['mo'] = 3.14719963
_Materials['ni'] = 3.523799438
_Materials['ge'] = 5.657568976977
_Materials['si'] = 5.4307

# Define strings to help the user define the json input
_options = 'Required and optional entry for json input:\n'
_options += 'REQUIRED\n'
_options += 'must define either "powder scan" or "pin scan"\n'
_options += 'must define "Powder Lines" if "powder scan" is defined\n'
_options += 'Optional\n'
_options += '"IPTS" for the proposal with powder or pin data (default is 22731)\n'
_options += '"powder scan": run number for measurment with powder line data\n'
_options += '"Powder Lines": list with materials and hkl or dspace for materials used to measure powder scan data\n'
_options += '"pin scan": run number for measurment with pin stepping data\n'
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

P_options = '\nOptions for Powder Lines inputs:\n'
P_options += 'User must input a list of dspace or materials with "_hkl" to define the dspace of the reflection\n'
P_options += 'Default materials are Si, Mo, fFe (ferritic), Ni, and Ge\n'
P_options += 'The input can contain a mixture of float dspace and string "material hkl" inputs\n'
P_options += '["Si 220", "fFe 200", ....]\n'

# DEFAULT VALUES FOR DATA PROCESSING
INSTRUMENT_CALIBRATION = None
DATA_MASK = None
POWDER_RUN = None
IPTS_ = 22731
PIN_RUN = None
HFIR_CYCLE = None
REFINE_METHOD = 'full'
POWDER_LINES = list()
SAVE_CALIB = True

# Allow a varriety of inputs to catch errors
powderlineinput = ['powder lines', 'powder_lines', 'powderlines', 'powder line', 'powder_line', 'powderline']
powerinput = ['powder scan', 'powder', 'powder_scan', 'powderscan', 'powder scans', 'powder_scans', 'powderscans']
pininput = ['pin scan', 'pin', 'pin_scan', 'pinscan', 'pin scans', 'pin_scans', 'pinscans']
calinput = ['calibration', 'old calibration', 'old_calibration']
methodinput = ['method', 'methods']
maskinput = ['mask', 'default mask']
exportinput = ['save', 'export', 'save calibration', 'save_calibration']

# Defualt check for method input
method_options = ["full", "geometry", "shifts", "shift x", "shift_x", "shift y", "shift_y",
                  "distance", "rotations", "wavelength"]

# set default calibration inputs
calibration_inputs = {"POWDER_LINES": None,
                      "POWDER_RUN": None,
                      "PIN_RUN": None,
                      "ipts": None,
                      "HFIR_CYCLE": None,
                      "REFINE_METHOD": 'shifts+distance+rotations+full',
                      "INSTRUMENT_CALIBRATION": None,
                      "DATA_MASK": None,
                      "min_tth": None,
                      "max_tth": None,
                      "min_eta": None,
                      "max_eta": None,
                      "SAVE_CALIB": False,
                      "bins": 512,
                      "min_subrun": 0,
                      "max_subrun": 0,
                      "solver": "least_squares"}


def _load_nexus_data(ipts, nexus_run, mask_file):
    nexus_file = '/HFIR/HB2B/IPTS-{}/nexus/HB2B_{}.nxs.h5'.format(ipts, nexus_run)
    converter = NeXusConvertingApp(nexus_file, mask_file)
    hidra_ws = converter.convert()

    return hidra_ws


def _get_mono_setting(dataset):
    return MonoSetting.getFromRotation(dataset.get_sample_log_value('mrot', 1))


def _run_calibration(calibrator, calib_method, solver):
    """
    """
    if solver.lower() == 'least_squares':
        Brute = False
    elif solver.lower() == 'brute':
        Brute = True
    elif solver.lower() == 'differential evolution':
        Brute = 2
    else:
        Brute = 3

    if calib_method == "full":
        calibrator.singlepeak = False
        calibrator.FullCalibration(ConstrainPosition=True, Brute=Brute)
    elif calib_method == "geometry":
        calibrator.singlepeak = False
        calibrator.CalibrateGeometry(ConstrainPosition=True)
    elif calib_method == "shifts":
        calibrator.singlepeak = False
        calibrator.CalibrateShift(ConstrainPosition=True)
    elif calib_method in ["shift x", "shift_x"]:
        calibrator.singlepeak = True
        calibrator.calibrate_shiftx(ConstrainPosition=True)
    elif calib_method in ["shift y", "shift_y"]:
        calibrator.singlepeak = False
        calibrator.calibrate_shifty(ConstrainPosition=True)
    elif calib_method == "distance":
        calibrator.singlepeak = False
        calibrator.calibrate_distance(ConstrainPosition=True)
    elif calib_method == "rotations":
        calibrator.singlepeak = False
        calibrator.CalibrateRotation(ConstrainPosition=True)
    elif calib_method == "wavelength":
        calibrator.singlepeak = True
        calibrator.calibrate_wave_length(ConstrainPosition=True)
    else:
        raise RuntimeError('{} is not a valid calibration method\n{}'.format(calib_method, M_options))

    calibrator.print_calibration(print_to_screen=False, refine_step=calib_method)

    return calibrator


def _parse_powder_line(json_entry):
    Input_error = False
    dspace = []
    for item in json_entry:

        if isinstance(item, str) or isinstance(item, type(u'a')):
            hkl = item.split(' ')[1]
            h_ = float(hkl[0])
            k_ = float(hkl[1])
            l_ = float(hkl[2])

            try:
                lattice = _Materials[item.split(' ')[0].lower()]
            except KeyError:
                Input_error = True

            dspace.append(lattice / np.sqrt(h_**2 + k_**2 + l_**2))

        elif type(item) == float:
            dspace.append(item)

        if Input_error:
            raise RuntimeError('{} is not in material list {}'.format(item.split(' ')[0].lower(),
                                                                      list(_Materials.keys())))

    return dspace


def _write_template():
    with open('template.json', 'w') as out:
        out.write('{\n')
        out.write('\t"IPTS": 22731,\n')
        out.write('\t"Powder scan": 1090,\n')
        out.write('\t"Pin scan": 1086,\n')
        out.write('\t"Method": "full",\n')
        out.write('\t"Powder Lines": ["fe 200", "Mo 211", "Ni 220", "ffe 211", "Mo 220", "Ni 311", ')
        out.write('"Ni 222", "fFe 220", "Mo 310", "fFe 310"]\n')
        out.write('}\n')


def check_method_input(REFINE_METHOD, SPLITTER):
    for calib_method in REFINE_METHOD.split(SPLITTER):
        if calib_method not in method_options:
            raise RuntimeError('\n{} is not a valid calibration method\n\n{}'.format(calib_method, M_options))

    return


if __name__ == '__main__':
    import sys
    import json

    if len(sys.argv) == 1:
        print('Requires a json input.\n{}\n{}\n{}'.format(_options, M_options, P_options))
        _write_template()
        raise RuntimeError('template.json was created as an example input')

    with open(sys.argv[1], 'r') as json_input:
        try:
            calibration_user_inputs = json.load(json_input)
        except ValueError:
            print('Formating error in json input file.\n{}\n{}\n{}'.format(_options, M_options, P_options))
            _write_template()
            raise RuntimeError('template.json was created as an example input')

    for key in list(calibration_user_inputs.keys()):
        if key.lower() in powderlineinput:
            calibration_inputs['POWDER_LINES'] = _parse_powder_line(calibration_user_inputs[key])
        elif key.lower() in powerinput:
            calibration_inputs['POWDER_RUN'] = calibration_user_inputs[key]
        elif key.lower() in pininput:
            calibration_inputs['PIN_RUN'] = calibration_user_inputs[key]
        elif key.lower() in calinput:
            calibration_inputs['INSTRUMENT_CALIBRATION'] = calibration_user_inputs[key]
        elif key.lower() in maskinput:
            calibration_inputs['DATA_MASK'] = calibration_user_inputs[key]
        elif key.lower() in exportinput:
            calibration_inputs['SAVE_CALIB'] = bool(str(calibration_user_inputs[key]).lower() == 'true')
        else:
            calibration_inputs[key.lower()] = calibration_user_inputs[key]

    if '+' in calibration_inputs['method']:
        SPLITTER = '+'
    elif ',' in calibration_inputs['method']:
        SPLITTER = ','
    else:
        SPLITTER = ' '

    check_method_input(calibration_inputs['method'], SPLITTER)

    if calibration_inputs['POWDER_RUN'] is not None:
        calibration_inputs['POWDER_RUN'] = _load_nexus_data(calibration_inputs['ipts'],
                                                            calibration_inputs['POWDER_RUN'],
                                                            calibration_inputs['DATA_MASK'])
        single_material = False
        mono = _get_mono_setting(calibration_inputs['POWDER_RUN'])

    if calibration_inputs['PIN_RUN'] is not None:
        calibration_inputs['PIN_RUN'] = _load_nexus_data(calibration_inputs['ipts'],
                                                         calibration_inputs['PIN_RUN'],
                                                         calibration_inputs['DATA_MASK'])
        single_material = True
        mono = _get_mono_setting(calibration_inputs['PIN_RUN'])

    calibrator = peakfit_calibration.PeakFitCalibration(powder_engine=calibration_inputs['POWDER_RUN'],
                                                        pin_engine=calibration_inputs['PIN_RUN'],
                                                        powder_lines=calibration_inputs['POWDER_LINES'],
                                                        single_material=single_material)

    calibrator.bins = calibration_inputs['bins']
    calibrator.min_tth = calibration_inputs['min_tth']
    calibrator.max_tth = calibration_inputs['max_tth']
    calibrator.min_eta = calibration_inputs['min_eta']
    calibrator.max_eta = calibration_inputs['max_eta']
    calibrator.min_subrun = calibration_inputs['min_subrun']
    calibrator.max_subrun = calibration_inputs['max_subrun']

    print(calibration_inputs['INSTRUMENT_CALIBRATION'])
    if calibration_inputs['INSTRUMENT_CALIBRATION'] is not None:
        calibrator.get_archived_calibration(calibration_inputs['INSTRUMENT_CALIBRATION'])
        print(calibrator._calib)

    calibration_methods = calibration_inputs['method']
    for calib_method in calibration_methods.split(SPLITTER):
        calibrator = _run_calibration(calibrator, calib_method, calibration_inputs["solver"])

    if calibration_inputs['SAVE_CALIB']:
        datatime = time.strftime('%Y-%m-%dT%H-%M', time.localtime())
        if HFIR_CYCLE is not None:
            FolderName = '/HFIR/HB2B/shared/CALIBRATION/cycle{}'.format(calibration_inputs['cycle'])
            if not os.path.exists(FolderName):
                os.makedirs(FolderName)
            CalibName = '/HFIR/HB2B/shared/CALIBRATION/cycle{}/HB2B_{}_{}.json'.format(HFIR_CYCLE, mono, datatime)
            calibrator.write_calibration(CalibName)

        CalibName = '/HFIR/HB2B/shared/CALIBRATION/HB2B_{}_{}.json'.format(mono, datatime)
        calibrator.write_calibration(CalibName)
        print(calibrator.refinement_summary)
    else:
        calibrator.print_calibration()
        print(calibrator.refinement_summary)
