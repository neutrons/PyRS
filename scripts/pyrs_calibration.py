from __future__ import (absolute_import, division, print_function)  # python3 compatibility
import numpy as np
# from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode
from pyrs.calibration import peakfit_calibration
from pyrs.core.nexus_conversion import NeXusConvertingApp
from pyrs.core.powder_pattern import ReductionApp

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
POWDER_LINES = []

# Entry error checking
powderlineinput = ['powder lines', 'powder_lines', 'powderlines', 'powder line', 'powder_line', 'powderline']
powerinput = ['powder scan', 'powder', 'powder_scan', 'powderscan', 'powder scans', 'powder_scans', 'powderscans']
pininput = ['pin scan', 'pin', 'pin_scan', 'pinscan', 'pin scans', 'pin_scans', 'pinscans']
calinput = ['calibration', 'old calibration', 'old_calibration']
methodinput = ['method', 'methods']
maskinput = ['mask', 'default mask']


def SaveCalibError(calibrator, fName):
    calibrator.singleEval(ConstrainPosition=True)

    tths = sorted(list(calibrator.ReductionResults.keys()))

    Rois = list(calibrator.ReductionResults[tths[0]].keys())
    DataPoints = len(calibrator.ReductionResults[tths[0]][Rois[0]][0])

    DataOut = np.zeros((DataPoints, 3*len(tths)*len(Rois)))
    header = ''

    lcv = -1
    for i_tth in tths:
        for j in list(calibrator.ReductionResults[i_tth].keys()):
            tempdata = calibrator.ReductionResults[i_tth][j]
            print(i_tth, j)

            lcv += 1
            DataOut[:, lcv*3+0] = tempdata[0]
            DataOut[:, lcv*3+1] = tempdata[1]
            DataOut[:, lcv*3+2] = tempdata[2]
            header += ',pos{}_roi{}_tth,pos{}_roi{}_obs,pos{}_roi{}_calc'.format(i_tth, j[0],
                                                                                 i_tth, j[0], i_tth, j[0])

    DataOut = DataOut[:, :lcv*3+3]

    print(DataOut.shape)
    np.savetxt(fName, DataOut, delimiter=',', header=header[1:])


def _load_nexus_data(ipts, nexus_run, mask_file):
    nexus_file = '/HFIR/HB2B/IPTS-{}/nexus/HB2B_{}.nxs.h5'.format(ipts, nexus_run)
    converter = NeXusConvertingApp(nexus_file, mask_file)
    hidra_ws = converter.convert()
    reducer = ReductionApp(bool('pyrs' == 'mantid'))
    reducer.load_hidra_workspace(hidra_ws)

    return hidra_ws


def _run_calibration(calibrator, calib_method):
    """
    """

    if calib_method == "full":
        calibrator.singlepeak = False
        calibrator.FullCalibration(ConstrainPosition=True)
    elif calib_method == "geometry":
        calibrator.singlepeak = False
        calibrator.CalibrateGeometry(ConstrainPosition=True)
    elif calib_method == "shifts":
        calibrator.singlepeak = False
        calibrator.CalibrateShift(ConstrainPosition=True)
    elif calib_method == "shift x":
        calibrator.singlepeak = True
        calibrator.calibrate_shiftx(ConstrainPosition=True)
    elif calib_method == "shift y":
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
        raise RuntimeError('{} is not a valid calibration method\n{}'.format(M_options))

    return calibrator


def _parse_powder_line(json_entry):
    Input_error = False
    dspace = []
    for item in json_entry:

        if type(item) == str or type(item) == unicode:
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
            raise RuntimeError('{} is not in material list {}'.format(item.split('_')[0].lower(),
                                                                      list(_Materials.keys())))

    return dspace


if __name__ == '__main__':
    import sys
    import json

    if len(sys.argv) == 1:
        raise RuntimeError('Requires a json input.\n{}\n{}\n{}'.format(M_options, P_options))

    with open('test.json', 'r') as json_input:
        try:
            calibration_inputs = json.load(json_input)
        except ValueError:
            raise RuntimeError('Formating error in json input file.\n{}\n{}'.format(_options, M_options, P_options))

    for key in list(calibration_inputs.keys()):
        if key.lower() in powderlineinput:
            POWDER_LINES = _parse_powder_line(calibration_inputs[key])
        elif key.lower() in powerinput:
            POWDER_RUN = calibration_inputs[key]
        elif key.lower() in pininput:
            PIN_RUN = calibration_inputs[key]
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

    if POWDER_RUN is not None:
        POWDER_RUN = _load_nexus_data(IPTS_, POWDER_RUN, DATA_MASK)
    if PIN_RUN is not None:
        PIN_RUN = _load_nexus_data(IPTS_, PIN_RUN, DATA_MASK)

    calibrator = peakfit_calibration.PeakFitCalibration(powder_engine=POWDER_RUN, pin_engine=PIN_RUN,
                                                        powder_lines=POWDER_LINES, single_material=False)

    if INSTRUMENT_CALIBRATION is not None:
        calibrator.get_archived_calibration(INSTRUMENT_CALIBRATION)

    for calib_method in REFINE_METHOD.split('+'):
        calibrator = _run_calibration(calibrator, calib_method)

    print(calibrator._calib)
    # datatime = time.strftime('%Y-%m-%dT%H-%M', time.localtime())
    # fName = '/HFIR/HB2B/shared/CAL/cycle{}/HB2B_{}_{}.json'.format(options.cycle, calibrator.mono, datatime)
#    file_name = os.path.join(os.getcwd(), fName)
    # calibrator.write_calibration(fName)
