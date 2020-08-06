# Class providing a series of static methods to work with files
from pyrs.utilities import checkdatatypes
from pyrs.core.instrument_geometry import AnglerCameraDetectorShift
import json


def read_calibration_json_file(calibration_file_name):
    """Import calibration file in json format

    Example:  input JSON
    {u'Lambda': 1.452,
    u'Rot_x': 0.0,
    u'Rot_y': 0.0,
    u'Rot_z': 0.0,
    u'Shift_x': 0.0,
    u'Shift_y': 0.0,
    u'Shift_z': 0.0,
    u'Status': 3,
    u'error_Lambda': 1.0829782933282927e-07,
    u'error_Rot_x': -1.0,
    u'error_Rot_y': -1.0,
    u'error_Rot_z': -1.0,
    u'error_Shift_x': -1.0,
    u'error_Shift_y': -1.0,
    u'error_Shift_z': -1.0}

    Parameters
    ----------
    calibration_file_name

    Returns
    -------
    ~tuple
        (AnglerCameraDetectorShift, AnglerCameraDetectorShift, float, float, int)
        detector position shifts as the calibration result,detector position shifts error from fitting
        status

    """

    # Check input
    checkdatatypes.check_file_name(calibration_file_name, True, False, False, 'Calibration JSON file')

    # Parse JSON file
    with open(calibration_file_name, 'r') as calib_file:
        calib_dict = json.load(calib_file)
    if calib_dict is None:
        raise RuntimeError('Failed to load JSON calibration file {}'.format(calibration_file_name))

    # Convert dictionary to AnglerCameraDetectorShift
    try:
        shift = AnglerCameraDetectorShift(shift_x=calib_dict['Shift_x'],
                                          shift_y=calib_dict['Shift_y'],
                                          shift_z=calib_dict['Shift_z'],
                                          rotation_x=calib_dict['Rot_x'],
                                          rotation_y=calib_dict['Rot_y'],
                                          rotation_z=calib_dict['Rot_z'])
    except KeyError as key_error:
        raise RuntimeError('Missing key parameter from JSON file {}: {}'.format(calibration_file_name, key_error))

    # shift error
    try:
        shift_error = AnglerCameraDetectorShift(shift_x=calib_dict['error_Shift_x'],
                                                shift_y=calib_dict['error_Shift_y'],
                                                shift_z=calib_dict['error_Shift_z'],
                                                rotation_x=calib_dict['error_Rot_x'],
                                                rotation_y=calib_dict['error_Rot_y'],
                                                rotation_z=calib_dict['error_Rot_z'])
    except KeyError as key_error:
        raise RuntimeError('Missing key parameter from JSON file {}: {}'.format(calibration_file_name, key_error))

    # Wave length
    try:
        wave_length = calib_dict['Lambda']
        wave_length_error = calib_dict['error_Lambda']
    except KeyError as key_error:
        raise RuntimeError('Missing wave length related parameter from JSON file {}: {}'
                           ''.format(calibration_file_name, key_error))

    # Calibration status
    try:
        status = calib_dict['Status']
    except KeyError as key_error:
        raise RuntimeError('Missing status parameter from JSON file {}: {}'.format(calibration_file_name, key_error))

    return shift, shift_error, wave_length, wave_length_error, status


def write_calibration_to_json(shifts, shifts_error, wave_length, wave_lenngth_error,
                              calibration_status, file_name=None):
    """Write geometry and wave length calibration to a JSON file

    Parameters
    ----------

    Returns
    -------
    None
    """
    # Check inputs
    checkdatatypes.check_file_name(file_name, False, True, False, 'Output JSON calibration file')
    assert isinstance(shifts, AnglerCameraDetectorShift)
    assert isinstance(shifts_error, AnglerCameraDetectorShift)

    # Create calibration dictionary
    calibration_dict = shifts.convert_to_dict()
    calibration_dict['Lambda'] = wave_length

    calibration_dict.update(shifts_error.convert_error_to_dict())
    calibration_dict['error_Lambda'] = wave_lenngth_error

    calibration_dict.update({'Status': calibration_status})

    print('DICTIONARY:\n{}'.format(calibration_dict))

    with open(file_name, 'w') as outfile:
        json.dump(calibration_dict, outfile)
    print('[INFO] Calibration file is written to {}'.format(file_name))
